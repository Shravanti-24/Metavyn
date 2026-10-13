import numpy as np
import pytest

from metavyn import (
    GlobalRidge,
    NeverAdapt,
    PeriodicRetrain,
    SeasonalNaive,
    SeverityRouter,
    WeightedMovingAverage,
    default_scenario,
    run_adaptive,
    run_baseline,
)
from metavyn.drift import DriftSignal
from metavyn.features import FeatureBuilder, FeatureSpec
from metavyn.panel import Panel, SeriesKey
from metavyn.ridge import TIERS
from metavyn.synthetic import make_panel


# ----------------------------------------------------------------- panel


def test_panel_rejects_mismatched_series_and_exog():
    values = np.zeros((5, 2))
    keys = [SeriesKey("A", "x"), SeriesKey("B", "x")]
    with pytest.raises(ValueError, match="series keys"):
        Panel(values=values, series=keys[:1])
    with pytest.raises(ValueError, match="exog"):
        Panel(values=values, series=keys, exog={"rain": np.zeros((4, 2))})


def test_intermittency_classes_follow_zero_fraction():
    values = np.zeros((10, 3))
    values[:, 0] = 5.0  # never zero -> dense
    values[:3, 1] = 5.0  # 70% zeros -> lumpy
    values[:9, 2] = 5.0  # 10% zeros -> dense boundary
    panel = Panel(values=values, series=[SeriesKey("A", str(i)) for i in range(3)])
    classes = panel.intermittency_class()
    assert list(classes) == ["dense", "lumpy", "dense"]


# -------------------------------------------------------------- features


def test_features_never_read_the_present_or_future():
    panel, _ = default_scenario(seed=0)
    builder = FeatureBuilder(spec=FeatureSpec(use_current_exog=False))
    builder.freeze_scale(panel, 40)
    row = builder.row(panel, 30)

    poisoned = Panel(
        values=panel.values.copy(),
        series=list(panel.series),
        exog={k: v.copy() for k, v in panel.exog.items()},
        month=panel.month.copy(),
    )
    poisoned.values[30:] = 1e6
    for arr in poisoned.exog.values():
        arr[30:] = 1e6
    poisoned_builder = FeatureBuilder(spec=FeatureSpec(use_current_exog=False))
    poisoned_builder.series_scale = builder.series_scale
    np.testing.assert_allclose(row, poisoned_builder.row(poisoned, 30))


def test_horizon_shifts_the_forecast_origin_back():
    panel, _ = default_scenario(seed=0)
    builder = FeatureBuilder()
    builder.freeze_scale(panel, 40)
    far = builder.row(panel, 30, horizon=3)
    near = builder.row(panel, 28, horizon=1)
    names = builder.spec.names
    history = [i for i, n in enumerate(names) if n.startswith(("lag", "roll", "rainfall"))]
    seasonal = [i for i, n in enumerate(names) if n.startswith(("sin", "cos"))]

    # History and exogenous features come from the same origin...
    np.testing.assert_allclose(far[:, history], near[:, history])
    # ...but month-of-year is a property of the month being predicted, which is
    # known in advance and so must not be shifted back with the origin.
    assert not np.allclose(far[:, seasonal], near[:, seasonal])


# ----------------------------------------------------------------- model


def test_fitted_model_beats_seasonal_naive_on_held_out_months():
    panel, _ = default_scenario(seed=0)
    ridge = run_adaptive(panel, GlobalRidge(), NeverAdapt(), 72)
    naive = run_baseline(panel, SeasonalNaive(), 72)
    assert ridge.report.mase < naive.report.mase


def test_mean_point_estimate_removes_the_downward_bias():
    panel, _ = default_scenario(seed=0)
    median = run_adaptive(panel, GlobalRidge(), NeverAdapt(), 72)
    mean = run_adaptive(panel, GlobalRidge(point_estimate="mean"), NeverAdapt(), 72)
    # Back-transforming a log-space fit gives the median, which sits below the
    # mean for skewed counts; smearing corrects the direction.
    assert median.report.signed_error < mean.report.signed_error
    assert abs(mean.report.signed_error) < abs(median.report.signed_error)


def test_unknown_tier_is_rejected():
    panel, _ = default_scenario(seed=0)
    model = GlobalRidge()
    model.fit(panel, 72)
    with pytest.raises(ValueError, match="unknown tier"):
        model.adapt(panel, 80, "turbo")


@pytest.mark.parametrize("tier", TIERS)
def test_every_tier_runs_and_reports_its_cost(tier):
    panel, _ = default_scenario(seed=0)
    model = GlobalRidge()
    model.fit(panel, 72)
    affected = np.zeros(panel.N, dtype=bool)
    affected[:10] = True
    before = model.predict(panel, 80).copy()
    result = model.adapt(panel, 80, tier, affected=affected)

    assert result.tier == tier
    assert result.n_params_updated >= 0
    assert np.all(np.isfinite(model.predict(panel, 80)))
    if tier == "none":
        np.testing.assert_allclose(before, model.predict(panel, 80))
        assert result.n_params_updated == 0
    else:
        assert result.n_params_updated > 0


def test_scoped_tier_leaves_unflagged_series_untouched():
    panel, _ = default_scenario(seed=0)
    model = GlobalRidge()
    model.fit(panel, 72)
    affected = np.zeros(panel.N, dtype=bool)
    affected[:8] = True
    before = model.predict(panel, 84).copy()
    model.adapt(panel, 84, "local_lowrank", affected=affected)
    after = model.predict(panel, 84)
    np.testing.assert_allclose(before[~affected], after[~affected])
    assert not np.allclose(before[affected], after[affected])


def test_full_refit_clears_any_scoped_correction():
    panel, _ = default_scenario(seed=0)
    model = GlobalRidge()
    model.fit(panel, 72)
    affected = np.ones(panel.N, dtype=bool)
    model.adapt(panel, 84, "local_lowrank", affected=affected)
    assert model.delta_local is not None
    model.adapt(panel, 85, "full")
    assert model.delta_local is None and model.local_mask is None


# -------------------------------------------------------------- policies


def test_periodic_policy_fires_on_its_cycle():
    policy = PeriodicRetrain(period=3, tier="full")
    signal = DriftSignal(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert [policy.route(signal) for _ in range(6)] == [
        "none",
        "none",
        "full",
        "none",
        "none",
        "full",
    ]


def test_severity_router_respects_the_cooldown():
    policy = SeverityRouter(cooldown=3)
    extreme = lambda t: DriftSignal(t, 0.999, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert policy.route(extreme(10)) == "full"
    # A second extreme month inside the cooldown drops a rung instead of
    # paying for another full refit.
    assert policy.route(extreme(11)) == "local_lowrank"
    assert policy.route(extreme(20)) == "full"


def test_router_returns_none_below_every_threshold():
    policy = SeverityRouter()
    assert policy.route(DriftSignal(1, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0)) == "none"


# -------------------------------------------------------------- backtest


def test_backtest_covers_the_holdout_and_records_one_decision_per_month():
    panel, _ = default_scenario(seed=2)
    result = run_adaptive(panel, GlobalRidge(), SeverityRouter(), 72)
    assert result.times == list(range(72, panel.T))
    assert len(result.tiers) == len(result.times) == len(result.signals)
    assert result.forecasts.shape == result.actuals.shape == (panel.T - 72, panel.N)
    np.testing.assert_allclose(result.actuals, panel.values[72:])


def test_frozen_policy_writes_no_parameters():
    panel, _ = default_scenario(seed=0)
    result = run_adaptive(panel, GlobalRidge(), NeverAdapt(), 72)
    assert result.total_params_updated == 0
    assert result.n_adaptations == 0


def test_quantile_forecasts_are_ordered_and_cover_sensibly():
    panel, _ = default_scenario(seed=0)
    result = run_adaptive(panel, GlobalRidge(), NeverAdapt(), 72)
    q = result.quantiles
    assert np.all(np.diff(q, axis=-1) >= -1e-9)  # non-decreasing in level
    assert 0.5 < result.report.coverage_80 < 0.98


def test_baselines_get_predictive_intervals_too():
    panel, _ = default_scenario(seed=0)
    result = run_baseline(panel, WeightedMovingAverage(), 72)
    # Otherwise any probabilistic comparison would be rigged in the adaptive
    # model's favour.
    assert result.report.coverage_80 is not None
    assert result.report.coverage_80 > 0.4


def test_generator_places_shocks_where_it_says_it_does():
    from metavyn.synthetic import Shock

    panel, truth = make_panel(
        n_periods=48,
        shocks=(Shock(start=24, districts=("D00",), duration=4, name="flood"),),
        seed=0,
    )
    assert truth["shock_months"][24]
    assert not truth["shock_months"][23]
    hit = np.array([k.district == "D00" for k in panel.series])
    response = truth["shock_by_series"]
    assert response[24][hit].mean() > 0
    assert response[24][~hit].sum() == 0
