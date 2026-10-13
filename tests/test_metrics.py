import numpy as np
import pytest

from metavyn.metrics import (
    evaluate,
    interval_coverage,
    mase,
    mean_signed_error,
    pinball_loss,
    seasonal_naive_scale,
    share_biased_high,
    smape,
)


def test_seasonal_naive_scale_is_mean_absolute_seasonal_difference():
    # Two series: one flat, one stepping by 2 every period.
    train = np.column_stack([np.ones(26), np.arange(26) * 2.0])
    scale = seasonal_naive_scale(train, season_length=12)
    assert scale[0] == 1.0  # flat series guarded away from zero
    assert scale[1] == pytest.approx(24.0)


def test_mase_is_one_when_error_equals_scale():
    y = np.zeros((4, 2))
    pred = np.full((4, 2), 3.0)
    scale = np.array([3.0, 3.0])
    assert mase(y, pred, scale) == pytest.approx(1.0)


def test_smape_handles_zero_demand_without_blowing_up():
    y = np.array([[0.0, 0.0]])
    pred = np.array([[0.0, 5.0]])
    out = smape(y, pred)
    assert np.isfinite(out)
    # Exact zero match contributes 0; a pure over-forecast contributes 200%.
    assert out == pytest.approx(100.0)


def test_signed_error_and_share_high_detect_over_forecasting():
    y = np.ones((5, 4))
    pred = y + 2.0
    scale = np.ones(4)
    assert mean_signed_error(y, pred, scale) == pytest.approx(2.0)
    assert share_biased_high(y, pred) == 1.0
    assert share_biased_high(y, y - 1.0) == 0.0


def test_signed_error_is_zero_for_balanced_errors():
    y = np.zeros((2, 2))
    pred = np.array([[1.0, -1.0], [1.0, -1.0]])
    assert mean_signed_error(y, pred, np.ones(2)) == pytest.approx(0.0)
    # ...but half the series are still biased high, which share_high reports.
    assert share_biased_high(y, pred) == pytest.approx(0.5)


def test_pinball_loss_penalises_asymmetrically():
    y = np.array([[10.0]])
    quantiles = np.array([0.9])
    under = pinball_loss(y, np.array([[[5.0]]]), quantiles)
    over = pinball_loss(y, np.array([[[15.0]]]), quantiles)
    # At q=0.9 under-forecasting should cost more than over-forecasting.
    assert under > over


def test_interval_coverage_counts_inclusively():
    y = np.array([[1.0, 5.0]])
    lower = np.array([[1.0, 0.0]])
    upper = np.array([[2.0, 4.0]])
    assert interval_coverage(y, lower, upper) == pytest.approx(0.5)


def test_evaluate_segments_by_class():
    rng = np.random.default_rng(0)
    y = rng.poisson(10, size=(20, 4)).astype(float)
    pred = y + 1.0
    scale = np.ones(4)
    classes = np.array(["dense", "dense", "lumpy", "lumpy"])
    report = evaluate(y, pred, scale, classes=classes)
    assert set(report.by_class) == {"dense", "lumpy"}
    assert report.by_class["dense"].n_obs == 40
    # Uniform +1 over-forecast must show up in every segment.
    assert all(sub.signed_error > 0 for sub in report.by_class.values())
