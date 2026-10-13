import numpy as np
import pytest

from metavyn.backtest import calibrate_detector
from metavyn.drift import SeverityDetector, rank_severity
from metavyn.ridge import GlobalRidge
from metavyn.synthetic import default_scenario, regime_scenario


def test_rank_severity_is_the_empirical_cdf():
    null = np.arange(10.0)
    assert rank_severity(-1.0, null) == 0.0
    assert rank_severity(5.0, null) == pytest.approx(0.5)
    # Beyond the calibration range it saturates toward 1 without exceeding it.
    assert 0.9 < rank_severity(1000.0, null) <= 1.0


def test_rank_severity_handles_empty_and_degenerate_nulls():
    assert rank_severity(1.0, np.empty(0)) == 0.0
    assert rank_severity(5.0, np.ones(4)) == 1.0


def test_input_channel_is_silent_until_reference_fills():
    det = SeverityDetector()
    X = np.ones((5, 3))
    assert det.input_statistic(X) == 0.0


def test_residual_channel_tracks_the_tail_not_the_mean():
    det = SeverityDetector(tail_fraction=0.25)
    det._resid_scale = np.ones(8)
    quiet = np.zeros(8)
    # Two series blow up, six are fine: the mean barely moves, the tail does.
    localised = np.array([10.0, 10.0, 0, 0, 0, 0, 0, 0])
    assert det.residual_statistic(quiet) == 0.0
    assert det.residual_statistic(localised) == pytest.approx(10.0)


def test_calibrated_detector_separates_shock_from_quiet_months():
    panel, truth = default_scenario(seed=3)
    model = GlobalRidge()
    model.fit(panel, 72)
    det = SeverityDetector()
    calibrate_detector(panel, model, det, 72)

    severities = {}
    for t in range(72, panel.T):
        resid = np.log1p(np.maximum(panel.values[t], 0.0)) - model._predict_log(panel, t)
        severities[t] = det.update(t, model.builder.row(panel, t), resid).severity

    shock_start = truth["shocks"][0].start
    quiet = [s for t, s in severities.items() if not truth["shock_months"][t]]
    assert severities[shock_start] > np.mean(quiet)
    # A calibrated rank should not fire on most quiet months.
    assert np.mean([s >= 0.88 for s in quiet]) < 0.35


def test_drifting_series_flags_the_districts_that_were_hit():
    panel, truth = regime_scenario(seed=0)
    model = GlobalRidge()
    model.fit(panel, 72)
    det = SeverityDetector()
    calibrate_detector(panel, model, det, 72)

    hit = set(truth["shocks"][0].districts)
    affected = np.array([k.district in hit for k in panel.series])
    signal = None
    for t in range(72, 92):
        resid = np.log1p(np.maximum(panel.values[t], 0.0)) - model._predict_log(panel, t)
        signal = det.update(t, model.builder.row(panel, t), resid)

    flags = signal.drifting_series()
    assert flags is not None
    # Flagging should concentrate on the struck districts rather than spread
    # evenly across the panel.
    assert flags[affected].mean() > flags[~affected].mean()


def test_calibration_window_shrinks_to_fit_a_short_panel():
    panel, _ = default_scenario(seed=0)
    model = GlobalRidge()
    model.fit(panel, 40)
    # Asking for more months than exist should give back what fits, not fail:
    # a real HMIS extract is short, and a coarse null still beats none.
    times = calibrate_detector(panel, model, SeverityDetector(), 40, calibration_months=40)
    assert len(times) < 40
    assert min(times) >= 13 and max(times) == 39


def test_calibration_fails_when_the_panel_is_genuinely_too_short():
    panel, _ = default_scenario(seed=0)
    model = GlobalRidge()
    model.fit(panel, 25)
    with pytest.raises(ValueError, match="months to calibrate on"):
        calibrate_detector(panel, model, SeverityDetector(), 25)
