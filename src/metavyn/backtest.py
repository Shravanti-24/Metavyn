"""Rolling-origin backtest driver.

One month at a time: forecast, observe the truth, grade the drift, let the
policy choose a tier, adapt, move on. Nothing in the loop may read past ``t``,
which is what keeps the comparison between policies honest.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .base import DEFAULT_QUANTILES, Forecaster
from .drift import DriftSignal, SeverityDetector
from .features import MIN_HISTORY
from .metrics import ErrorReport, evaluate, seasonal_naive_scale
from .panel import Panel
from .ridge import GlobalRidge
from .router import Policy

DEFAULT_CALIBRATION_MONTHS = 36
MIN_CALIBRATION_MONTHS = 12
MIN_HOLDOUT_MONTHS = 6  # months the held-out model needs to fit on


@dataclass
class RunResult:
    """Everything one policy did over the evaluation window."""

    label: str
    times: list[int]
    actuals: np.ndarray
    forecasts: np.ndarray
    quantiles: np.ndarray
    report: ErrorReport
    signals: list[DriftSignal] = field(default_factory=list)
    tiers: list[str] = field(default_factory=list)
    params_updated: list[int] = field(default_factory=list)
    adapt_seconds: float = 0.0

    @property
    def total_params_updated(self) -> int:
        return int(sum(self.params_updated))

    @property
    def tier_counts(self) -> dict[str, int]:
        return dict(Counter(self.tiers))

    @property
    def n_adaptations(self) -> int:
        return sum(1 for tier in self.tiers if tier != "none")

    def summary_row(self) -> dict[str, object]:
        """One line for the comparison table."""
        row: dict[str, object] = {"policy": self.label}
        row.update(self.report.to_row())
        row["adaptations"] = self.n_adaptations
        row["params_updated"] = self.total_params_updated
        row["adapt_s"] = round(self.adapt_seconds, 3)
        return row

    def errors_by_month(self, scale: np.ndarray) -> np.ndarray:
        """Per-month MASE, used to measure recovery after a shock."""
        err = np.abs(self.forecasts - self.actuals) / scale[None, :]
        return err.mean(axis=1)


def run_baseline(
    panel: Panel,
    forecaster: Forecaster,
    train_end: int,
    *,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    label: str | None = None,
) -> RunResult:
    """Backtest a non-adaptive forecaster."""
    forecaster.fit(panel, train_end)
    times = list(range(train_end, panel.T))
    preds, qpreds = [], []
    for t in times:
        preds.append(forecaster.predict(panel, t))
        qpreds.append(forecaster.predict_quantiles(panel, t, quantiles))
        forecaster.observe(panel, t)

    actuals = np.stack([panel.values[t] for t in times])
    forecasts = np.stack(preds)
    qarr = np.stack(qpreds)
    scale = seasonal_naive_scale(panel.values[:train_end], panel.season_length)
    report = evaluate(
        actuals,
        forecasts,
        scale,
        quantile_pred=qarr,
        quantiles=np.asarray(quantiles),
        classes=panel.intermittency_class(train_end),
    )
    return RunResult(
        label=label or forecaster.name,
        times=times,
        actuals=actuals,
        forecasts=forecasts,
        quantiles=qarr,
        report=report,
        tiers=["none"] * len(times),
        params_updated=[0] * len(times),
    )


def calibrate_detector(
    panel: Panel,
    forecaster: GlobalRidge,
    detector: SeverityDetector,
    train_end: int,
    calibration_months: int = DEFAULT_CALIBRATION_MONTHS,
) -> list[int]:
    """Fit the detector's null distributions on quiet, held-out months.

    The null must describe how big a residual looks on data the model has *not*
    seen. Calibrating on in-sample residuals makes the null far too tight, and
    the detector then flags ordinary out-of-sample months as drift. So a clone
    of the model is fitted on everything before the calibration window and
    scored across it, while the model that goes on to run keeps the benefit of
    the full training set.

    Returns:
        The months used for calibration.
    """
    # The window has to leave enough history behind it to fit the held-out
    # model. On a short panel it shrinks to fit rather than failing, because a
    # real HMIS extract is short and a coarser null is still better than none.
    # What it costs is resolution: with M quiet months the severity rank moves
    # in steps of 1/M, so the false-alarm rate a threshold implies gets
    # correspondingly blunter.
    usable = train_end - (MIN_HISTORY + MIN_HOLDOUT_MONTHS)
    if usable < MIN_CALIBRATION_MONTHS:
        raise ValueError(
            f"train_end={train_end} leaves only {max(usable, 0)} months to calibrate "
            f"on; need at least {MIN_CALIBRATION_MONTHS} after reserving "
            f"{MIN_HISTORY} for lags and {MIN_HOLDOUT_MONTHS} to fit the held-out model"
        )
    months = min(calibration_months, usable)
    start = train_end - months
    times = list(range(start, train_end))

    holdout = forecaster.clone()
    holdout.fit(panel, start)
    residuals = {
        t: np.log1p(np.maximum(panel.values[t], 0.0)) - holdout._predict_log(panel, t)
        for t in times
    }
    # Features come from the builder used at run time (the input channel does
    # not depend on the fit); only the residuals come from the held-out model.
    detector.calibrate(panel, forecaster.builder, residuals, times)
    return times


def run_adaptive(
    panel: Panel,
    forecaster: GlobalRidge,
    policy: Policy,
    train_end: int,
    *,
    detector: SeverityDetector | None = None,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    label: str | None = None,
    calibration_months: int = DEFAULT_CALIBRATION_MONTHS,
) -> RunResult:
    """Backtest an adaptive forecaster under one policy.

    Args:
        panel: the data.
        forecaster: a fresh, unfitted model (it is mutated in place).
        policy: the adaptation policy under test.
        train_end: first evaluation month; everything before is training.
        detector: severity detector, created and calibrated if omitted.
        quantiles: quantile levels to score.
        label: name for the results table.
        calibration_months: in-sample months used for the drift null.
    """
    forecaster.fit(panel, train_end)
    policy.reset()
    detector = detector or SeverityDetector()
    calibrate_detector(panel, forecaster, detector, train_end, calibration_months)

    times = list(range(train_end, panel.T))
    preds, qpreds, signals, tiers, costs = [], [], [], [], []
    adapt_seconds = 0.0

    for t in times:
        point = forecaster.predict(panel, t)
        qpred = forecaster.predict_quantiles(panel, t, quantiles)
        preds.append(point)
        qpreds.append(qpred)

        # The truth for t arrives; residual is measured against the model that
        # actually made the forecast, before any adaptation.
        actual_log = np.log1p(np.maximum(panel.values[t], 0.0))
        resid = actual_log - forecaster._predict_log(panel, t)
        forecaster.observe(panel, t)

        X = forecaster.builder.row(panel, t)
        signal = detector.update(t, X, resid)
        tier = policy.route(signal)
        result = forecaster.adapt(panel, t, tier, affected=signal.drifting_series())
        detector.note_adaptation(tier)

        signals.append(signal)
        tiers.append(result.tier)
        costs.append(result.n_params_updated)
        adapt_seconds += result.wall_seconds

    actuals = np.stack([panel.values[t] for t in times])
    forecasts = np.stack(preds)
    qarr = np.stack(qpreds)
    scale = seasonal_naive_scale(panel.values[:train_end], panel.season_length)
    report = evaluate(
        actuals,
        forecasts,
        scale,
        quantile_pred=qarr,
        quantiles=np.asarray(quantiles),
        classes=panel.intermittency_class(train_end),
    )
    return RunResult(
        label=label or policy.name,
        times=times,
        actuals=actuals,
        forecasts=forecasts,
        quantiles=qarr,
        report=report,
        signals=signals,
        tiers=tiers,
        params_updated=costs,
        adapt_seconds=adapt_seconds,
    )


def detection_latency(
    signals: list[DriftSignal], shock_start: int, threshold: float = 0.85
) -> int | None:
    """Months from a shock to the first severity reading above ``threshold``.

    Returns ``None`` if the shock was never flagged.
    """
    for signal in signals:
        if signal.t >= shock_start and signal.severity >= threshold:
            return signal.t - shock_start
    return None


def false_alarm_rate(
    signals: list[DriftSignal],
    drifted_months: np.ndarray,
    threshold: float = 0.90,
) -> float:
    """Share of genuinely quiet months whose severity crosses ``threshold``.

    This is the cost side of the detector. A router that fires constantly will
    look good on error while spending as much as retraining every month, so the
    alarm rate on quiet months is reported next to accuracy, not instead of it.

    Args:
        signals: the run's drift readings.
        drifted_months: boolean mask over panel months where drift is genuinely
            present. This must cover every month after a *permanent* regime
            change, not just the shock itself: once the mapping has moved, an
            alarm is correct, and counting those months as quiet would make the
            detector look far worse than it is.
        threshold: severity level counted as an alarm.
    """
    quiet = [s for s in signals if not drifted_months[s.t]]
    if not quiet:
        return float("nan")
    return float(np.mean([s.severity >= threshold for s in quiet]))


def recovery_months(
    result: RunResult,
    scale: np.ndarray,
    shock_start: int,
    baseline_error: float,
    tolerance: float = 1.15,
) -> int | None:
    """Months until per-month error falls back near its pre-shock level.

    Args:
        result: the run to measure.
        scale: MASE denominators.
        shock_start: month the shock began.
        baseline_error: pre-shock mean per-month MASE.
        tolerance: multiple of ``baseline_error`` counted as recovered.
    """
    per_month = result.errors_by_month(scale)
    target = baseline_error * tolerance
    seen_shock = False
    for t, err in zip(result.times, per_month):
        if t < shock_start:
            continue
        if not seen_shock:
            seen_shock = True
        if err <= target:
            return t - shock_start
    return None
