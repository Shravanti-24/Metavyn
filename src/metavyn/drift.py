"""Drift detection and severity grading for monthly panels.

Classical drift detectors assume a fast stream. Monthly district health data
gives roughly twelve observations a year, which is nowhere near enough to run a
windowed test on a single series. What makes detection viable here is the
*panel*: at month ``t`` there are N simultaneous observations across districts
and indicators, so a cross-sectional statistic has real sample size even though
each series is short.

Three channels are tracked, then graded into one severity score in ``[0, 1]``:

``input``
    Shift of this month's cross-sectional feature mean against the trailing
    twelve months. The reference *rolls*, which matters: against a fixed
    reference a slow upward trend saturates the channel within a couple of
    years and it never recovers, so it stops being a detector at all. A full
    seasonal cycle as reference also removes month-of-year effects.
``residual``
    Tail mean of the standardised forecast residual: the worst
    ``tail_fraction`` of series, not the panel average. Disasters are local, so
    a shock hitting a third of districts is invisible to a mean over all of
    them but obvious in the tail.
``persistent``
    An EWMA of the residual statistic, which separates a one-month spike from a
    sustained regime change.

Severity is the calibrated rank of each statistic against a null distribution
collected during a quiet burn-in, so the thresholds downstream read as
false-alarm rates rather than as raw, unit-dependent numbers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .features import MIN_HISTORY, FeatureBuilder
from .panel import Panel

REFERENCE_WINDOW = 12
TAIL_FRACTION = 0.25
EWMA_ALPHA = 0.35


@dataclass
class DriftSignal:
    """One month's drift reading."""

    t: int
    severity: float
    s_input: float
    s_residual: float
    s_persistent: float
    raw_input: float
    raw_residual: float
    raw_persistent: float
    series_z: np.ndarray | None = None
    """Per-series standardised residual, used to scope a targeted adaptation."""

    def drifting_series(self, z_threshold: float = 1.5) -> np.ndarray | None:
        """Boolean mask of series under *persistent* drift.

        Disasters are local. Knowing *which* series have moved is what lets the
        router adapt only those, instead of dragging a shared coefficient to a
        compromise that helps the affected districts and hurts everyone else.

        ``series_z`` is a smoothed per-series residual, not this month's raw
        value, and that distinction is load-bearing. Selecting on the current
        month picks whichever series happened to be noisy, fits a correction to
        that noise, and is then wrong in the other direction when they regress
        to the mean next month.
        """
        if self.series_z is None:
            return None
        return self.series_z >= z_threshold

    def as_row(self) -> dict[str, float]:
        return {
            "t": self.t,
            "severity": self.severity,
            "s_input": self.s_input,
            "s_residual": self.s_residual,
            "s_persistent": self.s_persistent,
        }


def rank_severity(value: float, null: np.ndarray) -> float:
    """Where ``value`` falls in the calibration null, as a number in ``[0, 1]``.

    Returns roughly 0 for a value typical of quiet months and approaches 1 as it
    exceeds everything seen during calibration. Using the empirical rank rather
    than a raw threshold is what lets one set of router thresholds carry across
    channels measured in completely different units.
    """
    if null.size == 0 or not np.isfinite(value):
        return 0.0
    below = float((null < value).mean())
    if below < 1.0:
        return below
    # Beyond the calibration range: keep ranking on the tail scale so that
    # genuinely extreme months separate from merely unusual ones instead of all
    # pinning at exactly 1.0.
    spread = float(null.max() - np.median(null))
    if spread <= 0:
        return 1.0
    excess = (value - float(null.max())) / spread
    return float(min(1.0, 1.0 - 0.05 * np.exp(-excess)))


@dataclass
class SeverityDetector:
    """Tracks the three channels and grades them into one severity score.

    Args:
        reference_window: months of trailing history used as the input-channel
            reference. One seasonal cycle by default.
        tail_fraction: share of worst-fitting series averaged by the residual
            channel.
        ewma_alpha: smoothing for the persistence channel.
        weight_instant: weight on the instantaneous channels against
            persistence.
    """

    reference_window: int = REFERENCE_WINDOW
    tail_fraction: float = TAIL_FRACTION
    ewma_alpha: float = EWMA_ALPHA
    series_ewma_alpha: float = 0.4
    weight_instant: float = 0.7

    _feature_history: deque = field(default_factory=lambda: deque(maxlen=REFERENCE_WINDOW))
    _resid_scale: np.ndarray | None = None
    _series_ewma: np.ndarray | None = None
    _ewma: float = 0.0
    _null_input: np.ndarray = field(default_factory=lambda: np.empty(0))
    _null_resid: np.ndarray = field(default_factory=lambda: np.empty(0))
    _null_persist: np.ndarray = field(default_factory=lambda: np.empty(0))
    _null_combined: np.ndarray = field(default_factory=lambda: np.empty(0))

    # ------------------------------------------------------------ statistics

    def input_statistic(self, X: np.ndarray) -> float:
        """RMS standardised shift of the feature mean against trailing months.

        Returns 0 until the reference window has filled.
        """
        if len(self._feature_history) < self.reference_window:
            return 0.0
        ref = np.stack(self._feature_history)
        mu = ref.mean(axis=0)
        sd = ref.std(axis=0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        shift = (X.mean(axis=0) - mu) / sd
        return float(np.sqrt(np.mean(shift**2)))

    def residual_statistic(self, residual: np.ndarray) -> float:
        """Mean standardised residual over the worst ``tail_fraction`` series."""
        if self._resid_scale is None:
            return 0.0
        z = np.abs(residual) / self._resid_scale
        k = max(1, int(round(self.tail_fraction * z.size)))
        return float(np.mean(np.sort(z)[-k:]))

    def _push_features(self, X: np.ndarray) -> None:
        self._feature_history.append(X.mean(axis=0))

    # ----------------------------------------------------------- calibration

    def calibrate(
        self,
        panel: Panel,
        builder: FeatureBuilder,
        residuals: dict[int, np.ndarray],
        times: list[int],
    ) -> None:
        """Learn the quiet-period scale and null distributions.

        Args:
            panel: the panel being monitored.
            builder: feature builder, already scale-frozen.
            residuals: held-out log-space residual vectors keyed by month.
            times: calibration months, assumed free of shocks and in order.

        The feature history is warmed on the months immediately before the
        calibration window so that the very first calibration month is scored
        exactly the way a run-time month will be.
        """
        if not times:
            raise ValueError("calibration needs at least one month")
        times = sorted(times)

        resid_stack = np.stack([residuals[t] for t in times])
        # Robust per-series scale: MAD is not pulled around by the one or two
        # odd months a short calibration window is likely to contain.
        mad = np.median(np.abs(resid_stack - np.median(resid_stack, axis=0)), axis=0)
        scale = 1.4826 * mad
        fallback = resid_stack.std(axis=0)
        scale = np.where(scale > 1e-6, scale, np.where(fallback > 1e-6, fallback, 1.0))
        self._resid_scale = scale

        self._feature_history = deque(maxlen=self.reference_window)
        warm_start = max(MIN_HISTORY, times[0] - self.reference_window)
        for t in range(warm_start, times[0]):
            self._push_features(builder.row(panel, t))

        # Pass 1: raw statistics for every calibration month, in order.
        rows: list[tuple[float, bool, float, float]] = []
        self._ewma = 0.0
        for t in times:
            X = builder.row(panel, t)
            input_ready = len(self._feature_history) >= self.reference_window
            s_in = self.input_statistic(X)
            s_re = self.residual_statistic(residuals[t])
            self._ewma = self.ewma_alpha * s_re + (1.0 - self.ewma_alpha) * self._ewma
            rows.append((s_in, input_ready, s_re, self._ewma))
            self._push_features(X)

        self._null_input = np.array([r[0] for r in rows if r[1]])
        self._null_resid = np.array([r[2] for r in rows])
        # The EWMA needs a few months to settle; its early values are not
        # representative of a quiet month and would make the null too wide.
        warm = min(self.reference_window // 2, max(0, len(rows) - 2))
        settled = rows[warm:]
        self._null_persist = np.array([r[3] for r in settled])
        self._ewma = float(np.mean(self._null_persist)) if self._null_persist.size else 0.0

        # Pass 2: calibrate the *combined* score as well. Each channel's rank is
        # uniform over quiet months by construction, but their blend is not, so
        # a threshold on the blend would not mean what it looks like. Ranking
        # the blend against its own quiet-month null makes a threshold read
        # approximately as a false-alarm rate. Only approximately: the null is
        # estimated on calibration months and scored in-sample, and later
        # months differ from them anyway as trend accumulates, so the realised
        # alarm rate on held-out quiet months runs higher than the nominal one.
        # Measure it with `backtest.false_alarm_rate` rather than assuming it.
        self._null_combined = np.array(
            [
                self._combine(
                    rank_severity(s_in, self._null_input) if ready else 0.0,
                    rank_severity(s_re, self._null_resid),
                    rank_severity(s_ph, self._null_persist),
                )
                for s_in, ready, s_re, s_ph in settled
            ]
        )

    def _combine(self, s_in: float, s_re: float, s_ph: float) -> float:
        """Blend the three channel ranks into one raw score."""
        instant = max(s_in, s_re)
        return self.weight_instant * instant + (1.0 - self.weight_instant) * s_ph

    @property
    def calibrated(self) -> bool:
        return self._resid_scale is not None

    # -------------------------------------------------------------- run time

    def update(self, t: int, X: np.ndarray, residual: np.ndarray) -> DriftSignal:
        """Grade month ``t``.

        Args:
            t: month index.
            X: ``(N, d)`` design matrix for this month.
            residual: ``(N,)`` log-space residuals for this month, measured
                against the model that actually produced the forecast.
        """
        raw_in = self.input_statistic(X)
        raw_re = self.residual_statistic(residual)
        self._ewma = self.ewma_alpha * raw_re + (1.0 - self.ewma_alpha) * self._ewma
        raw_ph = self._ewma
        self._push_features(X)
        scale = self._resid_scale if self._resid_scale is not None else 1.0
        instant_z = np.abs(residual) / scale
        if self._series_ewma is None:
            self._series_ewma = instant_z.copy()
        else:
            self._series_ewma = (
                self.series_ewma_alpha * instant_z
                + (1.0 - self.series_ewma_alpha) * self._series_ewma
            )
        series_z = self._series_ewma.copy()

        s_in = rank_severity(raw_in, self._null_input)
        s_re = rank_severity(raw_re, self._null_resid)
        s_ph = rank_severity(raw_ph, self._null_persist)

        severity = rank_severity(self._combine(s_in, s_re, s_ph), self._null_combined)
        return DriftSignal(
            t=t,
            severity=float(np.clip(severity, 0.0, 1.0)),
            s_input=s_in,
            s_residual=s_re,
            s_persistent=s_ph,
            raw_input=raw_in,
            raw_residual=raw_re,
            raw_persistent=raw_ph,
            series_z=series_z,
        )

    def note_adaptation(self, tier: str) -> None:
        """Damp the persistence channel after a structural adaptation.

        Without this the EWMA keeps reporting drift the model has already
        absorbed, and the router pays for the same shock several months running.
        """
        if tier in ("lowrank", "full"):
            target = float(np.mean(self._null_persist)) if self._null_persist.size else 0.0
            self._ewma = target
