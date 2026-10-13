"""Non-adaptive reference forecasters.

These are the rungs the adaptive model has to clear. The weighted moving average
is here deliberately: the CHORD/HELP Logistics 2025 benchmark of humanitarian
demand planning found simple weighted averages hard to beat, so "we beat
seasonal naive" is not on its own a persuasive result.
"""

from __future__ import annotations

import numpy as np

from .base import DEFAULT_QUANTILES, Forecaster
from .panel import Panel


class SeasonalNaive(Forecaster):
    """``y_hat[t] = y[t - season_length]``. The floor."""

    name = "seasonal_naive"

    def fit(self, panel: Panel, end: int) -> None:
        self._season = panel.season_length

    def predict(self, panel: Panel, t: int) -> np.ndarray:
        lag = self._season
        if t - lag < 0:
            return panel.values[max(t - 1, 0)].copy()
        return panel.values[t - lag].copy()


class WeightedMovingAverage(Forecaster):
    """Exponentially weighted mean of the last ``window`` months.

    Optionally multiplied by a seasonal index estimated from training history,
    which is what a competent spreadsheet forecast actually looks like.
    """

    name = "weighted_moving_average"

    def __init__(self, window: int = 6, decay: float = 0.7, seasonal: bool = True):
        self.window = window
        self.decay = decay
        self.seasonal = seasonal
        self._index: np.ndarray | None = None

    def fit(self, panel: Panel, end: int) -> None:
        if not self.seasonal:
            self._index = None
            return
        # Seasonal index per (month, series), normalised to mean 1.
        m = panel.season_length
        hist = panel.values[:end]
        months = panel.month[:end]
        overall = np.maximum(hist.mean(axis=0), 1e-6)
        index = np.ones((m, panel.N))
        for month in range(m):
            mask = months == month
            if mask.sum() >= 2:
                index[month] = hist[mask].mean(axis=0) / overall
        index = np.where(index > 1e-3, index, 1.0)
        self._index = index / index.mean(axis=0, keepdims=True)

    def predict(self, panel: Panel, t: int) -> np.ndarray:
        start = max(0, t - self.window)
        hist = panel.values[start:t]
        if hist.shape[0] == 0:
            return np.zeros(panel.N)
        w = self.decay ** np.arange(hist.shape[0] - 1, -1, -1)
        w = w / w.sum()
        base = (w[:, None] * hist).sum(axis=0)
        if self._index is not None:
            base = base * self._index[panel.month[t]]
        return np.maximum(base, 0.0)


class CrostonSBA(Forecaster):
    """Croston's method with the Syntetos-Boylan approximation.

    Separate exponential smoothing of demand size and inter-arrival interval,
    which is the standard reference for intermittent demand.
    """

    name = "croston_sba"

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._forecast: np.ndarray | None = None

    def _run(self, series: np.ndarray) -> float:
        nz = np.flatnonzero(series > 0)
        if nz.size == 0:
            return 0.0
        size = float(series[nz[0]])
        if nz.size == 1:
            interval = float(max(nz[0] + 1, 1))
        else:
            interval = float(np.diff(nz).mean())
        last = nz[0]
        for idx in nz[1:]:
            size += self.alpha * (series[idx] - size)
            interval += self.alpha * ((idx - last) - interval)
            last = idx
        interval = max(interval, 1.0)
        # SBA bias correction.
        return (1.0 - self.alpha / 2.0) * size / interval

    def fit(self, panel: Panel, end: int) -> None:
        self._refresh(panel, end)

    def _refresh(self, panel: Panel, end: int) -> None:
        hist = panel.values[:end]
        self._forecast = np.array([self._run(hist[:, i]) for i in range(panel.N)])

    def predict(self, panel: Panel, t: int) -> np.ndarray:
        # Croston is a recursive filter; refresh it on the history available now.
        self._refresh(panel, t)
        assert self._forecast is not None
        return np.maximum(self._forecast, 0.0)


def default_baselines() -> list[Forecaster]:
    """The reference ladder used by the demo and the experiment script."""
    return [SeasonalNaive(), WeightedMovingAverage(), CrostonSBA()]
