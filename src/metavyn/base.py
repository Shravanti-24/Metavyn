"""Forecaster interface shared by baselines and the adaptive model."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from .panel import Panel

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

# Trailing errors kept for empirical predictive intervals.
RESIDUAL_WINDOW = 24
MIN_RESIDUALS_FOR_QUANTILES = 6


@dataclass
class AdaptationResult:
    """What one adaptation step cost and touched.

    Attributes:
        tier: name of the tier applied.
        n_params_updated: parameters actually written. This is the cost axis the
            whole severity-routing argument turns on, so it is measured rather
            than assumed.
        wall_seconds: time spent adapting.
        note: optional detail for the run log.
    """

    tier: str
    n_params_updated: int
    wall_seconds: float
    note: str = ""


class Forecaster(abc.ABC):
    """A panel forecaster that predicts every series one month ahead."""

    name: str = "forecaster"

    @abc.abstractmethod
    def fit(self, panel: Panel, end: int) -> None:
        """Fit on ``panel.values[:end]``."""

    @abc.abstractmethod
    def predict(self, panel: Panel, t: int) -> np.ndarray:
        """One-step point forecast for time ``t``, using data before ``t``.

        Returns:
            ``(N,)`` non-negative forecasts.
        """

    def _residual_buffer(self) -> list[np.ndarray]:
        buf = getattr(self, "_level_residuals", None)
        if buf is None:
            buf = []
            self._level_residuals = buf
        return buf

    def predict_quantiles(
        self, panel: Panel, t: int, quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    ) -> np.ndarray:
        """Quantile forecasts for time ``t``.

        The default adds empirical quantiles of this model's own recent errors
        to its point forecast. Every forecaster gets this, so pinball loss and
        interval coverage compare models rather than rewarding whichever one
        happened to be given an uncertainty estimate.

        Returns:
            ``(N, Q)`` forecasts.
        """
        point = self.predict(panel, t)
        buf = self._residual_buffer()
        if len(buf) < MIN_RESIDUALS_FOR_QUANTILES:
            return np.repeat(point[:, None], len(quantiles), axis=1)
        resid = np.stack(buf[-RESIDUAL_WINDOW:])
        offsets = np.quantile(resid, quantiles, axis=0).T  # (N, Q)
        return np.maximum(point[:, None] + offsets, 0.0)

    def observe(self, panel: Panel, t: int) -> None:
        """Record the realised error at ``t``, once the truth is available."""
        buf = self._residual_buffer()
        buf.append(panel.values[t] - self.predict(panel, t))
        if len(buf) > 4 * RESIDUAL_WINDOW:
            del buf[:-RESIDUAL_WINDOW]

    def adapt(
        self,
        panel: Panel,
        t: int,
        tier: str,
        affected: np.ndarray | None = None,
    ) -> AdaptationResult:
        """Apply an adaptation tier. Static models ignore this."""
        return AdaptationResult(tier="none", n_params_updated=0, wall_seconds=0.0)

    @property
    def n_parameters(self) -> int:
        """Total trainable parameters, for cost accounting."""
        return 0
