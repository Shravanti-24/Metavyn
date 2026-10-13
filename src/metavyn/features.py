"""Design-matrix construction for the global forecaster.

Every series shares one feature layout so a single pooled model can be fitted
across the whole panel; per-series identity enters through a scale feature and a
per-series intercept rather than through separate models. Targets and count
features are modelled in ``log1p`` space, which keeps forecasts non-negative and
turns the multiplicative seasonality of the generator into an additive effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .panel import Panel

MIN_HISTORY = 13  # longest lag (12) plus one


@dataclass(frozen=True)
class FeatureSpec:
    """Which features the design matrix carries.

    Attributes:
        lags: autoregressive lags, in months.
        rolling_windows: trailing means to include, in months.
        harmonics: number of Fourier harmonics for month-of-year.
        exog_names: exogenous channels to include.
        use_current_exog: include the current month's exogenous value.
            Rainfall for the month in progress is observable in practice, but it
            is the one feature that is not strictly lagged, so it is switchable
            for ablations.
    """

    lags: tuple[int, ...] = (1, 2, 3, 12)
    rolling_windows: tuple[int, ...] = (3, 12)
    harmonics: int = 2
    exog_names: tuple[str, ...] = ("rainfall",)
    use_current_exog: bool = True

    @property
    def names(self) -> list[str]:
        """Human-readable feature names, in column order."""
        out = [f"lag{l}" for l in self.lags]
        out += [f"roll{w}" for w in self.rolling_windows]
        for h in range(1, self.harmonics + 1):
            out += [f"sin{h}", f"cos{h}"]
        for name in self.exog_names:
            if self.use_current_exog:
                out.append(f"{name}_t")
            out.append(f"{name}_lag1")
        out.append("series_scale")
        out.append("bias")
        return out

    @property
    def n_features(self) -> int:
        return len(self.names)


@dataclass
class FeatureBuilder:
    """Builds design matrices for a fixed panel and spec.

    ``series_scale`` is frozen at fit time from the training window so it cannot
    leak information from the evaluation period.
    """

    spec: FeatureSpec = field(default_factory=FeatureSpec)
    series_scale: np.ndarray | None = None

    def freeze_scale(self, panel: Panel, end: int) -> None:
        """Record per-series level from ``values[:end]``."""
        window = panel.values[:end]
        level = window.mean(axis=0) if window.shape[0] else np.ones(panel.N)
        self.series_scale = np.log1p(np.maximum(level, 0.0))

    def row(self, panel: Panel, t: int, horizon: int = 1) -> np.ndarray:
        """Features for predicting every series at time ``t``.

        Args:
            panel: the data.
            t: the month being predicted.
            horizon: how many months ahead the forecast is made. The forecast
                origin is ``t - horizon + 1``, and no observation at or after
                that origin is read, so a horizon-3 forecast for month ``t``
                genuinely only knows what was available three months earlier.

        Returns:
            ``(N, d)`` design matrix.
        """
        origin = t - horizon + 1
        if origin < MIN_HISTORY:
            raise ValueError(
                f"t={t} at horizon {horizon} has insufficient history "
                f"(need origin >= {MIN_HISTORY})"
            )
        if self.series_scale is None:
            raise RuntimeError("call freeze_scale() before building features")

        # Lags and rolling means enter as deviations from the series' own level.
        # Without this the pooled regression spends nearly all its capacity
        # mapping each series to its own scale, the autoregressive and
        # exogenous coefficients come out near zero, and adapting the shared
        # weights changes the forecasts hardly at all.
        hist = np.log1p(np.maximum(panel.values[:origin], 0.0))
        scale = self.series_scale
        cols: list[np.ndarray] = [hist[origin - lag] - scale for lag in self.spec.lags]
        cols += [
            hist[origin - w : origin].mean(axis=0) - scale
            for w in self.spec.rolling_windows
        ]

        m = panel.month[t]
        for h in range(1, self.spec.harmonics + 1):
            angle = 2.0 * np.pi * h * m / panel.season_length
            cols.append(np.full(panel.N, np.sin(angle)))
            cols.append(np.full(panel.N, np.cos(angle)))

        for name in self.spec.exog_names:
            ex = np.log1p(np.maximum(panel.exog[name], 0.0))
            if self.spec.use_current_exog:
                cols.append(ex[origin])
            cols.append(ex[origin - 1])

        cols.append(self.series_scale)
        cols.append(np.ones(panel.N))
        return np.stack(cols, axis=1)

    def target(self, panel: Panel, t: int) -> np.ndarray:  # noqa: D401
        """Centred target at ``t``: ``log1p(y) - series_scale``."""
        if self.series_scale is None:
            raise RuntimeError("call freeze_scale() before building targets")
        return np.log1p(np.maximum(panel.values[t], 0.0)) - self.series_scale

    def block(
        self, panel: Panel, start: int, end: int, horizon: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stack design rows and targets over ``[start, end)``.

        Returns:
            ``(X, y, series_index)`` where ``X`` is ``(M, d)``, ``y`` is ``(M,)``
            centred log-space targets and ``series_index`` is ``(M,)`` column
            ids.
        """
        start = max(start, MIN_HISTORY + horizon - 1)
        xs, ys, idx = [], [], []
        for t in range(start, end):
            xs.append(self.row(panel, t, horizon))
            ys.append(self.target(panel, t))
            idx.append(np.arange(panel.N))
        if not xs:
            d = self.spec.n_features
            return np.empty((0, d)), np.empty(0), np.empty(0, dtype=int)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(idx)
