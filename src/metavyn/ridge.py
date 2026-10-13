"""Global ridge forecaster with a parameter-efficient adaptation ladder.

This is the MVP backbone. It is deliberately not a transformer: the claim under
test is about *routing drift severity to depth of adaptation*, and that claim is
backbone-agnostic. A closed-form linear model makes the routing loop fast,
deterministic and testable, and :mod:`metavyn.torch_backbone` slots a transformer
with real LoRA adapters into the same interface when torch is available.

The adaptation ladder mirrors parameter-efficient fine-tuning:

===========  =============================================  ==================
Tier         What it updates                                Parameters touched
===========  =============================================  ==================
``none``     nothing                                        0
``bias``     per-series intercepts only                     N
``lowrank``  a rank-r subspace of the shared weights        r
``full``     all shared weights, with replay                d + N
===========  =============================================  ==================

The low-rank tier restricts the weight update to a fixed subspace spanned by the
top-r principal directions of the training design matrix, which is the same
device LoRA uses: a cheap, structured correction instead of a full re-solve.
"""

from __future__ import annotations

import time

import numpy as np

from .base import DEFAULT_QUANTILES, AdaptationResult, Forecaster
from .features import MIN_HISTORY, FeatureBuilder, FeatureSpec
from .panel import Panel

TIERS = ("none", "bias", "local_lowrank", "lowrank", "full")


class GlobalRidge(Forecaster):
    """Pooled ridge regression in ``log1p`` space with per-series intercepts.

    Args:
        spec: feature layout.
        ridge: L2 penalty for the shared weights.
        rank: dimension of the low-rank adaptation subspace.
        adapt_window: months of recent data an adaptation step fits on.
        replay_size: training rows retained and mixed into a full refit, which
            is what stops the model forgetting normal seasonality after a shock.
        bias_shrinkage: shrinks per-series intercepts toward zero; with ~12
            observations per series a raw mean is noisy.
        residual_window: months of residuals kept for quantile estimation.
    """

    name = "metavyn_ridge"

    def __init__(
        self,
        spec: FeatureSpec | None = None,
        ridge: float = 1.0,
        rank: int = 4,
        adapt_window: int = 12,
        replay_size: int = 4000,
        bias_shrinkage: float = 4.0,
        residual_window: int = 24,
        point_estimate: str = "median",
        horizon: int = 1,
        local_ridge: float = 8.0,
        local_damping: float = 0.6,
        bias_prior_weight: float = 24.0,
        seed: int = 0,
    ):
        if point_estimate not in ("median", "mean"):
            raise ValueError("point_estimate must be 'median' or 'mean'")
        self.point_estimate = point_estimate
        self.horizon = horizon
        self.local_ridge = local_ridge
        self.local_damping = local_damping
        self.bias_prior_weight = bias_prior_weight
        self._smearing = 1.0
        self._config = {
            "spec": spec,
            "ridge": ridge,
            "rank": rank,
            "adapt_window": adapt_window,
            "replay_size": replay_size,
            "bias_shrinkage": bias_shrinkage,
            "residual_window": residual_window,
            "point_estimate": point_estimate,
            "horizon": horizon,
            "local_ridge": local_ridge,
            "local_damping": local_damping,
            "bias_prior_weight": bias_prior_weight,
            "seed": seed,
        }
        self.builder = FeatureBuilder(spec=spec or FeatureSpec())
        self.ridge = ridge
        self.rank = rank
        self.adapt_window = adapt_window
        self.replay_size = replay_size
        self.bias_shrinkage = bias_shrinkage
        self.residual_window = residual_window
        self.rng = np.random.default_rng(seed)

        self.w_base: np.ndarray | None = None
        self.projection: np.ndarray | None = None
        self.delta: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.delta_local: np.ndarray | None = None
        self.local_mask: np.ndarray | None = None
        self._replay: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._residuals: list[np.ndarray] = []

    def clone(self) -> "GlobalRidge":
        """An unfitted copy with the same configuration."""
        return GlobalRidge(**self._config)

    # ---------------------------------------------------------------- fitting

    @property
    def weights(self) -> np.ndarray:
        """Effective shared weights, base plus any low-rank correction."""
        assert self.w_base is not None
        if self.delta is None or self.projection is None:
            return self.w_base
        return self.w_base + self.projection @ self.delta

    @property
    def n_parameters(self) -> int:
        d = self.builder.spec.n_features
        n = 0 if self.bias is None else self.bias.size
        return d + n

    def _solve_ridge(
        self, X: np.ndarray, y: np.ndarray, penalty: float
    ) -> np.ndarray:
        d = X.shape[1]
        gram = X.T @ X + penalty * np.eye(d)
        return np.linalg.solve(gram, X.T @ y)

    def _fit_bias(
        self,
        resid: np.ndarray,
        index: np.ndarray,
        n_series: int,
        prior: np.ndarray | float = 0.0,
        prior_weight: float | None = None,
    ) -> np.ndarray:
        """Per-series mean residual, shrunk toward ``prior``.

        At fit time the prior is zero and this is an ordinary shrunk mean. At
        adaptation time the prior is the *existing* bias: a tier that fires on
        twelve months of data must not throw away an estimate built from the
        whole training set and replace it with a noisier one. Shrinking toward
        the current value instead of toward zero is the difference between the
        bias tier helping and actively hurting.
        """
        weight = self.bias_shrinkage if prior_weight is None else prior_weight
        sums = np.bincount(index, weights=resid, minlength=n_series)
        counts = np.bincount(index, minlength=n_series).astype(float)
        return (sums + weight * np.asarray(prior)) / (counts + weight)

    def fit(self, panel: Panel, end: int) -> None:
        self.builder.freeze_scale(panel, end)
        X, y, idx = self.builder.block(panel, MIN_HISTORY, end, self.horizon)
        if X.shape[0] == 0:
            raise ValueError("not enough history to fit")

        self.w_base = self._solve_ridge(X, y, self.ridge)
        resid = y - X @ self.w_base
        self.bias = self._fit_bias(resid, idx, panel.N)
        self._smearing = float(np.clip(np.mean(np.exp(resid - self.bias[idx])), 0.5, 4.0))

        # Fixed low-rank subspace: top-r right singular vectors of the training
        # design. Frozen here so adaptation never re-derives its own basis.
        _, _, vt = np.linalg.svd(X - X.mean(axis=0, keepdims=True), full_matrices=False)
        self.projection = vt[: self.rank].T
        self.delta = np.zeros(self.rank)

        self._replay = self._subsample(X, y, idx, self.replay_size)
        self._residuals = []

    def _subsample(
        self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if X.shape[0] <= size:
            return X.copy(), y.copy(), idx.copy()
        pick = self.rng.choice(X.shape[0], size=size, replace=False)
        return X[pick], y[pick], idx[pick]

    # ------------------------------------------------------------ prediction

    def _predict_log(self, panel: Panel, t: int) -> np.ndarray:
        """Uncentred ``log1p``-space prediction.

        The model is fitted on targets centred by each series' own level, so
        the level is added back here.
        """
        assert self.bias is not None and self.builder.series_scale is not None
        X = self.builder.row(panel, t, self.horizon)
        out = X @ self.weights + self.bias + self.builder.series_scale
        return out + self._local_contribution(X, None)

    def _local_contribution(
        self, X: np.ndarray, idx: np.ndarray | None
    ) -> np.ndarray:
        """Extra term from the scoped low-rank delta, zero where unscoped.

        Args:
            X: design rows.
            idx: series id per row, or ``None`` when ``X`` already has one row
                per series in panel order.
        """
        if self.delta_local is None or self.local_mask is None:
            return 0.0 if idx is None else np.zeros(X.shape[0])
        effect = X @ self.delta_local
        mask = self.local_mask if idx is None else self.local_mask[idx]
        return effect * mask

    def predict(self, panel: Panel, t: int) -> np.ndarray:
        """Point forecast on the original count scale.

        Fitting happens in ``log1p`` space, so back-transforming the fitted
        value gives the conditional **median**, not the mean. For right-skewed
        counts the median always sits below the mean, which shows up as a
        negative signed error of roughly ``exp(sigma^2 / 2) - 1`` even when the
        model is correct.

        ``point_estimate="median"`` (the default) minimises the MAE/MASE family
        this project reports. ``point_estimate="mean"`` applies Duan's smearing
        estimator instead, which removes the directional bias at some cost in
        absolute error. Which one is right depends on the downstream decision:
        for stock quantification, under-forecasting is not symmetric with
        over-forecasting, and the mean is the safer target.
        """
        mu = self._predict_log(panel, t)
        if self.point_estimate == "median":
            return np.maximum(np.expm1(mu), 0.0)
        return np.maximum(np.exp(mu) * self._smearing - 1.0, 0.0)

    def _update_smearing(self) -> None:
        """Duan's smearing factor: the mean of ``exp(residual)`` in log space."""
        if not self._residuals:
            return
        recent = np.concatenate(self._residuals[-self.residual_window :])
        factor = float(np.mean(np.exp(recent)))
        # Guard against a single extreme month blowing up the correction.
        self._smearing = float(np.clip(factor, 0.5, 4.0))

    def predict_quantiles(
        self, panel: Panel, t: int, quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    ) -> np.ndarray:
        point_log = self._predict_log(panel, t)
        if len(self._residuals) < 4:
            return np.repeat(np.maximum(np.expm1(point_log), 0.0)[:, None], len(quantiles), axis=1)
        resid = np.stack(self._residuals[-self.residual_window :])
        offsets = np.quantile(resid, quantiles, axis=0).T  # (N, Q)
        return np.maximum(np.expm1(point_log[:, None] + offsets), 0.0)

    def observe(self, panel: Panel, t: int) -> None:
        """Record the realised residual at ``t`` in log space."""
        pred_log = self._predict_log(panel, t)
        actual_log = np.log1p(np.maximum(panel.values[t], 0.0))
        self._residuals.append(actual_log - pred_log)
        if len(self._residuals) > 4 * self.residual_window:
            self._residuals = self._residuals[-self.residual_window :]
        self._update_smearing()

    # ------------------------------------------------------------ adaptation

    def adapt(
        self,
        panel: Panel,
        t: int,
        tier: str,
        affected: np.ndarray | None = None,
    ) -> AdaptationResult:
        """Apply ``tier`` using data up to and including ``t``.

        Args:
            panel: the data.
            t: current month.
            tier: one of :data:`TIERS`.
            affected: optional ``(N,)`` boolean mask of drifting series, used
                by the scoped ``local_lowrank`` tier.

        Returns:
            What was updated and what it cost.
        """
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}, expected one of {TIERS}")
        started = time.perf_counter()
        if tier == "none":
            return AdaptationResult("none", 0, 0.0)

        start = max(MIN_HISTORY, t + 1 - self.adapt_window)
        X, y, idx = self.builder.block(panel, start, t + 1, self.horizon)
        if X.shape[0] == 0:
            return AdaptationResult(tier, 0, time.perf_counter() - started, "no data")

        assert self.w_base is not None and self.bias is not None

        if tier == "bias":
            resid = y - (X @ self.weights + self._local_contribution(X, idx))
            self.bias = self._fit_bias(
                resid, idx, panel.N, prior=self.bias, prior_weight=self.bias_prior_weight
            )
            n_updated = int(self.bias.size)
            note = f"window={X.shape[0]} rows"

        elif tier == "local_lowrank":
            assert self.projection is not None
            if affected is None or not np.any(affected):
                return AdaptationResult(
                    tier, 0, time.perf_counter() - started, "nothing flagged"
                )
            rows = affected[idx]
            Xa, ya, idxa = X[rows], y[rows], idx[rows]
            if Xa.shape[0] == 0:
                return AdaptationResult(
                    tier, 0, time.perf_counter() - started, "no rows for flagged series"
                )
            fitted = Xa @ self.weights + self.bias[idxa]
            # The scoped correction spans the full feature space, shrunk hard
            # toward zero. At d=14 a rank-r restriction saves nothing worth
            # having, and it cannot represent a change confined to one
            # coefficient (a rainfall sensitivity, say) unless that direction
            # happens to lie in the subspace. The parameter-efficiency argument
            # only bites on a backbone with many more weights than this.
            step = self._solve_ridge(Xa, ya - fitted, self.local_ridge)
            if self.delta_local is None:
                self.delta_local = np.zeros(self.builder.spec.n_features)
            self.delta_local = (
                1.0 - self.local_damping
            ) * self.delta_local + self.local_damping * step
            # Series keep their correction once flagged: the regime change this
            # tier exists for is persistent, so dropping the correction the
            # moment a series stops looking anomalous would undo the fix.
            self.local_mask = (
                affected if self.local_mask is None else (self.local_mask | affected)
            )
            resid_all = y - (
                X @ self.weights + self._local_contribution(X, idx)
            )
            refreshed = self._fit_bias(
                resid_all, idx, panel.N, prior=self.bias,
                prior_weight=self.bias_prior_weight,
            )
            self.bias = np.where(affected, refreshed, self.bias)
            n_updated = int(self.delta_local.size + int(np.sum(affected)))
            note = f"scoped to {int(np.sum(affected))}/{panel.N} series"

        elif tier == "lowrank":
            assert self.projection is not None and self.delta is not None
            # Fit the correction on what the current model still gets wrong.
            # Fit the correction against the *base* weights and blend toward
            # it, rather than adding a fresh step to the running total every
            # month. Accumulating steps fitted to one noisy window after
            # another is a random walk in parameter space: it drifts away from
            # the optimum even when each individual step points the right way.
            resid = y - (X @ self.w_base + self.bias[idx])
            Z = X @ self.projection
            step = self._solve_ridge(Z, resid, self.ridge)
            self.delta = (1.0 - self.local_damping) * self.delta + self.local_damping * step
            resid_after = y - (X @ self.weights + self._local_contribution(X, idx))
            self.bias = self._fit_bias(
                resid_after, idx, panel.N, prior=self.bias,
                prior_weight=self.bias_prior_weight,
            )
            n_updated = int(self.delta.size + self.bias.size)
            note = f"rank={self.rank}, |step|={np.linalg.norm(step):.4f}"

        else:  # full
            if self._replay is not None:
                Xr, yr, idxr = self._replay
                X = np.concatenate([X, Xr])
                y = np.concatenate([y, yr])
                idx = np.concatenate([idx, idxr])
            self.w_base = self._solve_ridge(X, y, self.ridge)
            self.delta = np.zeros(self.rank)
            self.delta_local = None
            self.local_mask = None
            resid = y - X @ self.w_base
            self.bias = self._fit_bias(resid, idx, panel.N)
            n_updated = int(self.w_base.size + self.bias.size)
            note = f"refit on {X.shape[0]} rows incl. replay"

        return AdaptationResult(tier, n_updated, time.perf_counter() - started, note)
