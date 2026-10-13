"""Forecast error metrics for sparse, shock-driven panels.

Two choices here follow directly from the humanitarian forecasting literature
rather than from convention:

* **No MAPE.** Roughly 40% of medical items have intermittent consumption
  (Van der Laan et al., 2016), so the denominator is zero often enough that MAPE
  is undefined or explosive. MASE is the headline point metric.
* **Signed bias is reported alongside magnitude.** The documented failure mode
  in humanitarian demand planning is systematic *over*-forecasting (on the order
  of 70% of items in the same study). MAE, RMSE, MASE, pinball loss and interval
  coverage are all magnitude-only and would not show it, so every report carries
  a signed term too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-8


def seasonal_naive_scale(train: np.ndarray, season_length: int = 12) -> np.ndarray:
    """Per-series MASE denominator: in-sample mean absolute seasonal difference.

    Args:
        train: ``(T, N)`` training values.
        season_length: seasonal period.

    Returns:
        ``(N,)`` strictly positive scales.
    """
    train = np.asarray(train, dtype=float)
    if train.shape[0] <= season_length:
        # Too short for a seasonal difference; fall back to a first difference.
        diffs = np.abs(np.diff(train, axis=0)) if train.shape[0] > 1 else None
    else:
        diffs = np.abs(train[season_length:] - train[:-season_length])
    if diffs is None or diffs.size == 0:
        scale = np.ones(train.shape[1])
    else:
        scale = diffs.mean(axis=0)
    # A flat series has zero scale; guard so MASE stays finite.
    return np.where(scale > EPS, scale, 1.0)


def mase(
    y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray, axis: int | None = None
) -> np.ndarray | float:
    """Mean absolute scaled error."""
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred)) / scale[None, :]
    return err.mean() if axis is None else err.mean(axis=axis)


def smape(
    y_true: np.ndarray, y_pred: np.ndarray, axis: int | None = None
) -> np.ndarray | float:
    """Symmetric MAPE in percent, safe at zero demand."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.where(denom > EPS, np.abs(y_true - y_pred) / np.maximum(denom, EPS), 0.0)
    out = 100.0 * ratio
    return out.mean() if axis is None else out.mean(axis=axis)


def mean_signed_error(
    y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray
) -> float:
    """Scaled mean signed error. Positive means the model over-forecasts."""
    err = (np.asarray(y_pred) - np.asarray(y_true)) / scale[None, :]
    return float(err.mean())


def share_biased_high(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of series whose mean error is positive (over-forecast).

    This is the direct analogue of the ~70% over-forecast finding on MSF data.
    A value near 0.5 means errors are directionally balanced across the panel.
    """
    per_series = (np.asarray(y_pred) - np.asarray(y_true)).mean(axis=0)
    return float((per_series > 0).mean())


def pinball_loss(
    y_true: np.ndarray, quantile_pred: np.ndarray, quantiles: np.ndarray
) -> float:
    """Mean pinball loss over quantiles.

    Args:
        y_true: ``(T, N)`` observations.
        quantile_pred: ``(T, N, Q)`` predicted quantiles.
        quantiles: ``(Q,)`` levels in ``(0, 1)``.
    """
    y_true = np.asarray(y_true, dtype=float)[..., None]
    quantile_pred = np.asarray(quantile_pred, dtype=float)
    q = np.asarray(quantiles, dtype=float)[None, None, :]
    diff = y_true - quantile_pred
    loss = np.maximum(q * diff, (q - 1.0) * diff)
    return float(loss.mean())


def interval_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Empirical coverage of a predictive interval."""
    y_true = np.asarray(y_true)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    return float(inside.mean())


@dataclass
class ErrorReport:
    """Point, probabilistic and directional error for one slice of a run."""

    n_obs: int
    mase: float
    smape: float
    mae: float
    rmse: float
    signed_error: float
    share_biased_high: float
    pinball: float | None = None
    coverage_80: float | None = None
    by_class: dict[str, "ErrorReport"] = field(default_factory=dict)

    def to_row(self) -> dict[str, float | int | None]:
        """Flat mapping for tabular display, excluding nested breakdowns."""
        return {
            "n_obs": self.n_obs,
            "MASE": self.mase,
            "sMAPE": self.smape,
            "MAE": self.mae,
            "RMSE": self.rmse,
            "signed_err": self.signed_error,
            "share_high": self.share_biased_high,
            "pinball": self.pinball,
            "cov80": self.coverage_80,
        }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: np.ndarray,
    *,
    quantile_pred: np.ndarray | None = None,
    quantiles: np.ndarray | None = None,
    classes: np.ndarray | None = None,
) -> ErrorReport:
    """Score a block of forecasts, optionally broken out by intermittency class.

    Args:
        y_true: ``(T, N)`` observations.
        y_pred: ``(T, N)`` point forecasts.
        scale: ``(N,)`` MASE denominators from :func:`seasonal_naive_scale`.
        quantile_pred: optional ``(T, N, Q)`` quantile forecasts.
        quantiles: optional ``(Q,)`` levels matching ``quantile_pred``.
        classes: optional ``(N,)`` labels to segment by.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    report = ErrorReport(
        n_obs=int(y_true.size),
        mase=float(mase(y_true, y_pred, scale)),
        smape=float(smape(y_true, y_pred)),
        mae=float(np.abs(err).mean()),
        rmse=float(np.sqrt((err**2).mean())),
        signed_error=mean_signed_error(y_true, y_pred, scale),
        share_biased_high=share_biased_high(y_true, y_pred),
    )

    if quantile_pred is not None and quantiles is not None:
        quantiles = np.asarray(quantiles, dtype=float)
        report.pinball = pinball_loss(y_true, quantile_pred, quantiles)
        lo_idx = int(np.argmin(np.abs(quantiles - 0.1)))
        hi_idx = int(np.argmin(np.abs(quantiles - 0.9)))
        if lo_idx != hi_idx:
            report.coverage_80 = interval_coverage(
                y_true, quantile_pred[..., lo_idx], quantile_pred[..., hi_idx]
            )

    if classes is not None:
        classes = np.asarray(classes)
        for label in sorted(set(classes.tolist())):
            mask = classes == label
            if not mask.any():
                continue
            sub_q = quantile_pred[:, mask] if quantile_pred is not None else None
            report.by_class[str(label)] = evaluate(
                y_true[:, mask],
                y_pred[:, mask],
                scale[mask],
                quantile_pred=sub_q,
                quantiles=quantiles,
                classes=None,
            )
    return report
