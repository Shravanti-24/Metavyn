"""Panel data structures.

The unit of analysis is a ``district x indicator`` monthly series. A :class:`Panel`
holds every such series for one study area as a dense ``(T, N)`` matrix, plus any
exogenous drivers (rainfall, for example) on the same grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SEASON_LENGTH = 12

# Zero-fraction cut points used to bucket series by how intermittent they are.
# Humanitarian health demand is a mix of dense and lumpy series, and pooling the
# two hides exactly the errors that matter, so every report is segmented by these.
DENSE_MAX_ZERO_FRAC = 0.10
INTERMITTENT_MAX_ZERO_FRAC = 0.40


@dataclass(frozen=True)
class SeriesKey:
    """Identifies one column of the panel."""

    district: str
    indicator: str

    def __str__(self) -> str:
        return f"{self.district}/{self.indicator}"


@dataclass
class Panel:
    """A monthly ``district x indicator`` panel.

    Attributes:
        values: ``(T, N)`` observed counts.
        series: ``N`` keys, one per column.
        exog: named ``(T, N)`` driver matrices aligned to ``values``.
        month: ``(T,)`` month-of-year in ``0..11``, used for seasonal features.
        season_length: periods per seasonal cycle.
    """

    values: np.ndarray
    series: list[SeriesKey]
    exog: dict[str, np.ndarray] = field(default_factory=dict)
    month: np.ndarray | None = None
    season_length: int = SEASON_LENGTH

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2-D (T, N), got shape {self.values.shape}")
        if len(self.series) != self.values.shape[1]:
            raise ValueError(
                f"{len(self.series)} series keys for {self.values.shape[1]} columns"
            )
        for name, arr in self.exog.items():
            arr = np.asarray(arr, dtype=float)
            if arr.shape != self.values.shape:
                raise ValueError(
                    f"exog {name!r} has shape {arr.shape}, expected {self.values.shape}"
                )
            self.exog[name] = arr
        if self.month is None:
            self.month = np.arange(self.T) % self.season_length
        else:
            self.month = np.asarray(self.month, dtype=int)
            if self.month.shape != (self.T,):
                raise ValueError(f"month must have shape ({self.T},)")

    @property
    def T(self) -> int:
        """Number of time periods."""
        return self.values.shape[0]

    @property
    def N(self) -> int:
        """Number of series."""
        return self.values.shape[1]

    def zero_fraction(self, end: int | None = None) -> np.ndarray:
        """Share of zero observations per series, over ``values[:end]``."""
        window = self.values[:end]
        if window.shape[0] == 0:
            return np.zeros(self.N)
        return (window == 0).mean(axis=0)

    def intermittency_class(self, end: int | None = None) -> np.ndarray:
        """Label each series ``dense``, ``intermittent`` or ``lumpy``.

        Van der Laan et al. (2016) found roughly 40% of humanitarian medical items
        have intermittent consumption. Error pooled across all three classes can
        hide a model that is accurate on dense series and badly biased on sparse
        ones, so results are always broken out this way.
        """
        zf = self.zero_fraction(end)
        labels = np.full(self.N, "lumpy", dtype=object)
        labels[zf <= INTERMITTENT_MAX_ZERO_FRAC] = "intermittent"
        labels[zf <= DENSE_MAX_ZERO_FRAC] = "dense"
        return labels

    def slice_time(self, start: int = 0, end: int | None = None) -> Panel:
        """A new panel over ``[start, end)``, sharing series identity."""
        end = self.T if end is None else end
        return Panel(
            values=self.values[start:end],
            series=list(self.series),
            exog={k: v[start:end] for k, v in self.exog.items()},
            month=self.month[start:end],
            season_length=self.season_length,
        )

    def districts(self) -> list[str]:
        """Unique district names, in first-seen order."""
        seen: dict[str, None] = {}
        for key in self.series:
            seen.setdefault(key.district, None)
        return list(seen)

    def indicators(self) -> list[str]:
        """Unique indicator names, in first-seen order."""
        seen: dict[str, None] = {}
        for key in self.series:
            seen.setdefault(key.indicator, None)
        return list(seen)
