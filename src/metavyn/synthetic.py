"""Synthetic post-disaster health demand panels.

Real HMIS extracts are slow to obtain and slower to clean, and the drift-routing
loop needs *known* shock timings to be evaluated at all. This module generates
panels whose shocks are placed by construction, so detection latency and
recovery can be measured against ground truth.

The generative story per series is::

    level x seasonality x (1 + shock response) + noise

with counts drawn from a negative binomial so that sparse indicators are
genuinely intermittent rather than Gaussian-noisy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .panel import Panel, SeriesKey

# Indicator archetypes: (name, baseline monthly level, seasonal amplitude,
# dispersion, flood sensitivity). Flood sensitivity is the multiplicative bump
# applied at the peak of a shock response.
DEFAULT_INDICATORS: tuple[tuple[str, float, float, float, float, float], ...] = (
    #  name                 level   amp  disp  flood  rain_beta
    ("fever_cases", 120.0, 0.35, 4.0, 1.60, 0.25),
    ("diarrhoea_cases", 80.0, 0.45, 3.0, 2.20, 0.40),
    ("ari_cases", 95.0, 0.30, 4.0, 1.35, -0.15),
    ("snakebite_cases", 4.0, 0.60, 1.2, 1.80, 0.30),
    ("ors_packets_issued", 60.0, 0.40, 2.5, 2.40, 0.35),
    ("antibiotic_courses", 45.0, 0.20, 3.0, 1.45, 0.10),
)


@dataclass(frozen=True)
class RegimeChange:
    """A permanent change in how demand responds to its drivers.

    This is the drift the project actually targets, and it is not the same
    thing as a :class:`Shock`. A shock moves the *level* for a few months, and
    any model carrying a lag-1 term absorbs it by itself one month later. A
    regime change moves the *mapping*: after it, the same rainfall produces a
    different case load, so a model with frozen coefficients stays wrong
    indefinitely and only adaptation recovers.

    Attributes:
        start: month the new regime begins.
        districts: districts affected; empty means all.
        rain_multiplier: scales each indicator's rainfall sensitivity.
        season_multiplier: scales seasonal amplitude.
        level_multiplier: permanent shift in baseline level.
        phase_shift_months: moves the seasonal peak, in months.
        name: label used in reporting.

    Note which of these a model can actually recover. Scaling amplitude or
    level mostly changes how *noisy* the series is, and no amount of parameter
    fitting recovers error that is irreducible: an oracle refitted on
    post-change data scores the same as a frozen model. A phase shift is
    different. The seasonal component is predictable, so a model carrying the
    old phase is systematically wrong in a way refitting genuinely fixes. If a
    drift experiment is to have any headroom for adaptation to compete over, it
    has to move something predictable.
    """

    start: int
    districts: tuple[str, ...] = ()
    rain_multiplier: float = 1.0
    season_multiplier: float = 1.0
    level_multiplier: float = 1.0
    phase_shift_months: float = 0.0
    name: str = "regime"


@dataclass(frozen=True)
class Shock:
    """A disaster affecting some districts from month ``start``.

    Attributes:
        start: month index at which the shock begins.
        districts: districts hit; empty means all.
        magnitude: scales each indicator's flood sensitivity.
        decay: per-month multiplicative decay of the response.
        duration: months the response is allowed to persist.
        name: label used in reporting.
    """

    start: int
    districts: tuple[str, ...] = ()
    magnitude: float = 1.0
    decay: float = 0.55
    duration: int = 6
    name: str = "shock"

    def response(self, horizon: int) -> np.ndarray:
        """Response weights for ``horizon`` months from ``start``."""
        t = np.arange(horizon)
        weights = np.where(t < self.duration, self.decay**t, 0.0)
        return weights * self.magnitude


def make_panel(
    n_districts: int = 12,
    n_periods: int = 96,
    indicators: tuple[tuple[str, float, float, float, float, float], ...] = DEFAULT_INDICATORS,
    shocks: tuple[Shock, ...] = (),
    regimes: tuple[RegimeChange, ...] = (),
    trend: float = 0.0015,
    rainfall_shock_boost: float = 2.5,
    seed: int = 0,
) -> tuple[Panel, dict[str, np.ndarray]]:
    """Generate a panel plus the ground-truth components used to score detection.

    Args:
        n_districts: number of districts.
        n_periods: number of monthly observations.
        indicators: indicator archetypes, see :data:`DEFAULT_INDICATORS`.
        shocks: disasters to inject.
        trend: per-month multiplicative drift in the baseline level.
        rainfall_shock_boost: multiplier applied to rainfall during a shock, so
            the exogenous channel carries a (noisy) early signal.
        seed: RNG seed.

    Returns:
        ``(panel, truth)`` where ``truth`` holds ``shock_months`` (a ``(T,)``
        indicator) and ``shock_by_series`` (a ``(T, N)`` response matrix).
    """
    rng = np.random.default_rng(seed)
    districts = [f"D{i:02d}" for i in range(n_districts)]

    series: list[SeriesKey] = []
    levels: list[float] = []
    amps: list[float] = []
    disps: list[float] = []
    sens: list[float] = []
    betas: list[float] = []
    for district in districts:
        # Districts differ in size; this is what makes a shared global model
        # need per-series scaling rather than one pooled intercept.
        size = float(np.exp(rng.normal(0.0, 0.45)))
        for name, level, amp, disp, sensitivity, beta in indicators:
            series.append(SeriesKey(district=district, indicator=name))
            levels.append(level * size)
            amps.append(amp)
            disps.append(disp)
            sens.append(sensitivity)
            betas.append(beta)

    level_arr = np.array(levels)
    amp_arr = np.array(amps)
    disp_arr = np.array(disps)
    sens_arr = np.array(sens)
    beta_arr = np.array(betas)
    n_series = len(series)

    month = np.arange(n_periods) % 12
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_series)
    growth = (1.0 + trend) ** np.arange(n_periods)[:, None]

    # Parameter paths over time. A regime change edits these permanently from
    # its start month, which is what makes it survive a lag-1 feature.
    amp_path = np.tile(amp_arr, (n_periods, 1))
    phase_path = np.zeros((n_periods, n_series))
    beta_path = np.tile(beta_arr, (n_periods, 1))
    level_path = np.tile(level_arr, (n_periods, 1))
    regime_months = np.zeros(n_periods, dtype=bool)
    for regime in regimes:
        if regime.start >= n_periods:
            continue
        hit = (
            np.ones(n_series, dtype=bool)
            if not regime.districts
            else np.array([k.district in regime.districts for k in series])
        )
        amp_path[regime.start :, hit] *= regime.season_multiplier
        phase_path[regime.start :, hit] += (
            2.0 * np.pi * regime.phase_shift_months / 12.0
        )
        beta_path[regime.start :, hit] *= regime.rain_multiplier
        level_path[regime.start :, hit] *= regime.level_multiplier
        regime_months[regime.start] = True

    seasonal = 1.0 + amp_path * np.sin(
        2.0 * np.pi * month[:, None] / 12.0 + phase[None, :] + phase_path
    )

    # Rainfall: a seasonal monsoon signal, amplified during shocks.
    rainfall = np.maximum(
        0.0,
        60.0
        + 55.0 * np.sin(2.0 * np.pi * (month - 4) / 12.0)[:, None]
        + rng.normal(0.0, 18.0, size=(n_periods, n_series)),
    )

    shock_response = np.zeros((n_periods, n_series))
    shock_months = np.zeros(n_periods, dtype=bool)
    for shock in shocks:
        if shock.start >= n_periods:
            continue
        hit = (
            np.ones(n_series, dtype=bool)
            if not shock.districts
            else np.array([k.district in shock.districts for k in series])
        )
        horizon = n_periods - shock.start
        weights = shock.response(horizon)
        # (horizon, 1) x (1, N) -> per-series response, scaled by sensitivity.
        contribution = weights[:, None] * ((sens_arr - 1.0) * hit)[None, :]
        shock_response[shock.start :] += contribution
        active = shock.start + np.flatnonzero(weights > 1e-6)
        shock_months[active] = True
        rainfall[shock.start : shock.start + 2] *= rainfall_shock_boost

    # Rainfall genuinely drives demand, so that changing its coefficient is a
    # real change in the input-output mapping rather than a cosmetic one.
    rain_z = (rainfall - rainfall.mean(axis=0)) / np.maximum(
        rainfall.std(axis=0), 1e-6
    )
    rain_effect = np.maximum(0.1, 1.0 + beta_path * rain_z)

    mean = level_path * seasonal * growth * rain_effect * (1.0 + shock_response)
    mean = np.maximum(mean, 1e-3)

    # Negative binomial via gamma-Poisson: dispersion controls how lumpy a
    # series is, which is what puts some indicators in the intermittent bucket.
    shape = disp_arr[None, :]
    gamma = rng.gamma(shape=shape, scale=mean / shape)
    values = rng.poisson(gamma).astype(float)

    panel = Panel(
        values=values,
        series=series,
        exog={"rainfall": rainfall},
        month=month,
    )
    truth = {
        "shock_months": shock_months,
        "shock_by_series": shock_response,
        "regime_months": regime_months,
    }
    return panel, truth


def default_scenario(seed: int = 0, n_periods: int = 120) -> tuple[Panel, dict]:
    """The scenario the demo and tests run on: ten years, two shocks.

    A severe flood hits a third of districts, then a milder, wider cyclone
    arrives twenty months later. The gap is deliberate, so recovery from the
    first is observable before the second lands.

    Ten years is longer than the ~72 monthly points a real Tamil Nadu HMIS
    extract gives per district-indicator. That is not an accident: the drift
    null has to be estimated from quiet months, and the number of quiet months
    available directly sets the resolution of the severity rank. Use
    :func:`short_scenario` to see what happens when the history is realistic
    rather than generous.
    """
    shocks = (
        Shock(
            start=84,
            districts=("D00", "D01", "D02", "D03"),
            magnitude=1.0,
            decay=0.55,
            duration=6,
            name="flood",
        ),
        Shock(
            start=104,
            districts=(),
            magnitude=0.45,
            decay=0.45,
            duration=4,
            name="cyclone",
        ),
    )
    panel, truth = make_panel(n_periods=n_periods, shocks=shocks, seed=seed)
    truth["shocks"] = shocks
    return panel, truth


def regime_scenario(seed: int = 0, n_periods: int = 120) -> tuple[Panel, dict]:
    """Ten years with a flood that *permanently changes the mapping*.

    The flood at month 84 is followed by a lasting regime change in the same
    districts: damaged water and sanitation infrastructure makes case loads
    roughly twice as sensitive to rainfall and the seasonal peak moves three
    months earlier.

    This is the scenario the severity router is designed for, and the contrast
    with :func:`default_scenario` is the point. There, the shock is transient
    and any lag-1 model absorbs it unaided, so adaptation buys nothing. Here the
    coefficients themselves have moved, a frozen model stays wrong for as long
    as it is left alone, and the question becomes how deeply and how often to
    adapt rather than whether to.
    """
    shocks = (
        Shock(
            start=84,
            districts=("D00", "D01", "D02", "D03"),
            magnitude=1.0,
            decay=0.55,
            duration=6,
            name="flood",
        ),
    )
    regimes = (
        RegimeChange(
            start=84,
            districts=("D00", "D01", "D02", "D03"),
            rain_multiplier=2.4,
            season_multiplier=1.0,
            level_multiplier=1.0,
            phase_shift_months=3.0,
            name="post_flood_regime",
        ),
    )
    panel, truth = make_panel(
        n_periods=n_periods, shocks=shocks, regimes=regimes, seed=seed
    )
    truth["shocks"] = shocks
    truth["regimes"] = regimes
    return panel, truth


def short_scenario(seed: int = 0) -> tuple[Panel, dict]:
    """Six years of history, one shock: the realistic-data stress test.

    This is the regime an actual HMIS extract puts the method in. The detector
    gets far fewer quiet months to calibrate against, so the severity rank is
    coarser and the false-alarm rate is correspondingly worse.
    """
    shocks = (
        Shock(
            start=60,
            districts=("D00", "D01", "D02", "D03"),
            magnitude=1.0,
            decay=0.55,
            duration=6,
            name="flood",
        ),
    )
    panel, truth = make_panel(n_periods=72, shocks=shocks, seed=seed)
    truth["shocks"] = shocks
    return panel, truth
