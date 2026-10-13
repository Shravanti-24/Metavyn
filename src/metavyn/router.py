"""Adaptation policies: when to adapt, and how deeply.

Each policy maps the month's :class:`~metavyn.drift.DriftSignal` to a tier name.
Swapping the policy while holding the backbone and the data fixed is exactly the
comparison the project is making, so the competing strategies from the
literature live here as first-class policies rather than as separate scripts.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from .drift import DriftSignal

# Default severity cut points. Because severity is a calibrated rank against a
# quiet-period null, these read directly as false-alarm rates: 0.90 fires on
# roughly the most extreme 10% of ordinary months.
DEFAULT_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.97, "full"),
    (0.93, "local_lowrank"),
    (0.88, "bias"),
)


class Policy(abc.ABC):
    """Decides the adaptation tier for a month."""

    name: str = "policy"

    @abc.abstractmethod
    def route(self, signal: DriftSignal) -> str:
        """Return a tier name for this month."""

    def reset(self) -> None:
        """Clear per-run state."""


@dataclass
class NeverAdapt(Policy):
    """Frozen model. Isolates the value of the backbone alone."""

    name: str = "frozen"

    def route(self, signal: DriftSignal) -> str:
        return "none"


@dataclass
class PeriodicRetrain(Policy):
    """Retrain on a fixed calendar, regardless of what the data is doing.

    This is current practice in deployed health forecasting pipelines and the
    reference the project has to beat on cost, not just on error.
    """

    period: int = 3
    tier: str = "full"
    name: str = "periodic"
    _seen: int = field(default=0, init=False)

    def route(self, signal: DriftSignal) -> str:
        self._seen += 1
        return self.tier if self._seen % self.period == 0 else "none"

    def reset(self) -> None:
        self._seen = 0


@dataclass
class AlwaysAdapt(Policy):
    """Adapt every month at one fixed depth, ignoring severity."""

    tier: str = "lowrank"
    name: str = "always"

    def route(self, signal: DriftSignal) -> str:
        return self.tier


@dataclass
class BinaryTrigger(Policy):
    """Detect-and-retrain: one threshold, one response depth.

    The thing the severity ladder has to improve on. If grading drift adds
    nothing, this policy will match it at equal or lower cost.
    """

    threshold: float = 0.95
    tier: str = "full"
    name: str = "binary"

    def route(self, signal: DriftSignal) -> str:
        return self.tier if signal.severity >= self.threshold else "none"


@dataclass
class SeverityRouter(Policy):
    """Metavyn's policy: deeper adaptation only for more severe drift.

    Args:
        thresholds: ``(cutoff, tier)`` pairs, checked from most to least severe.
        cooldown: months to wait after a full refit before allowing another, so
            a single long shock cannot trigger repeated expensive retrains.
    """

    thresholds: tuple[tuple[float, str], ...] = DEFAULT_THRESHOLDS
    cooldown: int = 3
    name: str = "severity_router"
    _last_full: int | None = field(default=None, init=False)

    def route(self, signal: DriftSignal) -> str:
        ladder = sorted(self.thresholds, reverse=True)
        chosen = len(ladder)  # past the end means "none"
        for i, (cutoff, _) in enumerate(ladder):
            if signal.severity >= cutoff:
                chosen = i
                break
        tier = "none" if chosen == len(ladder) else ladder[chosen][1]

        if tier == "full":
            inside_cooldown = (
                self._last_full is not None
                and signal.t - self._last_full < self.cooldown
            )
            if inside_cooldown:
                # Take the next rung down rather than paying for another full
                # refit. Which rung that is comes from the configured ladder,
                # so a caller who swaps the tiers gets their own next rung.
                tier = ladder[chosen + 1][1] if chosen + 1 < len(ladder) else "none"
            else:
                self._last_full = signal.t
        return tier

    def reset(self) -> None:
        self._last_full = None


def policy_suite() -> dict[str, Policy]:
    """The policies compared in the MVP experiment."""
    return {
        "frozen": NeverAdapt(),
        "quarterly_full": PeriodicRetrain(period=3, tier="full"),
        "always_lowrank": AlwaysAdapt(tier="lowrank"),
        "always_full": AlwaysAdapt(tier="full"),
        "binary_full": BinaryTrigger(threshold=0.93, tier="full"),
        "metavyn_global": SeverityRouter(
            thresholds=((0.97, "full"), (0.93, "lowrank"), (0.88, "bias"))
        ),
        "metavyn": SeverityRouter(),
    }
