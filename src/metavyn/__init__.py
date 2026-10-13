"""Metavyn: drift-aware, severity-routed adaptation for humanitarian demand forecasting.

The loop this package implements, one month at a time:

1. forecast every ``district x indicator`` series one month ahead;
2. measure drift across three channels and grade it into one severity score;
3. route that severity to a *depth* of adaptation, from doing nothing to a full
   refit with replay;
4. score the result on metrics that suit sparse, shock-driven demand.

The contribution under test is step 3. Everything else is scaffolding built so
that step 3 can be compared against the policies it has to beat.
"""

from .backtest import RunResult, run_adaptive, run_baseline
from .baselines import CrostonSBA, SeasonalNaive, WeightedMovingAverage
from .drift import DriftSignal, SeverityDetector
from .metrics import ErrorReport, evaluate, seasonal_naive_scale
from .panel import Panel, SeriesKey
from .ridge import GlobalRidge
from .router import (
    AlwaysAdapt,
    BinaryTrigger,
    NeverAdapt,
    PeriodicRetrain,
    SeverityRouter,
    policy_suite,
)
from .synthetic import Shock, default_scenario, make_panel

__version__ = "0.1.0"

__all__ = [
    "AlwaysAdapt",
    "BinaryTrigger",
    "CrostonSBA",
    "DriftSignal",
    "ErrorReport",
    "GlobalRidge",
    "NeverAdapt",
    "Panel",
    "PeriodicRetrain",
    "RunResult",
    "SeasonalNaive",
    "SeriesKey",
    "SeverityDetector",
    "SeverityRouter",
    "Shock",
    "WeightedMovingAverage",
    "default_scenario",
    "evaluate",
    "make_panel",
    "policy_suite",
    "run_adaptive",
    "run_baseline",
    "seasonal_naive_scale",
]
