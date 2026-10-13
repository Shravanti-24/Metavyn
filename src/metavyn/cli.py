"""Command line entry point: ``metavyn demo``."""

from __future__ import annotations

import argparse

import numpy as np

from .backtest import detection_latency, false_alarm_rate, run_adaptive, run_baseline
from .baselines import CrostonSBA, SeasonalNaive, WeightedMovingAverage
from .ridge import GlobalRidge
from .router import NeverAdapt, SeverityRouter
from .synthetic import default_scenario, regime_scenario, short_scenario

SCENARIOS = {
    "transient": default_scenario,
    "regime": regime_scenario,
    "short": short_scenario,
}


def demo(scenario: str, seed: int, horizon: int) -> None:
    """Run one scenario end to end and print what happened, month by month."""
    panel, truth = SCENARIOS[scenario](seed=seed)
    end = int(panel.T * 0.6)
    print(
        f"scenario={scenario}  panel={panel.T} months x {panel.N} series "
        f"({len(panel.districts())} districts x {len(panel.indicators())} indicators)"
    )
    print(f"train on months 0-{end - 1}, evaluate {end}-{panel.T - 1}, horizon={horizon}\n")

    print(f"{'model':<26}{'MASE':>8}{'sMAPE':>8}{'bias':>8}{'cov80':>8}")
    for model in (SeasonalNaive(), WeightedMovingAverage(), CrostonSBA()):
        r = run_baseline(panel, model, end)
        rep = r.report
        print(
            f"{model.name:<26}{rep.mase:8.4f}{rep.smape:8.2f}"
            f"{rep.signed_error:8.3f}{(rep.coverage_80 or 0):8.3f}"
        )

    runs = {}
    for label, policy in (("frozen", NeverAdapt()), ("metavyn", SeverityRouter())):
        r = run_adaptive(panel, GlobalRidge(horizon=horizon), policy, end, label=label)
        runs[label] = r
        rep = r.report
        print(
            f"{label:<26}{rep.mase:8.4f}{rep.smape:8.2f}"
            f"{rep.signed_error:8.3f}{(rep.coverage_80 or 0):8.3f}"
        )

    metavyn = runs["metavyn"]
    print("\nerror by intermittency class (metavyn):")
    for name, sub in sorted(metavyn.report.by_class.items()):
        print(f"  {name:<14} MASE={sub.mase:7.4f}  bias={sub.signed_error:+7.3f}  n={sub.n_obs}")

    shock_months = truth["shock_months"]
    drifted = shock_months.copy()
    for regime in truth.get("regimes", ()):
        drifted[regime.start :] = True
    print("\ndetector:")
    print(f"  false alarms on quiet months (sev>=0.88): {false_alarm_rate(metavyn.signals, drifted, 0.88):.1%}")
    for shock in truth.get("shocks", ()):
        lat = detection_latency(metavyn.signals, shock.start, 0.88)
        print(f"  {shock.name} at month {shock.start}: detection latency = {lat} months")

    print(f"\nadaptation: {metavyn.tier_counts}  total parameters written = {metavyn.total_params_updated}")

    print("\nmonths around the first shock:")
    first = truth["shocks"][0].start if truth.get("shocks") else end
    print(f"  {'month':>6}{'severity':>10}  tier")
    for signal, tier in zip(metavyn.signals, metavyn.tiers):
        if first - 3 <= signal.t <= first + 7:
            flag = " <- shock" if shock_months[signal.t] else ""
            print(f"  {signal.t:>6}{signal.severity:>10.2f}  {tier}{flag}")

    frozen_mase = runs["frozen"].report.mase
    delta = 100 * (metavyn.report.mase - frozen_mase) / frozen_mase
    verdict = "better than" if delta < 0 else "worse than"
    print(
        f"\nverdict: severity routing is {abs(delta):.2f}% {verdict} the frozen model "
        f"on this run. See README for what that means."
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="metavyn", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("demo", help="run one scenario end to end")
    run.add_argument("--scenario", choices=sorted(SCENARIOS), default="regime")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    if args.command == "demo":
        demo(args.scenario, args.seed, args.horizon)


if __name__ == "__main__":
    main()
