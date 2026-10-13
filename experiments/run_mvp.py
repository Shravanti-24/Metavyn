"""Reproduce the MVP result: do adaptation policies beat leaving the model alone?

Runs every policy against every scenario over several seeds and prints the
comparison, plus the detector's latency and false-alarm numbers.

    python experiments/run_mvp.py --seeds 8

The headline finding is reported whichever way it comes out. If the adaptive
policies do not beat the frozen model, that is the result.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np

from metavyn import (
    CrostonSBA,
    GlobalRidge,
    SeasonalNaive,
    WeightedMovingAverage,
    default_scenario,
    policy_suite,
    run_adaptive,
    run_baseline,
)
from metavyn.backtest import detection_latency, false_alarm_rate
from metavyn.synthetic import regime_scenario, short_scenario

SCENARIOS = {
    "transient_shock": default_scenario,
    "regime_change": regime_scenario,
    "short_history": short_scenario,
}


def train_end_for(scenario: str, n_periods: int) -> int:
    """Hold out the last 40% of the panel, leaving room to calibrate."""
    return int(n_periods * 0.6)


def run_scenario(name: str, seeds: int, horizon: int) -> tuple[list[dict], list[dict]]:
    """Run every forecaster and policy on one scenario."""
    rows: list[dict] = []
    detect: list[dict] = []

    for seed in range(seeds):
        panel, truth = SCENARIOS[name](seed=seed)
        end = train_end_for(name, panel.T)

        for model in (SeasonalNaive(), WeightedMovingAverage(), CrostonSBA()):
            result = run_baseline(panel, model, end)
            rows.append(
                {
                    "scenario": name,
                    "seed": seed,
                    "policy": model.name,
                    **result.report.to_row(),
                    "adaptations": 0,
                    "params": 0,
                }
            )

        for label, policy in policy_suite().items():
            result = run_adaptive(
                panel, GlobalRidge(horizon=horizon), policy, end, label=label
            )
            rows.append(
                {
                    "scenario": name,
                    "seed": seed,
                    "policy": label,
                    **result.report.to_row(),
                    "adaptations": result.n_adaptations,
                    "params": result.total_params_updated,
                }
            )
            if label == "metavyn":
                # Everything from a permanent regime change onward counts as
                # genuinely drifted, so alarms there are not false.
                drifted = truth["shock_months"].copy()
                for regime in truth.get("regimes", ()):
                    drifted[regime.start :] = True
                entry = {
                    "scenario": name,
                    "seed": seed,
                    "far_0.88": false_alarm_rate(result.signals, drifted, 0.88),
                    "far_0.93": false_alarm_rate(result.signals, drifted, 0.93),
                }
                for shock in truth.get("shocks", ()):
                    lat = detection_latency(result.signals, shock.start, 0.88)
                    entry[f"latency_{shock.name}"] = lat
                detect.append(entry)

    return rows, detect


def summarise(rows: list[dict], scenario: str) -> None:
    """Print a policy comparison for one scenario, best MASE first."""
    subset = [r for r in rows if r["scenario"] == scenario]
    by_policy: dict[str, list[dict]] = {}
    for row in subset:
        by_policy.setdefault(row["policy"], []).append(row)

    frozen = by_policy.get("frozen")
    reference = statistics.fmean(r["MASE"] for r in frozen) if frozen else None

    print(f"\n=== {scenario} ===")
    header = f"{'policy':<24}{'MASE':>8}{'sd':>7}{'vs frozen':>11}{'bias':>8}{'adapts':>8}{'params':>9}"
    print(header)
    print("-" * len(header))
    ordered = sorted(by_policy.items(), key=lambda kv: statistics.fmean(r["MASE"] for r in kv[1]))
    for policy, items in ordered:
        mases = [r["MASE"] for r in items]
        mean = statistics.fmean(mases)
        sd = statistics.pstdev(mases)
        delta = "" if reference is None else f"{100 * (mean - reference) / reference:+.2f}%"
        print(
            f"{policy:<24}{mean:8.4f}{sd:7.4f}{delta:>11}"
            f"{statistics.fmean(r['signed_err'] for r in items):8.3f}"
            f"{statistics.fmean(r['adaptations'] for r in items):8.1f}"
            f"{statistics.fmean(r['params'] for r in items):9.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    all_rows: list[dict] = []
    all_detect: list[dict] = []
    for name in SCENARIOS:
        rows, detect = run_scenario(name, args.seeds, args.horizon)
        all_rows += rows
        all_detect += detect
        summarise(rows, name)

    print("\n=== detector (metavyn policy) ===")
    keys = sorted({k for d in all_detect for k in d if k not in ("seed",)})
    for scenario in SCENARIOS:
        subset = [d for d in all_detect if d["scenario"] == scenario]
        if not subset:
            continue
        parts = []
        for key in keys:
            if key == "scenario":
                continue
            vals = [d[key] for d in subset if d.get(key) is not None]
            if vals:
                parts.append(f"{key}={statistics.fmean(vals):.2f}")
        print(f"  {scenario:<18}" + "  ".join(parts))

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "mvp_results.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
