# Metavyn

Capstone project. Drift-aware, severity-routed adaptation for humanitarian
demand forecasting.

This is an MVP. It implements the full loop end to end — generate a monthly
`district x indicator` panel, forecast it, detect drift, grade that drift into a
severity score, route the severity to a *depth* of model adaptation, and score
the result on metrics that suit sparse, shock-driven demand — so that the
central claim can actually be tested rather than assumed.

**The headline result is negative, and it is the most useful thing here.** On
every synthetic testbed in this repository, severity-routed adaptation does not
beat leaving the model alone. The diagnosis is in
[What the MVP found](#what-the-mvp-found), and it has direct consequences for
how the project should be scoped.

## Install and run

```bash
pip install -e .
metavyn demo --scenario regime          # one scenario, month by month
python experiments/run_mvp.py --seeds 8 # the full comparison
pytest                                  # 38 tests
```

Only numpy is required. PyTorch is optional and not needed for anything below.

## The loop

```
synthetic panel ──► forecaster ──► residuals ──► drift detector ──► severity
                        ▲                                              │
                        └────────── adaptation tier ◄── routing policy ┘
```

| Module | What it does |
| --- | --- |
| `synthetic.py` | Panels with shocks and regime changes placed by construction, so detection latency is measurable against ground truth |
| `features.py` | Lag, seasonal and exogenous features, centred per series, with an explicit forecast origin for multi-step horizons |
| `ridge.py` | Pooled ridge forecaster with a four-rung adaptation ladder |
| `drift.py` | Three detection channels, graded into one calibrated severity score |
| `router.py` | The policies being compared, including the ones from the literature |
| `metrics.py` | MASE, sMAPE, pinball, coverage, and signed bias, segmented by intermittency |
| `backtest.py` | Rolling-origin driver that never reads past `t` |

### The adaptation ladder

| Tier | Updates | Parameters |
| --- | --- | --- |
| `none` | nothing | 0 |
| `bias` | per-series intercepts, shrunk toward their current values | N |
| `local_lowrank` | a correction applied **only to the series flagged as drifting** | d + \|flagged\| |
| `lowrank` | a rank-r subspace of the shared weights | r + N |
| `full` | all shared weights, refitted with a replay buffer | d + N |

### Severity grading

Three channels, each ranked against a null distribution collected on quiet
months, then blended and ranked again:

- **input** — shift of the cross-sectional feature mean against the trailing
  twelve months. The reference *rolls*; against a fixed reference a slow trend
  saturates the channel within two years and it stops detecting anything.
- **residual** — tail mean of the standardised residual over the worst 25% of
  series. Disasters are local, and a shock hitting a third of districts is
  invisible to a panel average.
- **persistent** — an EWMA separating a one-month spike from a regime change.

## What the MVP found

### 1. Adaptation does not beat doing nothing

MASE over 8 seeds, holdout = last 40% of each panel. `frozen` is the same model
never updated after training.

| Policy | transient shock | regime change | short history | params written |
| --- | --- | --- | --- | --- |
| `quarterly_full` | **0.9334** (−0.46%) | **0.8963** (−0.22%) | **0.8882** (−0.86%) | 1376 |
| `binary_full` | 0.9333 (−0.47%) | 0.8963 (−0.22%) | 0.8941 (−0.20%) | 1075 |
| `always_full` | 0.9338 (−0.41%) | 0.8969 (−0.16%) | 0.8890 (−0.77%) | 4128 |
| `frozen` | 0.9377 | 0.8983 | 0.8959 | 0 |
| `metavyn` | 0.9435 (+0.62%) | 0.9107 (+1.38%) | 0.8967 (+0.09%) | 610 |
| `always_lowrank` | 0.9476 (+1.05%) | 0.9151 (+1.87%) | 0.8977 (+0.20%) | 3648 |
| `croston_sba` | 1.0246 | 1.0131 | 0.9559 | 0 |
| `weighted_moving_average` | 1.1036 | 1.1187 | 1.1069 | 0 |
| `seasonal_naive` | 1.2435 | 1.1587 | 1.1345 | 0 |

Seed standard deviation is 0.017–0.039, so **every adaptive policy is inside
noise of the frozen model.** Full refits are very slightly ahead; every
incremental tier is slightly behind. The severity router's one real win is
cost: it writes 610 parameters against 1376 for quarterly retraining, for
indistinguishable accuracy.

### 2. Why: lag features absorb the drift for free

This was traced rather than guessed, and the decisive experiment is the oracle.
On the regime-change scenario, measuring only the struck districts after the
change:

| Model | MASE on affected series |
| --- | --- |
| frozen, fitted before the regime change | 1.2327 |
| **oracle, refitted on post-change data** | **1.2323** |
| control: same series, no regime change at all | 0.8837 |

The regime change costs 0.35 MASE, and an oracle with perfect hindsight
recovers **none of it**. The reason is that the drift raised the *variance* of
the series, and no parameter update recovers irreducible noise. Replacing the
variance change with a learnable one (a three-month seasonal phase shift) did
not help either: a model carrying `lag1` tracks a shifted seasonal peak by
itself within a month or two. Holding the horizon at 1, 3 and 6 months changed
nothing.

So for one-step-ahead forecasting of a monthly panel with lag features, there
is very little for parameter adaptation to recover. This is the
"When to Retrain" counterexample, reproduced.

### 3. The point estimate is a real decision, not a detail

Fitting in `log1p` space and back-transforming gives the conditional **median**.
For right-skewed counts that sits below the mean, which shows up as a signed
error of about −0.44 MASE even when the model is correct.

| `point_estimate` | MASE | signed error |
| --- | --- | --- |
| `median` (default) | 0.847 | −0.358 |
| `mean` (Duan smearing) | 0.896 | +0.076 |

The median minimises the MAE/MASE family that gets reported; the mean is the
safer target for stock quantification, where under-forecasting a medicine is
not symmetric with over-forecasting it. The default optimises the metric, which
is not obviously the right choice for the application.

### 4. Detection works, and its cost scales with history length

Latency is measured from the shock month to the first severity reading ≥ 0.88.

| Scenario | detection latency | false alarms on quiet months |
| --- | --- | --- |
| transient shock | 0.25 months | 37% |
| regime change | 0.12 months | 24% |
| short history (6 years) | 2.0 months | 6% |

Detection is close to immediate when there is enough history to calibrate
against, and degrades to two months on a realistic six-year extract. The
false-alarm rates run well above nominal: severity is a rank against a null
estimated from calibration months and scored in-sample, and later months drift
away from that null as trend accumulates. Thresholds are therefore approximate
false-alarm rates, not exact ones — measure with `backtest.false_alarm_rate`
rather than assuming.

## What this means for the project

1. **Do not build sixteen weeks on "adaptation beats retraining" without first
   reproducing a setting where adaptation beats *anything*.** The cheapest next
   experiment is to find a generator where the oracle gap is large. If an oracle
   refitted on post-change data cannot beat a frozen model, no policy can, and
   the experiment cannot answer the question it was built to answer.
2. **The defensible contribution is cost, not accuracy.** Severity routing
   matched quarterly retraining while writing 44% as many parameters. That is a
   real claim, it is measurable, and it does not depend on winning on error.
3. **A backbone with d=14 cannot demonstrate parameter efficiency.** A rank-4
   restriction of fourteen weights saves nothing worth having, and it cannot
   represent a change confined to one coefficient. The PEFT argument needs the
   transformer; the routing argument does not, which is why the MVP separates
   them.
4. **Scope may matter more than depth.** A global weight update cannot fix a
   regime change confined to a third of districts — it can only find a
   compromise that helps those and hurts the rest. `local_lowrank` exists to
   test this and did not pay off here, but the axis is the more promising one.

## Honest limitations

- Everything is synthetic. No Tamil Nadu HMIS data has been loaded or validated,
  so nothing here says what real data would do.
- The ridge backbone is a stand-in. `base.Forecaster` is the seam a transformer
  with LoRA adapters would slot into; it has not been written.
- Scenarios were chosen to exercise the loop, not sampled from a realistic
  distribution of disasters.
- Eight seeds is enough to show that differences are inside noise. It is not
  enough to establish a small real difference if one exists.

## Licence

MIT.
