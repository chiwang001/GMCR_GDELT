# Dynamic Preference Estimation and GMCR Analysis of China-US Trade Conflict

This is the public, reproducible package for the dynamic-preference GMCR
analysis. The checked-in implementation follows the revised Demo4 model and
includes the model inputs, auditable GDELT extracts, 500 retained parameter
solutions, tests, and curated numerical and graphical results.

## Study scope

The empirical period is January 2017 through January 2020. The observed path
contains 37 monthly states from a graph of 76 feasible states and 411 directed
transitions. China and the United States each have five preference drivers.
The dynamic model estimates 42 core parameters:

- 30 theory-prespecified active event-to-driver coefficients (15 per actor);
- 2 persistence coefficients; and
- 10 long-run preference means.

The revised model uses fixed `0.6/0.3/0.1` distributed lags, bounded mean
reversion, within-actor/month utility standardization, smooth softmax support
for observed state changes, continuous SEQ-stability shortfall for unchanged
months, bounded forward-explanation shortfall, and a fixed state-score
perturbation process. Inactive mapping coefficients are fixed at zero and
active coefficients obey directional sign constraints.

The February 2020 forecast is a one-step extrapolation using information
available through January 2020. It does not impute a February event shock.
Pairwise preference entries are ranking frequencies across retained solutions;
they are not action probabilities or posterior probabilities.

## Repository layout

| Path | Contents |
| --- | --- |
| `DynamicGMCR/cod/` | Calibration, prediction, GMCR stability, robustness, plotting, and static-baseline scripts |
| `DynamicGMCR/tests/` | Focused unit and integration tests for the current implementation |
| `DynamicGMCR/input/` | State tables, transition graph, observed path, and monthly event-impact panel |
| `DynamicGMCR/results/` | Curated parameter, prediction, baseline, ablation, convergence, and robustness results |
| `GDELT_DATA/code/` | GDELT download/filter and monthly feature-construction scripts |
| `GDELT_DATA/input/` | 37 directed China-US monthly event extracts |
| `GDELT_DATA/output/` | Event audit, feature panels, CAMEO mapping, QC, and provenance summary |

Generated solver runs belong under `DynamicGMCR/output/` and are ignored by
Git. The published result tables under `DynamicGMCR/results/` are the inputs
used by the reproducibility commands below.

## Environment

The package was verified with Python 3.13.2. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Reproduction

Run the tests first:

```powershell
\.venv\Scripts\python -m pytest -q DynamicGMCR\tests
```

Rebuild the monthly GDELT feature panel from the included event extracts:

```powershell
\.venv\Scripts\python GDELT_DATA\code\aggregate_monthly_domain_shocks.py
```

The original GDELT monthly extracts can be downloaded again (the complete
range is network- and time-intensive):

```powershell
\.venv\Scripts\python GDELT_DATA\code\demo1_201701_202001_monthly.py `
  --start-date 20170101 --end-date 20200131 --scope trade_context
```

Run a small end-to-end calibration with the revised Demo4 solver:

```powershell
\.venv\Scripts\python DynamicGMCR\cod\solve_core_preference_model.py --preset quick
```

The paper-scale presets are `full`, `two_hour`, and `deep`. They require
substantially more CPU time. Each run writes a timestamped directory under
`DynamicGMCR/output/`.

Recompute the February 2020 preference matrices from the published solution
ensemble:

```powershell
\.venv\Scripts\python DynamicGMCR\cod\predict_preference_matrix.py `
  --results-dir DynamicGMCR\results\multiseed_parameters `
  --output-dir DynamicGMCR\output\reproduced_preference_prediction
```

Run probability-form GMCR stability on the reproduced matrices:

```powershell
\.venv\Scripts\python DynamicGMCR\cod\gmcr_probability_stability.py `
  --cn-matrix DynamicGMCR\output\reproduced_preference_prediction\predicted_preference_matrix_CN_202002.csv `
  --us-matrix DynamicGMCR\output\reproduced_preference_prediction\predicted_preference_matrix_US_202002.csv `
  --graph DynamicGMCR\input\state_transition_graph.csv `
  --output-dir DynamicGMCR\output\reproduced_probability_stability
```

Audit finite-ensemble robustness by repeatedly subsampling the 500 retained
solutions. The checked-in report and figures are under
`DynamicGMCR/results/equilibrium_robustness/`:

```powershell
\.venv\Scripts\python DynamicGMCR\cod\analyze_equilibrium_robustness.py `
  --project-root DynamicGMCR `
  --output-dir DynamicGMCR\output\equilibrium_robustness `
  --subset-sizes 10,50,100,200,300,400,490 `
  --replicates 100 `
  --sampling-seed 20261004
```

Use `--quick` for a smoke run. The analysis reports matrix error, GMCR margin
error, actor-state-concept agreement, joint-equilibrium Jaccard similarity,
threshold-separation certificates, and threshold sensitivity. The result is a
conditional finite-ensemble robustness analysis, not a claim of parameter
identification or statistical consistency.

Reproduce the self-contained static strategy-priority baseline:

```powershell
\.venv\Scripts\python DynamicGMCR\cod\run_static_strategy_baseline.py
\.venv\Scripts\python DynamicGMCR\cod\generate_static_baseline_figures.py
\.venv\Scripts\python DynamicGMCR\cod\compare_static_dynamic_all_seeds.py
\.venv\Scripts\python -m unittest DynamicGMCR.tests.test_static_strategy_baseline -v
```

Run the controlled component ablations and build the multi-objective table:

```powershell
\.venv\Scripts\python DynamicGMCR\cod\run_ablation_experiments.py `
  --output-dir DynamicGMCR\output\ablation\full_budget --preset full
\.venv\Scripts\python DynamicGMCR\cod\generate_multiobjective_ablation_table.py
```

The ablations are `full`, `no_historical_memory`, `no_forward_reward`,
`no_score_perturbation`, `no_nonlinear_saturation`, and
`no_stability_penalty`. Their checked-in summaries are in
`DynamicGMCR/results/ablation/full_budget/`.

## GDELT construction and limitations

The raw extracts originate from the GDELT 2.0 Event Database. The downloader
retains directed China-to-United States and United States-to-China root events
and records the 15-minute source timestamp. The monthly processor uses that
file timestamp as the information-arrival month, retains `SQLDATE` as the
referenced event date, and applies a timeliness weight. The model input uses
the `trade_context` scope.

The extracts do not include article text, GKG themes, or the GDELT Mentions
table. Generic diplomatic statements therefore cannot be verified as
trade-specific events. Interpret the event-derived drivers within this scope.

## Provenance

GDELT source: <https://www.gdeltproject.org/>. Downloadable event files:
<https://data.gdeltproject.org/gdeltv2/>. Cite GDELT and comply with its
applicable terms when redistributing or extending the extracts. No software or
data license is asserted by this package; add the intended license and final
paper citation metadata before publication.
