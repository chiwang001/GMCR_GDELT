# Dynamic Preference Estimation and GMCR Analysis of China-US Trade Conflict

This repository is the compact public research package for a paper on dynamic
preference estimation in the Graph Model for Conflict Resolution (GMCR). It
contains the complete analysis code, model inputs, auditable GDELT event data,
the 500 fitted parameter sets used in the reported robustness analysis, tests,
and curated numerical and graphical results. Development caches, temporary
files, logs, duplicate figures, intermediate optimizer histories, and paper
drafts are intentionally excluded.

## Study scope

The empirical period is January 2017 through January 2020. The observed GMCR
path contains 37 monthly states selected from 76 feasible states connected by
411 directed transitions. China and the United States each have five preference
drivers. The dynamic model estimates 42 parameters:

- 30 theory-prespecified active event-to-driver coefficients (15 per actor);
- 2 persistence coefficients; and
- 10 long-run preference means.

The model uses fixed `0.6/0.3/0.1` distributed lags, bounded mean reversion,
within-actor/month utility standardization, smooth softmax compatibility for
observed state changes, continuous SEQ-stability shortfall for unchanged months,
and fixed state-score perturbations. Inactive event-to-driver coefficients are
fixed at zero and active coefficients follow directional sign constraints.

### Static preference acquisition

The static baseline obtains preferences from theory-informed strategy-priority
declarations rather than from a fitted utility or a random draw. For actor
`i`, declaration `p` is encoded as a binary satisfaction indicator
`b_ip(s)`. The fixed lexicographic order is represented by
`V_i(s) = sum_p 2^(8-p) b_ip(s)`, giving weights `128, 64, ..., 1`.
These weights only preserve declaration order; they are not estimated
preference intensities. The complete declaration table and Chinese paper-style
description are in
`DynamicGMCR/results/static_baseline/静态模型历史转移相容性分析.md`, with the
machine-readable declarations in `strategy_priority_declarations.csv`.

The one-step preference forecast is for February 2020. It uses information
available through January 2020 and does not impute a February event shock.

## Repository contents

| Path | Contents |
| --- | --- |
| `DynamicGMCR/cod/` | Dynamic GMCR estimation, prediction, stability, robustness, plotting, and static-baseline code |
| `DynamicGMCR/tests/` | 34 focused tests for model structure, probability matrices, stability, multi-seed execution, ablations, and visualization utilities |
| `DynamicGMCR/input/` | The five model-ready state, graph, path, and event-impact tables |
| `DynamicGMCR/results/multiseed_parameters/` | The 500 fitted parameter JSON files used in the paper analyses |
| `DynamicGMCR/results/parameter_distribution/` | Long-form 42-parameter samples, summaries, and figures |
| `DynamicGMCR/results/preference_prediction/` | February 2020 preference matrices, seed-level preferences/utilities, figures, and GMCR stability results |
| `DynamicGMCR/results/preference_matrix_convergence/` | Nested-seed convergence distances, final 500-seed matrices, and figure |
| `DynamicGMCR/results/preference_matrix_independent_sampling/` | Independent subset-sampling replicates, quantiles, final matrices, and figure |
| `DynamicGMCR/results/static_baseline/` | Declared-option static GMCR baseline and comparisons with all dynamic seeds |
| `DynamicGMCR/results/ablation/` | Full-budget component ablations with parameters, monthly fit, optimization history, summaries, and the complete nine-objective table |
| `GDELT_DATA/code/` | Reproducible GDELT download/filter and monthly feature construction scripts |
| `GDELT_DATA/input/` | 37 monthly directed China-US event extracts |
| `GDELT_DATA/output/` | Event audit, monthly feature panels, CAMEO mapping, QC, and processing summary |
| `GDELT_DATA/results/` | Download provenance summary |

The two concise Markdown reports under `DynamicGMCR/results/` document the
static declaration baseline and the multi-objective ablation table. Other
generated Markdown, caches, and large run directories remain excluded.

## GDELT data construction

The raw extracts come from the GDELT 2.0 Event Database. The downloader retains
directed China-to-United States and United States-to-China root events and stores
the 15-minute source timestamp for provenance. The monthly processor uses the
file/`SourceFileTimestamp` month as the information-arrival month; `SQLDATE` is
retained as the referenced-event date and contributes to a timeliness weight.
This prevents later reporting of an older event from being backdated into the
forecasting record.

Three event scopes are retained:

- `all_bilateral`: every valid directed bilateral root event;
- `trade_context`: economic/coercive actions, relevant actor types, and selected
  legal or regulatory actions, with lower weights for lower-confidence context;
- `direct_trade`: high-precision CAMEO actions for economic cooperation, aid,
  sanctions, embargoes, administrative restrictions, and property actions.

The supplied data contain 101,398 valid bilateral root events, including 19,947
`trade_context` events and 11,408 `direct_trade` events. The processed long-form
panel contains 1,332 month-country-domain-scope rows. Direction, cooperation and
conflict components, capped attention measures, Goldstein scale, QuadClass,
tone, actor and action geography, timeliness, past-only robust z-scores,
innovations, and decayed stocks are retained. The model input uses the
`trade_context` scope.

The extracts do not include article text, GKG themes, or the GDELT Mentions
table. Generic consultations and diplomatic statements therefore cannot be
verified as trade-specific events. This limitation should be retained when
interpreting the event scope.

## Environment

The package was verified with Python 3.13.2. Create an isolated environment
from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The pinned numerical and plotting dependencies are in `requirements.txt`.

## Reproduction

Run the test suite first:

```powershell
.\.venv\Scripts\python -m pytest -q DynamicGMCR\tests
```

Rebuild all GDELT monthly features from the included event extracts. This also
refreshes `DynamicGMCR/input/monthly_country_domain_impact.csv`:

```powershell
.\.venv\Scripts\python GDELT_DATA\code\aggregate_monthly_domain_shocks.py
```

The original monthly extracts can be downloaded again from GDELT. The complete
range is network- and time-intensive; `--max-files` can be used for a smoke run.

```powershell
.\.venv\Scripts\python GDELT_DATA\code\demo1_201701_202001_monthly.py `
  --start-date 20170101 --end-date 20200131 --scope trade_context
```

Run a small end-to-end model calibration:

```powershell
.\.venv\Scripts\python DynamicGMCR\cod\solve_core_preference_model.py --preset quick
```

The paper-scale configurations are `full`, `two_hour`, and `deep`. They require
substantially more CPU time than `quick`. Generated solver runs are written to
`DynamicGMCR/output/` and are ignored by Git.

Recompute the February 2020 aggregate preference matrices directly from the 500
published parameter sets:

```powershell
.\.venv\Scripts\python DynamicGMCR\cod\predict_preference_matrix.py `
  --results-dir DynamicGMCR\results\multiseed_parameters `
  --output-dir DynamicGMCR\output\reproduced_preference_prediction
```

Run the probability-based and core-utility GMCR analyses on the reproduced
tables:

```powershell
.\.venv\Scripts\python DynamicGMCR\cod\gmcr_probability_stability.py `
  --cn-matrix DynamicGMCR\output\reproduced_preference_prediction\predicted_preference_matrix_CN_202002.csv `
  --us-matrix DynamicGMCR\output\reproduced_preference_prediction\predicted_preference_matrix_US_202002.csv `
  --graph DynamicGMCR\input\state_transition_graph.csv `
  --output-dir DynamicGMCR\output\reproduced_probability_stability

.\.venv\Scripts\python DynamicGMCR\cod\gmcr_core_stability_from_utilities.py `
  --utilities DynamicGMCR\output\reproduced_preference_prediction\predicted_utilities_by_seed_202002.csv `
  --input-dir DynamicGMCR\input `
  --output-dir DynamicGMCR\output\reproduced_core_stability
```

Reproduce the static strategy-priority baseline and its publication figures
(the checked-in results are already in `DynamicGMCR/results/static_baseline/`):

```powershell
.\.venv\Scripts\python DynamicGMCR\cod\run_static_strategy_baseline.py `
  --output-dir DynamicGMCR\results\static_baseline
.\.venv\Scripts\python DynamicGMCR\cod\generate_static_baseline_figures.py `
  --output-dir DynamicGMCR\results\static_baseline
```

Run the full-budget component ablations. The public result set includes the
new `no_stability_penalty` variant:

```powershell
.\.venv\Scripts\python DynamicGMCR\cod\run_ablation_experiments.py `
  --output-dir DynamicGMCR\output\ablation\full_budget --preset full
.\.venv\Scripts\python DynamicGMCR\cod\generate_multiobjective_ablation_table.py `
  --output-root DynamicGMCR\output `
  --ablation-dir DynamicGMCR\output\ablation\full_budget `
  --static-dir DynamicGMCR\results\static_baseline
```

The resulting table reports transition support, mean conditional support, four
retention rates, forward shortfall, stability shortfall, and preference-matrix
dispersion separately. Higher is better for support and retention metrics;
lower is better for the two shortfalls and dispersion. Full model entries are
seed means with 5%--95% quantiles; fixed-seed ablations and the static baseline
are reported as single values. No aggregate score is constructed. The
definitions and the distinction between `loss`, `transition objective`, and
`summed transition probability` are recorded in
`DynamicGMCR/results/ablation/full_budget/multiobjective_ablation_table.md`.

Every entry point supports `--help`. Commands should be run from the repository
root so that the documented relative paths are preserved.

## Curated results

The public result package retains the data behind each figure as CSV or JSON and
keeps PNG for GitHub preview plus PDF for publication use. The static-baseline
figures additionally retain SVG so that the Times New Roman text remains
editable in vector workflows. Repeated intermediate matrices are omitted. The
final 500-seed matrices and all independent-sampling replicate statistics are
included.

For the historical comparison, the static model explains 45.45% of the 11
changed transitions with a mean probability of 0.1730 and has an unchanged-month
SEQ rate of 0.24. Across 500 dynamic estimates, the median changed-transition
positive rate is 1.00, median changed-transition probability is 0.4426, and the
median unchanged-month SEQ rate is 0.52. The complete seed-level metrics and
uncertainty intervals are in `DynamicGMCR/results/static_baseline/`.

## Provenance and public release

The GDELT source is <https://www.gdeltproject.org/> and the downloadable Event
files are hosted at <https://data.gdeltproject.org/gdeltv2/>. Users should cite
GDELT and comply with its applicable terms when redistributing or extending the
event extracts. The local event audit and download summary provide row-level and
file-level provenance for this package.

No software or data license is asserted by this repository package. The
repository owner should add the intended license and the paper's final author,
title, DOI, and citation metadata when those publication details are available.
