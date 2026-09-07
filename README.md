# SLT-D — Stage Life Testing with Degradation Data

Code, simulation results, and manuscript for the paper

> **Stage life testing with degradation data: a stage-wise hybrid life–degradation test for
> highly reliable products**
> Ramakrushna Mishra, Decision Sciences Area, IIM Lucknow

In a progressively censored life test, the units withdrawn at the stage-change time are not
discarded and not re-tested to failure (as in stage life testing of Laumen & Cramer, 2019);
instead they are moved to an instrumented rig at an elevated stress, where a baseline reading
and periodic degradation measurements are taken until the end of the test. The same physical
units contribute failure-time information first and degradation information afterwards.

The model uses an inverse Gaussian degradation process with power-law time transformation and
path-level cumulative exposure. The paper derives the exact likelihood, proves
identifiability results, derives a planning Fisher information (with an "alive-weighting"
correction for mid-test failures of withdrawn units), characterizes V-optimal plans, and
studies robustness (L9 arrays) and Bayesian planning under a pilot-informed prior. Two case
studies anchor the work: stress relaxation of electrical connectors (Yang, 2007, Ex. 8.7 —
the same data as Ma et al., 2021) and GaAs laser diodes (Meeker & Escobar, 1998, p. 339).

## Repository layout

```
paper/     manuscript.tex (source) and manuscript.pdf (compiled)
src/       model, likelihoods, simulators, MLE, design theory, studies
tests/     correctness tests (score identities, quadrature vs MC, K=1 equivalence, ...)
scripts/   figure generation, design optimization, diagnostics
results/   generated tables (.tex fragments), figures (.png), per-replication CSVs, JSONs
MODEL.md    formal model and likelihood derivation
RESULTS.md  study results and findings
```

## Key results (see the manuscript for details)

| Scenario (log-scale RMSE of the use-condition 0.1-quantile) | (a) discard | (b) SLT re-test | (c) SLT-D |
|---|---|---|---|
| S1 connector-like | 45.4 | 27.7 | **0.84** |
| S2 moderate | 24.1 | 1.82 | **0.53** |
| S3 severe (zero failures) | unidentifiable | 8.95 | **0.41** |

Constrained-optimal plan for the connector case (budget $6,000): stage 0 at x0 = 0.10,
withdrawal at 1750 h with pi1 = 0.90, rig at x1 = 0.85, readings every 150 h, n = 20;
Avar(log xi_0.1) = 0.036 (+-21% on the quantile). The price of the sequential constraint
relative to an unconstrained parallel ADT is a factor of ~5.5 in asymptotic variance.

## Reproduction

Requires Python 3 with numpy, scipy, pandas, matplotlib (`pip install -r requirements.txt`).

```bash
# run the correctness tests (~5 min)
python3 -m unittest discover -s tests

# main simulation study (1000 reps x 5 design cells x 3 schemes; ~1 h, parallel)
PYTHONPATH=. python3 src/study.py 1000 results/per_rep_1000.csv

# figures and LaTeX tables from the study output
PYTHONPATH=. python3 src/make_tables.py
PYTHONPATH=. python3 scripts/make_figures_extra.py

# design optimization, Bayesian planning, laser case study
PYTHONPATH=. python3 scripts/design_optimize.py
PYTHONPATH=. python3 src/bayes_design.py
PYTHONPATH=. python3 src/study_laser.py 500

# manuscript
cd paper && latexmk -pdf manuscript.tex
```

The GaAs laser data (`data_GaAsLaser.csv`) come from Meeker's public repository of
repeated-measured degradation datasets (file GaAsLaser.csv).
