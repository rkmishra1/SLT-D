<div align="center">

# SLT-D — Stage Life Testing with Degradation Data

**A stage-wise hybrid life–degradation test for highly reliable products**

Code, simulation results, and manuscript for the paper by
**Ramakrushna Mishra**, Decision Sciences Area, IIM Lucknow

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-scientific-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-stats-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Manuscript](https://img.shields.io/badge/Manuscript-PDF-B31B1B)](SLT-D_manuscript.pdf)

</div>

---

## Overview

In a progressively censored life test, the units withdrawn at the stage-change time are
usually discarded — or, in the **stage life testing (SLT)** of Laumen & Cramer (2019),
re-tested to failure at higher stress. **SLT-D** does neither.

Withdrawn units are moved to an instrumented rig at an elevated stress, where a baseline
reading and periodic **degradation measurements** are taken until the end of the test. The
same physical units contribute **failure-time information first and degradation information
afterwards** — recovering usable inference even when the ordinary life test produces few or
zero failures.

### Model

- Inverse Gaussian degradation process with power-law time transformation
- Path-level cumulative exposure
- Exact likelihood + identifiability proofs
- Planning Fisher information with an **alive-weighting** correction for mid-test failures of withdrawn units
- V-optimal plan characterization, robustness study (L9 arrays), and Bayesian planning under a pilot-informed prior

### Case studies

| Dataset | Source |
|---|---|
| Stress relaxation of electrical connectors | Yang (2007), Ex. 8.7 — same data as Ma et al. (2021) |
| GaAs laser diodes | Meeker & Escobar (1998), p. 339 |

---

## Key results

Log-scale RMSE of the use-condition 0.1-quantile, across three designs:

| Scenario | (a) discard | (b) SLT re-test | (c) **SLT-D** |
|---|---:|---:|---:|
| S1 — connector-like | 45.4 | 27.7 | **0.84** |
| S2 — moderate | 24.1 | 1.82 | **0.53** |
| S3 — severe (zero failures) | *unidentifiable* | 8.95 | **0.41** |

**Constrained-optimal plan** for the connector case (budget \$6,000):
stage 0 at $x_0 = 0.10$, withdrawal at 1750 h with $\pi_1 = 0.90$, rig at $x_1 = 0.85$,
readings every 150 h, $n = 20$; $\mathrm{Avar}(\log \xi_{0.1}) = 0.036$ (±21 % on the
quantile). The price of the sequential constraint relative to an unconstrained parallel ADT
is a factor of ≈ 5.5 in asymptotic variance.

See the [manuscript](SLT-D_manuscript.pdf) for full details.

---

## Repository layout

```
src/                  model, likelihoods, simulators, MLE, design theory, studies
tests/                correctness tests (score identities, quadrature vs MC, K=1 equivalence, ...)
scripts/              figure generation, design optimization, diagnostics
results/              generated tables (.tex), figures (.png), per-replication CSVs, JSONs

MODEL.md              formal model and likelihood derivation
RESULTS.md            study results and findings
SLT-D_manuscript.pdf  compiled manuscript
data_GaAsLaser.csv    GaAs laser degradation dataset
```

---

## Reproduction

Requires **Python 3** with `numpy`, `scipy`, `pandas`, `matplotlib`.

```bash
pip install -r requirements.txt
```

<details>
<summary><b>Full pipeline</b> (click to expand)</summary>

```bash
# correctness tests (~5 min)
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
```

</details>

### Data

`data_GaAsLaser.csv` comes from Meeker's public repository of repeated-measured degradation
datasets (file `GaAsLaser.csv`).

---

## Citation

> Mishra, R. *Stage life testing with degradation data: a stage-wise hybrid
> life–degradation test for highly reliable products.*

```bibtex
@unpublished{mishra-sltd,
  author = {Mishra, Ramakrushna},
  title  = {Stage life testing with degradation data: a stage-wise hybrid
            life--degradation test for highly reliable products},
  note   = {Decision Sciences Area, IIM Lucknow},
  url    = {https://github.com/rkmishra1/SLT-D}
}
```

---

## References

- Laumen, B., & Cramer, E. (2019). Stage life testing. *Naval Research Logistics*.
- Ma, Z., et al. (2021). *(connector stress-relaxation analysis)*.
- Meeker, W. Q., & Escobar, L. A. (1998). *Statistical Methods for Reliability Data*. Wiley.
- Yang, G. (2007). *Life Cycle Reliability Engineering*. Wiley.
