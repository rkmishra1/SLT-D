# SLT-D Pilot Study — Results and Paper Plan

**Model:** Stage life testing with degradation follow-up (SLT-D). Working-tile paper:
*Stage life testing with degradation data: a stage-wise hybrid life–degradation test for
highly reliable products*. All artifacts in this folder: `MODEL.md` (derivation),
`src/` (validated implementation), `tests/` (correctness tests), `results/` (study output).

---

## 1. What was built and validated

| Component | Status | Evidence |
|---|---|---|
| IG-process math (densities, exact first-passage CDF/PDF/quantiles, CE virtual age) | done | 7 unit tests, incl. first-passage CDF vs 60k-path Monte Carlo |
| World simulator (exact inversion sampling; schemes a/b/c as observation masks) | done | failure-rate vs CDF tests; continuation-rate vs G test |
| Scheme likelihoods (a: PC-FCT, b: SLT re-test with latent-level quadrature, c: SLT-D closed form) | done | score identity `E[grad ell]=0` per scheme (SE-scaled); CRN-paired perturbation tests; GL-quadrature vs 2M-draw MC |
| MLE + observed-information + delta-method CI for `log xi_p` | done | multi-start Nelder–Mead, boxed parameters; per-rep fits beat true-theta loglik |
| **External validation on real data** | **done** | our fit of the connector stress-relaxation data `(a0,a1,lam,q) = (-1.894, 1.738, 0.629, 0.449)` vs Ma et al. 2021 published `(-1.88, 1.73, 0.653, 0.449)`; `xi_0.1(40C)`: 119,743 h vs ~116,278 h (3% apart) |

Study: 200 reps × {S1, S2 (π₁ ∈ .25/.5/.75), S3} × 3 schemes = 3,000 fits, common random
numbers across schemes. True θ = real-data fit. Estimand: `log xi_0.1` at use condition (x = 0).

## 2. Headline results (RMSE of log ξ̂₀.₁ ≈ relative error of the quantile)

| Scenario | (a) discard | (b) SLT re-test | (c) SLT-D | ratio b/c |
|---|---|---|---|---|
| **S1** connector-like (85→100 °C, ω=30, d₁≈0) | 45.4 | 27.7 | **0.90** | 31× |
| **S2** moderate (65→85 °C, ω=10, π₁=.5) | 24.1 | 1.59 | **0.50** | 3.2× |
| **S2** π₁=.25 | 27.0 | 3.39 | **2.93** | 1.2× |
| **S2** π₁=.75 | 24.3 | 1.64 | **0.47** | 3.5× |
| **S3** severe (65→85 °C, ω=30, zero failures at s₀) | 0.69* | 11.5 | **0.40** | 28× |

Median absolute errors tell the same story with the heavy tail removed:
S1: (b) 0.84 vs (c) 0.59; S2 π₁=.5: (b) 0.70 vs (c) 0.32; S3: (b) 0.22 vs (c) 0.29 (medians
close, but (b) has catastrophic p90 = 17.2 vs (c) 0.68 — (b)'s failure-time-only continuation
data occasionally leave the likelihood with near-flat directions).

*(a)-S3 footnote: with zero failures anywhere on stage s₀, scheme (a)'s likelihood contains only
survival terms at a single stress; extrapolation to x = 0 is **not identifiable**. The tame RMSE
shown is an artifact of truth-anchored optimizer starts landing at similar ridge points
(bias +0.60, SE never finite). In S1/S2, where some failure data exist, the same scheme produces
RMSE 24–45. For the paper: report (a) via a profile-likelihood ridge plot, not RMSE.*

### Findings (paper-ready claims)

1. **The withdrawn units carry all the stress-transfer information.** Discarding them (scheme a)
   leaves single-stress data and makes use-condition extrapolation unidentifiable — this is a
   structural, not numerical, statement, and it strengthens the motivation of both SLT
   (Laumen–Cramer) and hybrid ATs (Ma et al.).
2. **Degradation follow-up dominates failure re-testing** — 3–31× lower RMSE of `log ξ̂₀.₁` at
   equal (n, τ₁, T_max) — and never produces the catastrophic fits that plague scheme (b)
   (p90 absolute error 17.2 vs 0.68 in S3).
3. **The gain is largest exactly where existing methods fail:** S1/S3 (highly reliable product,
   no early failures) — the regime that motivates accelerated testing in the first place. In
   S3 the life test yields *zero* failures, yet SLT-D still estimates the use-condition quantile
   to within ~±40% (log-scale 0.40) with 30 units.
4. **Withdrawal fraction:** scheme (c) improves monotonically in π₁ (RMSE 2.93→0.47 from
   π₁=.25→.75 in S2): the more units are recycled into degradation measurement, the better —
   provided stage s₀ retains enough units for its (censoring) information.
5. **Inference quality:** scheme (c) always yields finite observed information (100%) and Wald
   CIs are conservative (empirical coverage 1.000); scheme (b) has unstable curvature
   (SE-finite 78–93%).

### Cost caveat (must be stated in the paper)
Schemes were compared at equal (n, τ₁, T_max). Scheme (c) additionally costs L readings per
withdrawn unit (L = 20 in S1/S3, 60 in S2); scheme (b) costs re-test chamber time. A cost-based
comparison (V-optimality under a budget like Wang 2016 / Ma 2021) is the natural sequel and the
main design-theory contribution of the planned full paper.

## 3. Figures

- `results/fig1_rmse_by_scheme.png` — RMSE by scheme × scenario.
- `results/fig2_pi_sweep.png` — RMSE/median-AE vs π₁ (S2).
- `results/summary.csv`, `results/per_rep.csv` — full aggregates and per-rep records.

## 4. Paper outline (target: RESS / Technometrics)

1. **Introduction.** Progressive censoring discards information; SLT recycles withdrawn units as
   failure re-tests; hybrid ATs (Ma et al.) split units between ALT/ADT *in parallel*. We make
   the hybrid *sequential*: withdrawn units become the degradation cohort (unit recycling).
2. **The SLT-D test scheme** (MODEL.md §2) + path-level CE remark (two CE notions; lifetime-shift
   CE of SLT vs path-level CE needed for degradation data).
3. **Likelihood theory** (MODEL.md §3): ignorable-withdrawal lemma (block-independence analogue
   of Laumen–Cramer Lemma 3.3); closed-form scheme-(c) likelihood; latent-level integrals for
   scheme (b); identifiability discussion (why scheme (a) cannot extrapolate).
4. **Estimation & asymptotics:** MLE, observed information, delta-method for `xi_p`; EM variant
   when no baseline reading is taken (MODEL.md remark — to be written up).
5. **Simulation study** (this pilot, scaled up): hierarchy (a)<(b)<(c); π₁ sweep; sensitivity to
   planning-parameter misspecification (à la Wang 2016 Table 5).
6. **Case study:** connector stress-relaxation data — model fit (validated vs Ma et al.),
   then *what an SLT-D on this product would have delivered* vs SLT vs discarding.
7. **Optimal design outlook:** choosing (τ₁, π₁, L, x₁) by V-optimality under budget — the
   sequel paper (this is where the RESS-style design contribution lands).

## 5. Immediate next steps

1. Scale reps to 1,000 for the paper tables; add sensitivity analysis (±10% planning-parameter
   errors, as in both RESS papers).
2. Write the EM variant for "no baseline reading" (variant B in MODEL.md §3).
3. Ridge/profile-likelihood figure for scheme (a) demonstrating unidentifiability.
4. Analytic Fisher-information decomposition `I = I_fail + I_deg` (planning-version asymptotic
   variance) — needed for the design-optimization sequel.
5. R implementation for reproducibility (both anchor papers use R).
