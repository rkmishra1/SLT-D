# Stage Life Testing with Degradation Follow-up (SLT-D)

**Working title of the paper:** *Stage life testing with degradation data: a stage-wise hybrid
life–degradation test for highly reliable products*

**Positioning.** Laumen & Cramer (2019, *NRL*) introduced stage life testing (SLT): in a
progressively censored life test, the units withdrawn at a stage-change time are re-tested on a
second stage and their *failure times* are incorporated in the inference. Ma et al. (2021,
*RESS*) introduced the *parallel* hybrid accelerated test: scarce test units are split between an
ALT arm (run to failure at the highest stress) and an ADT arm (degradation measurements at lower
stresses). This note develops the **sequential** counterpart: the withdrawn units of a stage life
test are not run to failure but are **degradation-monitored** on the second stage. The same
physical units thus deliver failure-time information first and degradation information
afterwards ("unit recycling"), which is attractive precisely when units are expensive and few.

---

## 1. Degradation model and failure definition

Let `{Y_x(t), t >= 0}` be an inverse Gaussian (IG) process under standardized stress `x in [0,1]`
(`x = 0` use condition, `x = 1` highest allowable stress), with parameter vector
`theta = (alpha_0, alpha_1, lambda, q)`:

- `Y_x(t) ~ IG(a_x(t), b_x(t))`,  `a_x(t) = mu_x * Lambda(t)`,  `b_x(t) = lambda * Lambda(t)^2`,
- `mu_x = exp(alpha_0 + alpha_1 x)`  (Arrhenius-type link on the standardized scale, as in
  Ma et al. 2021, Eq. (7)),
- `Lambda(t) = t^q`, `q > 0`.

Increments over disjoint intervals are independent with

`Y_x(t2) - Y_x(t1) ~ IG( mu_x * (t2^q - t1^q),  lambda * (t2^q - t1^q)^2 ).`

A unit fails when its degradation crosses the threshold `omega > 0`. Because IG paths are a.s.
strictly increasing, the first-passage time `T_x = inf{t : Y_x(t) >= omega}` has CDF

```
F_x(t) = P(T_x <= t) = P(Y_x(t) >= omega) = Phi(u1(t)) - exp(2 b/a + log Phi(-u2(t)))
u1(t) = sqrt(b_x(t)/omega) * (1 - omega/a_x(t)),   u2(t) = sqrt(b_x(t)/omega) * (1 + omega/a_x(t)),
```

(with the second term evaluated in log space for stability; this is Wang et al. 2016, Eq. (3)).
The density `f_x(t) = dF_x/dt` is obtained by numerical differentiation. The `p`-quantile of the
lifetime at use condition is `xi_p = F_0^{-1}(p)`; for inference we also use the normal
approximation of the first passage (Ye et al. 2014; Ma et al. 2021, Eq. (5)):

```
xi_p(x) ≈ [ (mu_x / (4 lambda)) * ( z_p + sqrt(z_p^2 + 4 lambda omega / mu_x^2) )^2 ]^(1/q).
```

**Cumulative exposure (path-level CE).** We adopt the *degradation-level* CE model of Wang et al.
(2016, Assumption A4): after a stress change at time `tau` from `x_0` to `x_1`, the future path
depends only on the accumulated degradation and the new stress. A unit that has level `b` at the
switch has *virtual age* under the new stress

```
w(b) = ( b / mu_{x1} )^(1/q),
```

and its continuation at real time `tau + u` is distributed as `Y'_{x1}(w + u) - Y'_{x1}(w) + b`
for a fresh IG process `Y'_{x1}` at stress `x_1`. Consequently, writing
`dLam(u | b) = (w(b)+u)^q - w(b)^q`:

- the continuation crosses `omega` within offset `u` iff the continuation increment reaches
  `omega - b`, so

  `G(u | b) = P(cross by u | b) = S_IG( omega - b ;  mu_{x1} dLam(u|b),  lambda dLam(u|b)^2 )`,

  with density `g(u|b) = dG/du`;
- continuation increments between offsets `u1 < u2`:
  `IG( mu_{x1} * ((w+u2)^q - (w+u1)^q),  lambda * ((w+u2)^q - (w+u1)^q)^2 )`.

*Remark (two CE notions).* The SLT paper links the two stages through the **lifetime**-distribution
shift `F_{x0}(tau1) = F_{x1}(v1)` (Sedyakin–Nelson). For process-based models the natural — and
here the only directly usable — notion is the **path-level** CE above (the lifetime shift would
require matching both IG parameters, which generally fails). The induced continuation failure law
`G(u|b)` (crossing of the *residual* level `omega - b` by the continuation increment) plays the
role of the shifted density `f_1(z + v1 - tau1)` in SLT's Theorem 3.1, with the latent level `b`
replacing the deterministic shift. This distinction should be discussed in the paper.

---

## 2. The test scheme (SLT-D)

`n` identical units are placed on a life test on stage `s0` (stress `x0`). Let `T_i` be iid
`F_{x0}`.

1. Failures on stage `s0` are recorded. Let `D1 = #{i : T_i <= tau1}` with realized value `d1`.
2. At the prefixed stage-change time `tau1`, `R1* = rho(D1)` of the `n - d1` survivors are
   randomly withdrawn (Type-P: `rho(d) = floor(pi1 * (n-d))`; Type-M: `min(n-d, R0_1)`), and
   moved to a **degradation rig on stage `s1` (stress `x1 > x0`)**. Their degradation is read at
   the switch (baseline) and at `L` prefixed epochs `tau1 = u_0 < u_1 < ... < u_L <= T_max`;
   readings stop when a reading reaches `omega` (failure detected at inspection).
3. The remaining `D2 = n - d1 - R1*` units stay on stage `s0` until failure.
4. All observation stops at the global test horizon `T_max` (survivors are right-censored).

**Comparison schemes** (identical `n`, `tau1`, `T_max` and the same withdrawal rule; only the
use made of the withdrawn units differs):

- **(a) PC-FCT baseline** — withdrawn units are discarded: contribution `S_{x0}(tau1)` each;
- **(b) SLT** — withdrawn units are re-tested on `s1` to failure (censored at `T_max`);
- **(c) SLT-D (this paper)** — withdrawn units are degradation-monitored on `s1` (baseline +
  increments).

---

## 3. Likelihood

**Lemma (ignorable withdrawal / block independence).** Because the withdrawal at `tau1` is by
randomization among survivors — independent of the degradation levels — the missingness mechanism
is ignorable, and, given `D1 = d1`, the blocks (pre-tau1 failures), (post-tau1 failures of
stayers), and (per withdrawn unit: baseline + continuation) are conditionally independent with
their ordinary (unconditional) laws. This is the analogue of Lemma 3.3 of Laumen & Cramer (2019)
and makes the likelihood a plain product of densities — no combinatorial censoring factors
depending on `theta`.

Dropping constants in `theta`, the log-likelihoods of the three schemes are:

**(a) PC-FCT baseline.**
```
ell_a = sum_{i in failures on s0} log f_{x0}(t_i)     (d1 failures <= tau1 and d2 failures in (tau1, T_max])
      + sum_{stayers alive at T_max} log(1 - F_{x0}(T_max))
      + r1* * log(1 - F_{x0}(tau1)).
```

**(b) SLT (re-test to failure).** A withdrawn unit with latent level `B ~ IG(a_{x0}(tau1), b_{x0}(tau1))`
(truncated at `omega`, automatic) that fails at offset `u <= u_max = T_max - tau1` contributes the
1-d integral

```
m_f(u)  = Int_0^omega  f_IG(b) * g(u | b)  db,
```

and if it survives the re-test to `T_max`,
```
m_s     = Int_0^omega  f_IG(b) * (1 - G(u_max | b))  db.
ell_b   = [same first two lines as ell_a] + sum_k [ log m_f(u_k) or log m_s ].
```
(The integrals are computed by Gauss–Legendre quadrature on `(0, omega)`.)

**(c) SLT-D (degradation follow-up).** Withdrawn unit `k` with baseline reading `b_k` and
increment `Delta_kj` between reading epochs `(u_{j-1}, u_j]` (readings stopping at the first one
reaching `omega` — the crossing increment is still an ordinary IG observation, so no special
censoring term is needed under inspection-based failure detection):
```
ell_c   = [same first two lines as ell_a]
        + sum_k [ log f_IG( b_k ; a_{x0}(tau1), b_{x0}(tau1) ) ]
        + sum_k sum_j log f_IG( Delta_kj ; mu_{x1} * dLambda_kj, lambda * dLambda_kj^2 ),
          dLambda_kj = (w_k + u_j)^q - (w_k + u_{j-1})^q,  w_k = w(b_k).
```
Fully closed-form; no latent variables. (If no baseline reading is taken, `b_k` is latent and the
unit's contribution becomes `Int_0^omega f_IG(b) * prod_j f_IG(Delta_kj | b) db`; an EM algorithm
with `b` in the complete data is natural. Deferred to the extension section.)

**Stayer block (all schemes, identical):** failures contribute `log f_{x0}(t)`, units alive at
`T_max` contribute `log(1 - F_{x0}(T_max))`.

**Identifiability.** Stage `s0` failure data identify the first-passage law at `x0`
(functional combinations of `theta`), the stage-`s1` readings identify the IG law at `x1`
including `q` (within-path nonlinearity), and the two stress levels separate `alpha_0, alpha_1`.
`xi_p` at `x = 0` is estimable by extrapolation of the link.

---

## 4. Estimation and asymptotic inference

- MLE of `theta = (alpha_0, alpha_1, lambda, q)` by multi-start numerical optimization over the
  unconstrained parameterization `(alpha_0, alpha_1, log lambda, log q)`.
- Asymptotic covariance from the observed information (numerical Hessian); asymptotic variance of
  `hat{xi_p}` by the delta method applied to the closed-form quantile (gradient by central
  differences).
- Reparametrization invariance lets us report Wald CIs for `log xi_p`.

---

## 5. Simulation study design (pilot)

Anchor parameters from the electrical-connector stress-relaxation data (Ma et al. 2021):
`alpha_0 = -1.88, alpha_1 = 1.73, lambda = 0.653, q = 0.449, omega = 30`, Arrhenius
standardization `x(T) = (1/313.15 - 1/T) / (1/313.15 - 1/373.15)` gives `x(65C) = 0.46`,
`x(85C) = 0.78`, `x(100C) = 1.0`.

| Scenario | x0 | x1 | omega | n | tau1 | T_max | readings | purpose |
|---|---|---|---|---|---|---|---|---|
| S1 (connector-like) | 0.78 | 1.00 | 30 | 20 | 2000 | 5000 | every 150h (L=20) | sparse failures; degradation pays |
| S2 (moderate product) | 0.46 | 0.78 | 10 | 20 | 2000 | 5000 | every 50h (L=60) | rich failures everywhere |
| S3 (severe) | 0.46 | 0.78 | 30 | 30 | 2000 | 5000 | every 150h (L=20) | life test nearly blank |

Common random numbers: per replicate, the same latent world (unit lifetimes `T_i`, baselines
`B_k`, withdrawal draw) feeds all three schemes; continuation draws for (b)/(c) are sampled
independently (both exact). Schemes are compared at identical `(n, tau1, T_max)`.

**Metrics:** mean `d1`, `r1*`; relative bias and RMSE of `hat theta`; RMSE and median absolute
error of `log hat xi_{0.1}` (i.e., relative error of the use-condition 0.1-quantile);
empirical coverage of the 95% Wald CI for `log xi_{0.1}`; convergence rate. Additionally, a
sweep over `pi1 in {0.25, 0.5, 0.75}` in S2.

---

## 6. Validation strategy

1. `F_x` (first passage) validated against Monte Carlo hitting times of simulated IG paths.
2. Score identity: `E_theta0[grad ell] = 0` for every scheme (checked by Monte Carlo over worlds).
3. MLE recovery on large `n` complete data.
4. Scheme-(b) quadrature integrals `m_f, m_s` validated against Monte Carlo integration.
5. External check: fitting the plain IG model (constant stress, no stages) to the real
   stress-relaxation data reproduces the published estimates of Ma et al. (2021)
   `(alpha_0, alpha_1, lambda, q) = (-1.88, 1.73, 0.653, 0.449)`.
