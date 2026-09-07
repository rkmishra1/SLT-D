"""Generate the five illustrative figures for the manuscript.

1. fig_scheme.png          -- simulated SLT-D worlds: K=1 and K=3 (timelines + degradation paths)
2. fig_design_landscape.png -- budget-constrained Avar over (tau1, x0) with the survival cliff
3. fig_error_dist.png      -- |error| distributions of log xi_hat by scheme (tails)
4. fig_bayes_prior.png     -- prior draws vs collapse boundary; Avar-over-prior boxplots
5. fig_realdata.png        -- connector stress-relaxation data + fitted mean paths
"""
from __future__ import annotations

import multiprocessing as mp
import os
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import src.ig_process as ig
import src.kstage as ks
from src.bayes_design import (C_B, C_MEA, N_MAX, OMEGA, P, TMAX, THETA_BASE,
                              bayes_criterion, build_design, feasible_n, pilot_prior,
                              prior_draws)
from src.design import avar_log_xi
from src.fit_real_data import DATA, TIMES, x_of_temp
from src.simulate import simulate_world

THETA = np.array([-1.8944, 1.7377, 0.6285, 0.4491])
OUT = "results"
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ 1. scheme illustration

def fig_scheme():
    d1 = {"x0": 0.46, "x1": 0.78, "omega": 10.0, "n": 20, "tau1": 2000.0, "tmax": 5000.0,
          "pi1": 0.5, "read_offsets": np.arange(50.0, 3001.0, 50.0)}
    d3 = {"x0": 0.46, "omega": 10.0, "n": 20, "tmax": 5000.0,
          "stages": [(1000.0, 0.15, 0.78), (2000.0, 0.20, 0.90), (3500.0, 0.25, 1.00)],
          "read_offsets": np.arange(50.0, 5000.0 - 3500.0 + 1.0, 50.0)}
    rng = np.random.default_rng(zlib.crc32(b"scheme|fig|7"))
    w1 = simulate_world(rng, THETA, d1)
    w3 = ks.simulate_world_kstage(rng, THETA, d3)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))

    # --- panel (a): K=1 timeline
    ax = axes[0, 0]
    rng2 = np.random.default_rng(1)
    idx = 0
    stay_i = 0
    labels = []
    for i in range(d1["n"]):
        pass
    # reconstruct unit-level records: failures (times), stayers, withdrawn
    y = 0
    for t in w1["fail_pre"]:
        ax.plot([0, t], [y, y], color="0.3", lw=1.2)
        ax.plot(t, y, "x", color="crimson", ms=6)
        labels.append("fail")
        y += 1
    for t in w1["stayers"]:
        if t is None:
            ax.plot([0, d1["tmax"]], [y, y], color="0.3", lw=1.2)
            ax.plot(d1["tmax"], y, "o", mfc="white", mec="0.3", ms=4)
        else:
            ax.plot([0, t], [y, y], color="0.3", lw=1.2)
            ax.plot(t, y, "x", color="crimson", ms=6)
        y += 1
    for k in w1["withdrawn"]:
        ax.plot([0, d1["tau1"]], [y, y], color="0.3", lw=1.2)
        ax.plot(d1["tau1"], y, ">", color="tab:blue", ms=6)
        y += 1
    ax.axvline(d1["tau1"], color="tab:blue", ls="--", lw=1)
    ax.set_ylim(-1, d1["n"])
    ax.set_ylabel("unit")
    ax.set_title(r"(a) $K{=}1$: stage-0 life test ($x_0=0.46$, $\omega=10$)", fontsize=9)
    ax.text(0.985, 0.02, r"$\times$: failure;  $\triangleright$: withdrawn (dashed $=\tau_j$);  $\circ$: censored",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="0.25",
            bbox=dict(fc="white", ec="none", alpha=0.85))

    # --- panel (b): K=1 degradation paths of withdrawn units
    ax = axes[0, 1]
    for k in w1["withdrawn"]:
        b = k["B"]
        offs = np.asarray(k["read_offsets"], float)
        lev = np.asarray(k["read_levels"], float)
        tt = d1["tau1"] + offs
        crossed = lev >= d1["omega"]
        ax.plot(tt, lev, color="tab:blue", lw=1.0, alpha=0.85)
        ax.plot(d1["tau1"], b, "o", color="tab:blue", ms=3.5)
        if crossed.any():
            j = int(np.argmax(crossed))
            ax.plot(tt[j], lev[j], "x", color="crimson", ms=5)
    ax.axhline(d1["omega"], color="crimson", ls=":", lw=1.2)
    ax.text(60, d1["omega"] + 0.3, r"threshold $\omega$", color="crimson", fontsize=8)
    ax.axvline(d1["tau1"], color="tab:blue", ls="--", lw=1)
    ax.set_ylabel("degradation level")
    ax.set_title(r"(b) $K{=}1$: rig readings at $x_1=0.78$ (per withdrawn unit)", fontsize=9)

    # --- panel (c): K=3 timeline
    ax = axes[1, 0]
    y = 0
    for t in w3["failures"]:
        ax.plot([0, t], [y, y], color="0.3", lw=1.2)
        ax.plot(t, y, "x", color="crimson", ms=6)
        y += 1
    n_w3 = sum(len(c["units"]) for c in w3["cohorts"])
    for _ in range(w3["n_cens"]):
        ax.plot([0, d3["tmax"]], [y, y], color="0.3", lw=1.2)
        ax.plot(d3["tmax"], y, "o", mfc="white", mec="0.3", ms=4)
        y += 1
    for ci, c in enumerate(w3["cohorts"]):
        for _ in c["units"]:
            ax.plot([0, c["tau"]], [y, y], color="0.3", lw=1.2)
            ax.plot(c["tau"], y, ">", color=f"C{ci}", ms=6)
            y += 1
    for c in w3["cohorts"]:
        ax.axvline(c["tau"], color=f"C{[0,1,2][[1000.0,2000.0,3500.0].index(c['tau'])]}",
                   ls="--", lw=1)
    ax.set_ylim(-1, d3["n"])
    ax.set_xlabel("time on test (h)")
    ax.set_ylabel("unit")
    ax.set_title(r"(c) $K{=}3$: three withdrawal times $\tau_1<\tau_2<\tau_3$", fontsize=9)

    # --- panel (d): K=3 degradation paths colored by cohort stress
    ax = axes[1, 1]
    for ci, c in enumerate(w3["cohorts"]):
        for k in c["units"]:
            b = k["B"]
            offs = np.asarray(k["read_offsets"], float)
            lev = np.asarray(k["read_levels"], float)
            tt = c["tau"] + offs
            crossed = lev >= d3["omega"]
            ax.plot(tt, lev, color=f"C{ci}", lw=1.0, alpha=0.85)
            ax.plot(c["tau"], b, "o", color=f"C{ci}", ms=3.5)
            if crossed.any():
                j = int(np.argmax(crossed))
                ax.plot(tt[j], lev[j], "x", color="crimson", ms=5)
    ax.axhline(d3["omega"], color="crimson", ls=":", lw=1.2)
    ax.text(1330, d3["omega"] + 0.35, r"threshold $\omega$", color="crimson", fontsize=8)
    for c in w3["cohorts"]:
        ax.axvline(c["tau"], color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("calendar time (h)")
    ax.set_ylabel("degradation level")
    ax.set_title(r"(d) $K{=}3$: rig readings at $x_1=0.78$, $x_2=0.90$, $x_3=1.0$", fontsize=9)

    fig.tight_layout(h_pad=2.4, w_pad=2.0)
    fig.savefig(f"{OUT}/fig_scheme.png", dpi=170)
    plt.close(fig)
    print("fig_scheme done")


# ------------------------------------------------------------------ 2. design landscape

def _avar_cell(args):
    x0, tau1 = args
    d = {"x0": x0, "x1": 0.85, "omega": OMEGA, "n": 1, "tau1": tau1, "tmax": TMAX, "pi1": 0.9,
         "read_offsets": np.arange(150.0, TMAX - tau1 + 1e-9, 150.0)}
    n = feasible_n(d, THETA_BASE)
    if n < 4:
        return x0, tau1, np.nan, n
    v = avar_log_xi(THETA_BASE, d, n, p=P, x_use=0.0)
    return x0, tau1, v, n


def fig_design_landscape():
    x0s = np.arange(0.10, 1.0001, 0.05)
    taus = np.arange(500.0, 4000.1, 250.0)
    cells = [(x0, t) for x0 in x0s for t in taus]
    with mp.Pool(max(1, os.cpu_count() - 4)) as pool:
        res = pool.map(_avar_cell, cells)
    A = np.full((len(x0s), len(taus)), np.nan)
    N = np.zeros_like(A)
    for x0, t, v, n in res:
        A[np.where(x0s == x0)[0][0], np.where(taus == t)[0][0]] = v
        N[np.where(x0s == x0)[0][0], np.where(taus == t)[0][0]] = n
    A[A < 1e-9] = np.nan  # placeholder guard
    A[x0s > 0.85 - 0.051, :] = np.nan  # invalid: rig stress must exceed stage-0 stress

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    Lv = np.log10(np.where(np.isfinite(A) & (A > 0), A, np.nan))
    pc = ax.pcolormesh(taus, x0s, Lv, shading="nearest", cmap="viridis_r", vmin=-1.6, vmax=1.4)
    cs = ax.contour(taus, x0s, Lv, levels=[-1, 0, 1], colors="w", linewidths=0.7)
    ax.clabel(cs, fmt="10^%d", fontsize=7)
    # survival cliff contours: F_{x0}(tau1) = 0.5 and 0.95
    TT, XX = np.meshgrid(taus, x0s)
    for lev, lab in [(0.5, r"$F_{x_0}(\tau_1)=0.5$"), (0.95, r"$F_{x_0}(\tau_1)=0.95$")]:
        Z = np.array([[float(ig.passage_cdf(t, x0, THETA_BASE, OMEGA))
                       for t in taus] for x0 in x0s])
        ax.contour(taus, x0s, Z, levels=[lev], colors="red", linewidths=1.6)
        # label at first crossing
        idx = np.argwhere(np.diff((Z > lev).astype(int), axis=1))
        if len(idx):
            i, j = idx[0]
            ax.text(taus[j], x0s[i] + 0.012, lab, color="red", fontsize=8)
    i0, j0 = np.unravel_index(np.nanargmin(A), A.shape)
    ax.plot(taus[j0], x0s[i0], "r*", ms=16, mec="white")
    ax.annotate(f"optimum\n$\\mathrm{{Avar}}={A[i0, j0]:.3f}$, $n={int(N[i0, j0])}$",
                (taus[j0], x0s[i0]), textcoords="offset points", xytext=(12, 10),
                fontsize=8, color="white")
    cb = fig.colorbar(pc, ax=ax, label=r"$\log_{10}\mathrm{Avar}(\log\hat\xi_{0.1}(0))$")
    ax.set_xlabel(r"stage-change time $\tau_1$ (h)")
    ax.set_ylabel(r"stage-0 stress $x_0$")
    ax.set_title(r"Budget-constrained $\mathrm{Avar}$ over $(x_0,\tau_1)$ at $\pi_1=0.9$, $x_1=0.85$, "
                 r"150 h readings", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_design_landscape.png", dpi=170)
    plt.close(fig)
    print("fig_design_landscape done")


# ------------------------------------------------------------------ 3. error distributions

def fig_error_dist():
    import pandas as pd

    df = pd.read_csv(f"{OUT}/per_rep_1000.csv")
    df = df[(df.ok == 1) & (np.isclose(df.pi1, 0.5))]
    scen_order = ["S1-connector", "S2-moderate", "S3-severe"]
    sch_order = ["a", "b", "c"]
    data, pos, cols = [], [], []
    colors = {"a": "0.55", "b": "tab:orange", "c": "tab:blue"}
    xpos = 0
    ticks, ticklabs = [], []
    for sc in scen_order:
        start = xpos + 0.5
        for s in sch_order:
            g = df[(df.scenario == sc) & (df.scheme == s)]
            err = (g["log_xi"] - g["log_xi_true"]).abs().values
            err = err[err > 0]
            data.append(err)
            pos.append(xpos)
            cols.append(colors[s])
            xpos += 1
        ticks.append((start + xpos - 1) / 2 - 0.5)
        ticklabs.append({"S1-connector": "S1", "S2-moderate": "S2", "S3-severe": "S3"}[sc])
        xpos += 1.6
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bp = ax.boxplot(data, positions=pos, widths=0.75, showfliers=True,
                    flierprops=dict(marker=".", ms=2.5, alpha=0.4),
                    medianprops=dict(color="black", lw=1.2), patch_artist=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabs)
    ax.set_ylabel(r"$|\,\ln\hat\xi_{0.1}(0)-\ln\xi_{0.1}(0)\,|$  (log scale)")
    ax.axhline(0.5, color="0.4", ls=":", lw=1)
    ax.text(6.4, 0.78, r"$e^{0.5}\approx64\%$ error", fontsize=7.5, color="0.4", ha="center")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(fc=colors[s], alpha=0.75,
                             label={"a": "(a) discard", "b": "(b) SLT re-test", "c": "(c) SLT-D"}[s])
                       for s in sch_order], fontsize=8, loc="upper left")
    ax.set_title(r"Absolute errors of $\ln\hat\xi_{0.1}(0)$ by scheme and scenario "
                 r"($\pi_1=0.5$, 1000 reps)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_error_dist.png", dpi=170)
    plt.close(fig)
    print("fig_error_dist done")


# ------------------------------------------------------------------ 4. prior vs cliff

def _cliff_curve(x0, tau1, level, a1s):
    """alpha0 values where F_{x0}(tau1) = level, for each alpha1 (bisection)."""
    out = []
    for a1 in a1s:
        lo, hi = -6.0, 3.0

        def F(a0):
            th = np.array([a0, a1, THETA[2], THETA[3]])
            return float(ig.passage_cdf(tau1, x0, th, OMEGA)) - level

        if F(lo) * F(hi) > 0:
            out.append(np.nan)
            continue
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if F(lo) * F(mid) <= 0:
                hi = mid
            else:
                lo = mid
        out.append(0.5 * (lo + hi))
    return np.array(out)


def fig_bayes_prior():
    z_hat, cov, _ = pilot_prior(kappa=4.0)
    draws = prior_draws(z_hat, cov, m=400, seed=7)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1))
    ax = axes[0]
    ax.scatter(draws[:, 0], draws[:, 1], s=8, alpha=0.45, color="tab:blue", edgecolors="none")
    ax.plot(THETA_BASE[0], THETA_BASE[1], "r*", ms=13, label="pilot MLE / prior mean")
    a1s = np.linspace(0.2, 3.2, 120)
    for lev, ls in [(0.5, "--"), (0.95, "-")]:
        curve = _cliff_curve(0.10, 1750.0, lev, a1s)
        ax.plot(curve, a1s, color="crimson", ls=ls, lw=1.6,
                label=f"collapse boundary $F_{{x_0}}(\\tau_1)={lev}$")
    ax.set_xlabel(r"$\alpha_0$")
    ax.set_ylabel(r"$\alpha_1$")
    ax.set_title(r"(a) prior draws vs the collapse boundary of the optimal plan "
                 r"($x_0{=}0.10$, $\tau_1{=}1750$ h; $\kappa{=}4$ shown)", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    z1, c1, _ = pilot_prior(kappa=1.0)
    draws1 = prior_draws(z1, c1, m=400, seed=7)
    d_opt = build_design(0.10, 1750.0, 0.9, 0.85, 150.0)
    d_opt["n"] = feasible_n(d_opt, THETA_BASE)
    vals_p = bayes_criterion(d_opt, draws1)
    vals_b = bayes_criterion(d_opt, draws)
    bp = ax.boxplot([vals_p, vals_b], tick_labels=[r"$\kappa{=}1$ (full pilot)",
                                                   r"$\kappa{=}4$ (weak pilot)"],
                    widths=0.5, showfliers=True, flierprops=dict(marker=".", ms=3, alpha=0.5),
                    medianprops=dict(color="black", lw=1.2), patch_artist=True)
    for patch, c in zip(bp["boxes"], ["tab:orange", "tab:blue"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\mathrm{Avar}(\log\hat\xi_{0.1}(0))$ over prior draws (log scale)")
    ax.set_title(r"(b) criterion distribution over the prior for the optimal plan "
                 r"($400$ draws)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_bayes_prior.png", dpi=170)
    plt.close(fig)
    print("fig_bayes_prior done")


# ------------------------------------------------------------------ 5. real data

def fig_realdata():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    temps = [65.0, 85.0, 100.0]
    colors = {65.0: "tab:blue", 85.0: "tab:orange", 100.0: "tab:red"}
    for temp in temps:
        ts = np.asarray(TIMES[temp], float)
        for lev in DATA[temp]:
            ax.plot(ts, lev, color=colors[temp], lw=0.9, alpha=0.75)
        x = x_of_temp(temp)
        mu = np.exp(THETA[0] + THETA[1] * x)
        tt = np.linspace(1, 3000, 200)
        ax.plot(tt, mu * tt ** THETA[3], color=colors[temp], ls="--", lw=1.8,
                label=f"{temp:.0f} $^\\circ$C (fit: $\\mu t^{{0.449}}$)")
    ax.axhline(30, color="crimson", ls=":", lw=1.2)
    ax.text(2100, 30.6, r"failure threshold $\omega=30$", color="crimson", fontsize=8)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("stress relaxation (%)")
    ax.set_title("Electrical-connector stress-relaxation data (Yang 2007, Ex. 8.7) "
                 "and fitted IG mean paths", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_realdata.png", dpi=170)
    plt.close(fig)
    print("fig_realdata done")



def fig_laserdata():
    """Laser data: 15 paths + fitted homogeneous mean + threshold."""
    import src.ig_process as ig
    from src.fit_laser_data import fit_laser

    fit = fit_laser()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    shades = plt.cm.viridis(np.linspace(0.15, 0.85, len(fit["data"])))
    for (ts, ys), c in zip(fit["data"].values(), shades):
        crossed = ys >= 10.0
        ax.plot(ts, ys, color=c, lw=1.0, alpha=0.85)
        if crossed.any():
            j = int(np.argmax(crossed))
            ax.plot(ts[j], ys[j], "x", color="crimson", ms=6)
    tt = np.linspace(1, 4000, 300)
    ax.plot(tt, fit["mu"] * tt ** fit["q"], color="0.2", ls="--", lw=1.8,
            label=f"fitted mean $\\mu t^{{\\hat q}}$ ($\\hat q={fit['q']:.2f}$)")
    ax.axhline(10.0, color="crimson", ls=":", lw=1.2)
    ax.text(2100, 10.25, r"failure threshold $\omega=10$ (10% current increase)",
            color="crimson", fontsize=8)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("operating-current increase (%)")
    ax.set_title("GaAs laser degradation data (Meeker & Escobar 1998, p. 339) and "
                 "fitted IG mean path", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_laserdata.png", dpi=170)
    plt.close(fig)
    print("fig_laserdata done")


if __name__ == "__main__":
    fig_scheme()
    fig_error_dist()
    fig_realdata()
    fig_bayes_prior()
    fig_design_landscape()
    fig_laserdata()
