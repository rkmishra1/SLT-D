"""Generate LaTeX table fragments + final figures for the manuscript (1000-rep data)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import src.analysis as A

TRUE = (-1.8944, 1.7377, 0.6285, 0.4491)
SC = {"(a)": "a", "(b)": "b", "(c)": "c"}
SCEN = {"S1-connector": "S1 (connector-like)", "S2-moderate": "S2 (moderate)", "S3-severe": "S3 (severe)"}


def main():
    import os

    src_csv = "results/per_rep_1000.csv"  # canonical run (crc32-seeded, corrected inference)
    if not os.path.exists(src_csv):
        src_csv = "results/per_rep_1000_fixed.csv"
    df = pd.read_csv(src_csv)
    A.TRUE = TRUE
    summary = A.summarize(df)
    summary.to_csv("results/summary_1000.csv", index=False)
    A.figures(summary, df)  # regenerate figs at 1000 reps

    out = []

    # ---- Table: hierarchy (pi1 = 0.5)
    out.append("% auto-generated: tab:hierarchy")
    out.append(r"\begin{tabular}{llrrrrrr}")
    out.append(r"\toprule")
    out.append(r"Scenario & Scheme & $\bar d_1$ & $\bar r_1^*$ & Bias & RMSE & MdAE & p90AE \\")
    for scen in ["S1-connector", "S2-moderate", "S3-severe"]:
        for lab, sch in SC.items():
            r = summary[(summary.scenario == scen) & (summary.pi1 == 0.5) & (summary.scheme == sch)].iloc[0]
            out.append(
                f"{SCEN[scen]} & {lab} & {r.d1_bar:.2f} & {r.r1_bar:.2f} & "
                f"{r.bias_log_xi:.3f} & {r.rmse_log_xi:.3f} & {r.mdae_log_xi:.3f} & {r.p90_ae_log_xi:.3f} \\\\"
            )
        if scen != "S3-severe":
            out.append(r"\addlinespace")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append("")

    # ---- Table: pi sweep in S2
    out.append("% auto-generated: tab:pisweep")
    out.append(r"\begin{tabular}{crrrrrrr}")
    out.append(r"\toprule")
    out.append(r"$\pi_1$ & Scheme & $\bar d_1$ & $\bar r_1^*$ & Bias & RMSE & MdAE & Cov. \\")
    for pi in [0.25, 0.5, 0.75]:
        for lab, sch in SC.items():
            r = summary[(summary.scenario == "S2-moderate") & (np.isclose(summary.pi1, pi))
                        & (summary.scheme == sch)].iloc[0]
            cov = "--" if np.isnan(r.cover95) else f"{100 * r.cover95:.1f}"
            out.append(
                f"{pi:.2f} & {lab} & {r.d1_bar:.2f} & {r.r1_bar:.2f} & "
                f"{r.bias_log_xi:.3f} & {r.rmse_log_xi:.3f} & {r.mdae_log_xi:.3f} & {cov} \\\\"
            )
        out.append(r"\addlinespace")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append("")

    # ---- Table: predicted vs empirical Asvar (scheme c)
    import src.ig_process as ig
    from src.design import avar_log_xi

    out.append("% auto-generated: tab:avarcheck")
    out.append(r"\begin{tabular}{lccc}")
    out.append(r"\toprule")
    out.append(r"Design & $n$ & predicted $\Asvar(\log\hat\xi_{0.1})$ & empirical var \\")
    designs = {
        "S1": (dict(x0=0.78, x1=1.0, omega=30.0, tau1=2000.0, tmax=5000.0, pi1=0.5,
                    read_offsets=np.arange(150.0, 3001.0, 150.0)), 20, "S1-connector"),
        "S2": (dict(x0=0.46, x1=0.78, omega=10.0, tau1=2000.0, tmax=5000.0, pi1=0.5,
                    read_offsets=np.arange(50.0, 3001.0, 50.0)), 20, "S2-moderate"),
        "S3": (dict(x0=0.46, x1=0.78, omega=30.0, tau1=2000.0, tmax=5000.0, pi1=0.5,
                    read_offsets=np.arange(150.0, 3001.0, 150.0)), 30, "S3-severe"),
    }
    for name, (d, n, scen) in designs.items():
        pred = avar_log_xi(np.array(TRUE), d, n, p=0.1, x_use=0.0)
        g = df[(df.scenario == scen) & (np.isclose(df.pi1, 0.5)) & (df.scheme == "c") & (df.ok == 1)]
        emp = float(np.var(g["log_xi"], ddof=1))
        out.append(f"{name} & {n} & {pred:.3f} & {emp:.3f} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append("")

    # ---- Table: design optimization
    with open("results/design_results.json") as f:
        dr = json.load(f)
    out.append("% auto-generated: tab:design")
    out.append(r"\begin{tabular}{lrrrrrrr}")
    out.append(r"\toprule")
    out.append(r"Design & $x_0$/ $x_a$ & $\tau_1$ & $\pi_1$ & $x_1$/ $x_b$ & step & $n$ & $\Asvar$ \\")
    b = dr["base"]
    out.append(f"constrained SLT-D & {b[0]:.2f} & {b[1]:.0f} & {b[2]:.2f} & {b[3]:.2f} & {b[4]:.0f} & {b[5]:.0f} & {b[6]:.4f} \\\\")
    a_ = dr["adt"]
    out.append(f"parallel two-level ADT & {a_[0]:.2f} & -- & -- & {a_[1]:.2f} & {a_[2]:.0f} & {a_[3] + a_[4]:.0f} & {a_[5]:.4f} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append("")

    # ---- Table: L9 plan robustness
    out.append("% auto-generated: tab:l9plan")
    out.append(r"\begin{tabular}{rrrrrrrr}")
    out.append(r"\toprule")
    out.append(r"$\epsilon_{\alpha_0}$ & $\epsilon_{\alpha_1}$ & $\epsilon_{\lambda}$ & $\epsilon_q$ & "
               r"$\tau_1$ & $\pi_1$ & step & $\Asvar$ at truth \\")
    for row in dr["l9"]:
        e = row["eps"]
        p = row["plan"]
        out.append(f"{100 * e[0]:+.0f}\\% & {100 * e[1]:+.0f}\\% & {100 * e[2]:+.0f}\\% & {100 * e[3]:+.0f}\\% & "
                   f"{p[1]:.0f} & {p[2]:.2f} & {p[4]:.0f} & {row['asvar_at_true']:.4f} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append("")

    # ---- Table: estimation-robustness sensitivity (if available)
    try:
        ds = pd.read_csv("results/sensitivity_estimation.csv")
        ds = ds[ds.ok == 1]
        # expected number of withdrawn units per combo under the perturbed truth
        # (S2 design: n=20, tau1=2000, omega=10, x0=0.46, pi1=0.5)
        import src.ig_process as ig
        from scipy.stats import binom

        BASE = np.array(TRUE)
        L9 = [(1, 1, 1, 1), (1, 2, 2, 2), (1, 3, 3, 3),
              (2, 1, 2, 3), (2, 2, 3, 1), (2, 3, 1, 2),
              (3, 1, 3, 2), (3, 2, 1, 3), (3, 3, 2, 1)]

        def exp_r1(combo_idx):
            eps = np.array([{1: -0.10, 2: 0.0, 3: 0.10}[c] for c in L9[combo_idx]])
            th = BASE * (1.0 + eps)
            F = float(ig.passage_cdf(2000.0, 0.46, th, 10.0))
            k = np.arange(0, 21)
            pr = binom.pmf(k, 20, F)
            return float(np.sum(pr * np.floor(0.5 * (20 - k))))

        rows = []
        for combo, g in ds.groupby("combo"):
            rr = {}
            for sch in ["b", "c"]:
                gg = g[g.scheme == sch]
                err = gg["log_xi"] - gg["log_xi_true"]
                rr[sch] = (err.mean(), np.sqrt(np.mean(err**2)), err.abs().median())
            rows.append((combo, rr))
        out.append("% auto-generated: tab:sensest")
        out.append(r"\begin{tabular}{crrrrrrr}")
        out.append(r"\toprule")
        out.append(r"Combo & $\E[r_1^*]$ & \multicolumn{3}{c}{scheme (b)} & \multicolumn{3}{c}{scheme (c)} \\")
        out.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}")
        out.append(r" & & Bias & RMSE & MdAE & Bias & RMSE & MdAE \\")
        for combo, rr in rows:
            b_ = rr["b"]
            c_ = rr["c"]
            er1 = exp_r1(int(combo))
            out.append(f"{combo + 1} & {er1:.1f} & {b_[0]:.3f} & {b_[1]:.3f} & {b_[2]:.3f} & "
                       f"{c_[0]:.3f} & {c_[1]:.3f} & {c_[2]:.3f} \\\\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
    except FileNotFoundError:
        out.append("% sensitivity_estimation.csv not ready")

    # ---- Table: k-stage comparison
    try:
        dk = pd.read_csv("results/kstage_study.csv")
        out.append("% auto-generated: tab:kstage")
        out.append(r"\begin{tabular}{cp{4.6cm}rrrrrr}")
        out.append(r"\toprule")
        out.append(r"$K$ & stages $(\tau,\pi,x)$ & $\E[R_1^{*}]$ & $\E[\#\text{readings}]$ "
                   r"& Bias & RMSE & MdAE & RMSE$\cdot\sqrt{r}$ \\")
        stages_txt = {
            "K1": r"$(2000,\,0.50,\,0.78)$",
            "K2": r"$(1500,0.30,0.78)$; $(3000,0.30,1)$",
            "K3": r"$(1000,0.15,0.78)$; $(2000,0.20,0.90)$; $(3500,0.25,1)$",
        }
        for name in ["K1", "K2", "K3"]:
            g = dk[dk.design == name]
            err = g["log_xi"] - g["log_xi_true"]
            k = int(name[1])
            rmse = float(np.sqrt((err ** 2).mean()))
            nr = g["n_readings"].mean()
            out.append(f"${k}$ & {stages_txt[name]} & {g['n_withdrawn'].mean():.2f} & "
                       f"{nr:.1f} & {err.mean():+.3f} & {rmse:.3f} & {err.abs().median():.3f} & "
                       f"{rmse * np.sqrt(nr):.2f} \\\\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append("")
    except FileNotFoundError:
        out.append("% kstage_study.csv not ready")

    # ---- Table: Bayesian vs plug-in plans
    try:
        with open("results/bayes_design.json") as f:
            jb = json.load(f)
        out.append("% auto-generated: tab:bayes")
        out.append(r"\begin{tabular}{llrrrrrrrrr}")
        out.append(r"\toprule")
        out.append(r"Prior & Plan & $x_0$ & $\tau_1$ & $\pi_1$ & $x_1$ & step & $n$ "
                   r"& $J$ & $J_{90}$ & $\Asvar$ at mean \\")
        for key, kap in [("kappa1", r"$\kappa{=}1$"), ("kappa4", r"$\kappa{=}4$")]:
            if key not in jb.get("scenarios", {}):
                continue
            sc = jb["scenarios"][key]
            for plan_key, lab in [("bayes_plan", "Bayesian"), ("plugin_plan", "plug-in")]:
                p = sc[plan_key]
                ev = sc["bayes_eval"] if plan_key == "bayes_plan" else sc["plugin_eval"]
                out.append(f"{kap} & {lab} & {p[0]:.2f} & {p[1]:.0f} & {p[2]:.1f} & {p[3]:.2f} & "
                           f"{p[4]:.0f} & {p[5]:.0f} & {ev['J_mean']:.4f} & {ev['J_p90']:.4f} & "
                           f"{ev['avar_true']:.4f} \\\\")
            out.append(r"\addlinespace")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append("")
    except FileNotFoundError:
        out.append("% bayes_design.json not ready")

    # ---- Table: laser case study (design comparison under heterogeneous truth)
    try:
        dl = pd.read_csv("results/laser_study.csv")
        out.append("% auto-generated: tab:laser")
        out.append(r"\begin{tabular}{lccrrrr}")
        out.append(r"\toprule")
        out.append(r"Design & monitored & degenerate & Bias & RMSE & MdAE & $p_{90}$AE \\")
        out.append(r" & units & fits & & & & \\")
        lab = {"full": "full monitoring (historical)",
               "sltd": r"\SLTD{} ($\tau_1{=}2000$, $\pi_1{=}0.5$)",
               "discard": "life test only"}
        mon = {"full": "15 from $t{=}0$", "sltd": r"$\approx7$ from $\tau_1$", "discard": "0"}
        for name in ["full", "sltd", "discard"]:
            g = dl[dl.design == name]
            e_all = g["log_xi"] - g["log_xi_true"]
            fin = e_all[np.isfinite(e_all)]
            n_degen = int((~np.isfinite(e_all)).sum())
            out.append(f"{lab[name]} & {mon[name]} & {n_degen}/500 & {fin.mean():+.3f} & "
                       f"{float(np.sqrt((fin**2).mean())):.3f} & {e_all.abs().median():.3f} & "
                       f"{e_all.abs().quantile(0.9):.3f} \\\\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append("")

    except FileNotFoundError:
        out.append("% laser_study.csv not ready")

    with open("results/tables.tex", "w") as f:
        f.write("\n".join(out))
    print("wrote results/tables.tex")


if __name__ == "__main__":
    main()
