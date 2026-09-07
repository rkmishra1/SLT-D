"""Aggregate per-rep results into tables + figures for RESULTS.md."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCHEME_LABEL = {"a": "(a) discard (PC-FCT)", "b": "(b) SLT re-test to failure", "c": "(c) SLT-D degradation"}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scen, pi1, scheme), g in df[df.ok == 1].groupby(["scenario", "pi1", "scheme"]):
        err = g["log_xi"] - g["log_xi_true"]
        se_ok = g[g["se"].notna()]
        cov = ((se_ok["lo"] < se_ok["log_xi_true"]) & (se_ok["log_xi_true"] < se_ok["hi"])).mean() if len(se_ok) else np.nan
        rows.append(dict(
            scenario=scen, pi1=pi1, scheme=scheme, scheme_label=SCHEME_LABEL[scheme],
            reps=len(g), d1_bar=g["d1"].mean(), r1_bar=g["r1"].mean(),
            failfrac_b=(g["n_with_fail_b"] / g["r1"].replace(0, np.nan)).mean() if scheme == "b" else np.nan,
            crossfrac_c=(g["n_cross_c"] / g["r1"].replace(0, np.nan)).mean() if scheme == "c" else np.nan,
            se_finite_rate=g["se"].notna().mean(),
            bias_a0=g["a0"].mean() - g["a0"].iloc[0] * 0 - TRUE[0],
            rmse_a1=np.sqrt(np.mean((g["a1"] - TRUE[1]) ** 2)),
            rmse_q=np.sqrt(np.mean((g["q"] - TRUE[3]) ** 2)),
            bias_log_xi=err.mean(), rmse_log_xi=np.sqrt(np.mean(err**2)),
            mdae_log_xi=err.abs().median(),
            p90_ae_log_xi=err.abs().quantile(0.9),
            cover95=cov,
        ))
    return pd.DataFrame(rows)


TRUE = (-1.8944, 1.7377, 0.6285, 0.4491)


def figures(summary: pd.DataFrame, df: pd.DataFrame, outdir="results"):
    os.makedirs(outdir, exist_ok=True)

    # Fig 1: RMSE(log xi) by scheme x scenario (pi1 = 0.5)
    sub = summary[(summary.pi1 == 0.5) & (summary.scenario != "S2-moderate") |
                  ((summary.scenario == "S2-moderate") & (summary.pi1 == 0.5))]
    scen_order = [s for s in ["S1-connector", "S2-moderate", "S3-severe"] if s in set(sub.scenario)]
    sub = sub[sub.pi1 == 0.5]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.25
    xs = np.arange(len(scen_order))
    for j, scheme in enumerate(["a", "b", "c"]):
        vals = []
        for sc in scen_order:
            v = sub[(sub.scenario == sc) & (sub.scheme == scheme)]["rmse_log_xi"]
            vals.append(float(v.iloc[0]) if len(v) else np.nan)
        vals = np.array(vals)
        vals_capped = np.minimum(vals, 12)  # cap for display
        ax.bar(xs + (j - 1) * width, vals_capped, width, label=SCHEME_LABEL[scheme])
        for x, v, vc in zip(xs + (j - 1) * width, vals, vals_capped):
            if v > 12:
                ax.text(x, 12.2, f"{v:.0f}", ha="center", fontsize=8, color="firebrick")
    ax.set_xticks(xs)
    ax.set_xticklabels(scen_order)
    ax.set_ylabel("RMSE of log xi_hat_0.1 (capped at 12)")
    ax.set_title("Precision of the use-condition 0.1-quantile by scheme (200 reps)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig1_rmse_by_scheme.png", dpi=150)

    # Fig 2: pi1 sweep in S2
    s2 = summary[(summary.scenario == "S2-moderate")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, metric, lab in [(axes[0], "rmse_log_xi", "RMSE of log xi_hat"),
                            (axes[1], "mdae_log_xi", "median |error| of log xi_hat")]:
        for scheme in ["a", "b", "c"]:
            g = s2[s2.scheme == scheme].sort_values("pi1")
            ax.plot(g["pi1"], g[metric], "o-", label=SCHEME_LABEL[scheme])
        ax.set_xlabel("withdrawal proportion pi1")
        ax.set_ylabel(lab)
        ax.legend(fontsize=8)
    fig.suptitle("S2-moderate: effect of withdrawal proportion")
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig2_pi_sweep.png", dpi=150)


def main():
    df = pd.read_csv("results/per_rep.csv")
    summary = summarize(df)
    summary.to_csv("results/summary.csv", index=False)
    with pd.option_context("display.width", 250):
        cols = ["scenario", "pi1", "scheme", "reps", "d1_bar", "r1_bar", "bias_log_xi",
                "rmse_log_xi", "mdae_log_xi", "p90_ae_log_xi", "se_finite_rate", "cover95"]
        print(summary[cols].round(3).to_string(index=False))
    figures(summary, df)


if __name__ == "__main__":
    main()
