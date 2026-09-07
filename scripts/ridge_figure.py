"""Profile-likelihood ridge figure: scheme (a) vs scheme (c) on the same S3 world
(zero failures on stage 0 -> scheme (a) has only single-stress survival information).

Left: profile loglik over (alpha0, alpha1) for scheme (a) -- an elongated ridge along
mu_{x0} = e^{a0+a1 x0} = const, so use-condition extrapolation is unidentifiable.
Right: the same profile for scheme (c) -- concentrated around the truth.
Annotates the range of log xi_0.1(x=0) attained inside the ~15% profile-likelihood region.
"""
import numpy as np

import src.ig_process as ig
import src.likelihood as lik
from src.simulate import simulate_world
from scipy.optimize import minimize

THETA = np.array([-1.8944, 1.7377, 0.6285, 0.4491])
DESIGN = {
    "x0": 0.46, "x1": 0.78, "omega": 30.0, "n": 30, "tau1": 2000.0, "tmax": 5000.0, "pi1": 0.5,
    "read_offsets": np.arange(150.0, 3001.0, 150.0),
}


def profile_over_lam_q(scheme, a0, a1, world, warm=None):
    def nll2(z):
        th = np.array([a0, a1, np.exp(z[0]), np.exp(z[1])])
        v = -lik.loglik(scheme, th, world)
        return v if np.isfinite(v) else 1e12

    starts = [np.array([np.log(0.63), np.log(0.45)]), np.array([np.log(2.0), np.log(0.3)]),
              np.array([np.log(0.1), np.log(0.8)])]
    if warm is not None:
        starts.insert(0, warm)
    best = None
    for z0 in starts:
        r = minimize(nll2, z0, method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-6, "fatol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


def main():
    rng = np.random.default_rng(2024)
    world = simulate_world(rng, THETA, DESIGN)
    print(f"world: d1={world['d1']} r1={world['r1']} stayers={len(world['stayers'])} "
          f"stayer failures={sum(1 for t in world['stayers'] if t is not None)}")

    a0_grid = np.linspace(-4.6, 0.7, 23)
    a1_grid = np.linspace(0.4, 3.1, 19)
    results = {}
    for scheme in ["a", "c"]:
        prof = np.full((len(a1_grid), len(a0_grid)), np.nan)
        zmap = {}
        for i, a1 in enumerate(a1_grid):
            warm = None
            for j, a0 in enumerate(a0_grid):
                v, z = profile_over_lam_q(scheme, a0, a1, world, warm)
                prof[i, j] = v
                zmap[(i, j)] = z
                warm = z
        results[scheme] = prof
        np.save(f"results/ridge_prof_{scheme}.npy", prof)
    np.savez("results/ridge_grids.npz", a0=a0_grid, a1=a1_grid)

    # log-xi range inside the ~15%-of-max profile region (rough 1.5-loglik-unit contour)
    print("\nlog xi_0.1(x=0) at grid nodes (scheme a shows the ridge):")
    for scheme in ["a", "c"]:
        prof = results[scheme]
        cut = prof.max() - 1.5
        lxs = []
        for i in range(len(a1_grid)):
            for j in range(len(a0_grid)):
                if prof[i, j] >= cut:
                    lx = np.log(ig.xi_p_approx(0.1, 0.0, np.array([a0_grid[j], a1_grid[i], 0.6285, 0.4491]), 30.0))
                    lxs.append((a0_grid[j], a1_grid[i], lx))
        arr = np.array(lxs)
        print(f"{scheme}: region has {len(arr)} nodes; log-xi range [{arr[:,2].min():.2f}, {arr[:,2].max():.2f}]"
              f"  -> xi factor e^{arr[:,2].max()-arr[:,2].min():.1f}")

    # figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, scheme, title in [(axes[0], "a", "(a) discard withdrawn units"),
                              (axes[1], "c", "(c) SLT-D degradation follow-up")]:
        prof = results[scheme]
        rel = prof - prof.max()
        A0, A1 = np.meshgrid(a0_grid, a1_grid)
        cs = ax.contourf(A0, A1, rel, levels=np.arange(-12, 0.5, 0.75), cmap="viridis")
        ax.contour(A0, A1, rel, levels=[-1.5], colors="white", linewidths=1.2)
        ax.plot(THETA[0], THETA[1], "r*", ms=14, label="true $(\\alpha_0,\\alpha_1)$")
        if scheme == "a":
            # ridge direction: alpha0 + x0*alpha1 = const through truth
            line_a1 = np.linspace(a1_grid[0], a1_grid[-1], 10)
            ax.plot(THETA[0] + DESIGN["x0"] * (THETA[1] - line_a1), line_a1, "w--", lw=1.0,
                    label="ridge $\\mu_{x_0}=\\mathrm{const}$")
        fig.colorbar(cs, ax=ax, label="profile loglik (rel.)")
        ax.set_xlabel("$\\alpha_0$")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("$\\alpha_1$")
    fig.suptitle("Profile likelihood over $(\\alpha_0,\\alpha_1)$, maximized over $(\\lambda,q)$ — same S3 world (zero failures on stage 0)")
    fig.tight_layout()
    fig.savefig("results/fig3_ridge.png", dpi=170)
    print("wrote results/fig3_ridge.png")


if __name__ == "__main__":
    main()
