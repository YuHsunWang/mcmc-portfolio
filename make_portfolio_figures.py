"""Generate portfolio figures (convergence + benchmark) from the included results.

Reads the realistic-scenario MCMC output and writes two PNGs into figures/.
Paths are relative to this script, so it runs anywhere the repo is cloned.
English labels only (avoids CJK font issues).

    python make_portfolio_figures.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent
OUT = SRC / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RES = SRC / "outputs_realistic_100u_10000iter"
npz = np.load(RES / "gpu_mcmc_results.npz", allow_pickle=True)
summary = json.loads((RES / "gpu_mcmc_summary.json").read_text())

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
TEAL = "#0f766e"
AMBER = "#d97706"
SLATE = "#475569"

# ---------------------------------------------------------------- Figure 1
accept = npz["accept_trace"]          # (n_iter, 3): a_u, s_u, C_switch
acc = npz["accuracy_trace"]           # (n_iter,)
theta_s = npz["theta_s_trace"]        # (n_iter,)
theta_switch = npz["theta_switch_trace"]
demo_s = npz["demo_s_trace"]          # (n_iter, 5)
n_iter = accept.shape[0]
it = np.arange(n_iter)

fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))

# (a) acceptance rates
labels = [r"$a_u$ (preference)", r"$s_u$ (diversity)", r"$C^{switch}_u$"]
colors = [SLATE, TEAL, AMBER]
for j, (lab, c) in enumerate(zip(labels, colors)):
    ax[0, 0].plot(it, accept[:, j], color=c, lw=1.3, label=lab)
ax[0, 0].set_title("(a) Metropolis-Hastings acceptance rate")
ax[0, 0].set_xlabel("MCMC iteration"); ax[0, 0].set_ylabel("acceptance rate")
ax[0, 0].set_ylim(0, 1); ax[0, 0].legend(frameon=False, fontsize=8)

# (b) train accuracy
ax[0, 1].plot(it, acc, color=TEAL, lw=1.3)
ax[0, 1].axhline(summary["final_accuracy"], color=AMBER, ls="--", lw=1,
                 label=f"final = {summary['final_accuracy']:.3f}")
ax[0, 1].set_title("(b) Train next-category accuracy")
ax[0, 1].set_xlabel("MCMC iteration"); ax[0, 1].set_ylabel("accuracy")
ax[0, 1].legend(frameon=False, fontsize=8)

# (c) population-level hyperparameter convergence
ax2 = ax[1, 0]
l1 = ax2.plot(it, theta_s, color=TEAL, lw=1.3, label=r"$\theta_s$ (pop. diversity)")
ax2.set_ylabel(r"$\theta_s$", color=TEAL); ax2.tick_params(axis="y", labelcolor=TEAL)
ax2b = ax2.twinx(); ax2b.grid(False)
l2 = ax2b.plot(it, theta_switch, color=AMBER, lw=1.3, label=r"$\theta_{switch}$ (pop. switch cost)")
ax2b.set_ylabel(r"$\theta_{switch}$", color=AMBER); ax2b.tick_params(axis="y", labelcolor=AMBER)
ax2.set_title("(c) Population-level hyperparameter convergence")
ax2.set_xlabel("MCMC iteration")
ax2.legend(l1 + l2, [h.get_label() for h in l1 + l2], frameon=False, fontsize=8, loc="center right")

# (d) user-level parameter traces (5 demo users)
for u in range(demo_s.shape[1]):
    ax[1, 1].plot(it, demo_s[:, u], lw=1.0, alpha=0.85, label=f"user {npz['demo_user_ids'][u]}")
ax[1, 1].set_title(r"(d) User-level diversity $s_u$ traces (5 sampled users)")
ax[1, 1].set_xlabel("MCMC iteration"); ax[1, 1].set_ylabel(r"$s_u$")
ax[1, 1].legend(frameon=False, fontsize=7, ncol=2)

fig.suptitle("MCMC convergence diagnostics — realistic sparse-interest scenario "
             f"(100 users, {n_iter:,} iterations)", fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "convergence.png", bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "convergence.png")

# ---------------------------------------------------------------- Figure 2
bm = pd.read_csv(RES / "benchmark_comparison.csv")
bm.columns = [c.strip().lstrip("﻿") for c in bm.columns]
order = ["MCMC Preference Model", "Markov Transition", "User Most Frequent", "Global Popularity"]
bm = bm.set_index("Model").loc[order].reset_index()
metrics = ["Top-1 Accuracy", "Top-3 Accuracy", "Sequence Similarity"]
bar_colors = [TEAL, SLATE, AMBER, "#94a3b8"]

x = np.arange(len(metrics)); w = 0.2
fig, ax = plt.subplots(figsize=(9, 5))
for i, model in enumerate(bm["Model"]):
    vals = bm.loc[i, metrics].astype(float).values
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=model, color=bar_colors[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel("score"); ax.set_ylim(0, 1.0)
ax.set_title("Held-out benchmark comparison — realistic sparse-interest scenario")
ax.legend(frameon=False, fontsize=8.5, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "benchmark.png", bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "benchmark.png")
