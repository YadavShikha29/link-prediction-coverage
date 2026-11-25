import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="figures")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

cov = pd.read_csv(os.path.join(a.results, "coverage_vs_avgdegree.csv"))
need = {"dataset", "variant", "cov2", "avg_degree", "avg_clustering"}
assert need <= set(cov.columns), f"missing columns: {need - set(cov.columns)}"

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 7.5, "axes.linewidth": 0.6})
NAME = {"facebook": "Facebook", "email-eu-core": "Email-Eu-core", "ca-grqc": "CA-GrQc",
        "ca-hepth": "CA-HepTh", "wiki-vote": "Wiki-Vote", "karate": "Karate Club"}

# ---------------------------------------------------------------- Fig. 1
# 2-hop share vs fraction of edges kept (edge-thinned variants + the full graph)
frac = {"edge0.2": 0.2, "edge0.4": 0.4, "edge0.6": 0.6, "edge0.8": 0.8, "full": 1.0}
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for d, m in zip(["ca-grqc", "facebook", "email-eu-core"], ["o", "s", "^"]):
    s = cov[(cov.dataset == d) & cov.variant.isin(frac)].copy()
    s["f"] = s.variant.map(frac)
    s = s.sort_values("f")
    ax.plot(s.f, s.cov2, marker=m, ms=3.5, lw=1, label=NAME[d])
ax.set_xlabel("Fraction of edges retained before the split")
ax.set_ylabel("2-hop share of held-out links")
ax.set_ylim(0, 1.03)
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(alpha=0.3, lw=0.4)
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(a.out, "fig1.png"), dpi=300)
plt.close()

# ---------------------------------------------------------------- Fig. 2
# 2-hop share vs average degree (log scale) and vs average clustering, all 27 graphs
rho_deg = spearmanr(cov.avg_degree, cov.cov2)[0]
rho_clu = spearmanr(cov.avg_clustering, cov.cov2)[0]
print(f"Spearman(avg_degree, cov2)  = {rho_deg:.3f}")
print(f"Spearman(avg_clustering, cov2) = {rho_clu:.3f}")
marks = {"ca-grqc": "o", "ca-hepth": "D", "facebook": "s", "email-eu-core": "^", "wiki-vote": "v", "karate": "x"}
fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.5), sharey=True)
for ax, col, lab in zip(axes, ["avg_degree", "avg_clustering"], ["Average degree (log scale)", "Average clustering"]):
    for k, m in marks.items():
        s = cov[cov.dataset == k]
        if len(s):
            ax.scatter(s[col], s.cov2, s=11, marker=m, lw=0.8, label=NAME[k])
    ax.set_xlabel(lab)
    ax.grid(alpha=0.3, lw=0.4)
    ax.set_ylim(0.25, 1.03)
axes[0].set_xscale("log")
axes[0].xaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 50]))
axes[0].xaxis.set_major_formatter(FixedFormatter(["3", "5", "10", "20", "50"]))
axes[0].xaxis.set_minor_locator(NullLocator())
axes[0].set_ylabel("2-hop share of held-out links")
axes[0].set_title(f"(a)  \u03C1 = {rho_deg:.2f}", fontsize=7.5)
axes[1].set_title(f"(b)  \u03C1 = {rho_clu:.2f}", fontsize=7.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, loc="lower center", frameon=False, fontsize=6, handletextpad=0.2, columnspacing=0.8)
plt.tight_layout(rect=(0, 0.13, 1, 1))
plt.savefig(os.path.join(a.out, "fig2.png"), dpi=300)
plt.close()

# ---------------------------------------------------------------- data for Fig. 3-5
raw = pd.read_csv(os.path.join(a.results, "raw_results.csv"))
imp = pd.read_csv(os.path.join(a.results, "feature_importance.csv"))
# ---- Fig 3: AUC (easy) vs 2-hop share, 27 graphs
e = raw[raw.neg_type == "easy"].groupby(["dataset", "variant", "method"]).auc.mean().reset_index()
e = e.merge(cov[["dataset", "variant", "cov2"]], on=["dataset", "variant"])
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for m, mk, lab in [("RA", "o", "RA"), ("CN", "s", "CN"), ("PA", "^", "PA")]:
    s = e[e.method == m]; rho = spearmanr(s.cov2, s.auc)[0]
    ax.scatter(s.cov2, s.auc, s=11, marker=mk, label=f"{lab} (\u03C1 = {rho:.2f})", alpha=.85)
ax.set_xlabel("2-hop share of held-out links"); ax.set_ylabel("AUC, all links vs. random negatives")
ax.grid(alpha=.3, lw=.4); ax.legend(frameon=False, fontsize=6.5, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig3.png"), dpi=300); plt.close()

# ---- Fig 4: AUC by setting, real networks, mean +- std over splits (full width)
meths = ["CN", "AA", "RA", "PA", "LP", "RA+", "RandForest"]; lab = ["CN", "AA", "RA", "PA", "LP", "RA+", "RF"]
real = ["facebook", "email-eu-core", "ca-grqc", "ca-hepth", "wiki-vote"]
f = raw[raw.variant == "full"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
for ax, neg, ttl in zip(axes, ["easy", "hop2", "far"], ["(a) All links vs. random negatives", "(b) 2-hop candidate ranking", "(c) Links beyond two hops"]):
    g = f[f.neg_type == neg].groupby(["dataset", "method"]).auc.agg(["mean", "std"])
    w = 0.8 / len(meths)
    for i, (m, lb) in enumerate(zip(meths, lab)):
        ax.bar(np.arange(len(real)) + i * w, [g.loc[(d, m), "mean"] for d in real], w,
               yerr=[g.loc[(d, m), "std"] for d in real], capsize=0.8, error_kw={"lw": .5}, label=lb)
    ax.axhline(.5, color="k", ls=":", lw=.7)
    ax.set_xticks(np.arange(len(real)) + 3 * w); ax.set_xticklabels([NAME[d] for d in real], rotation=25, ha="right", fontsize=6)
    ax.set_title(ttl, fontsize=7); ax.set_ylim(0.3, 1.02)
axes[0].set_ylabel("AUC")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=7, loc="upper center", frameon=False, fontsize=6.5, columnspacing=1.0, handletextpad=.3)
plt.tight_layout(rect=(0, 0, 1, 0.9)); plt.savefig(os.path.join(a.out, "fig4.png"), dpi=300); plt.close()

# ---- Fig 5: Random-Forest feature importance (full graphs)
feats = ["CN", "Jaccard", "AA", "RA", "PA", "LP", "RA+"]
im = imp[imp.variant == "full"].groupby("dataset")[feats].mean()
order = ["ca-grqc", "ca-hepth", "email-eu-core", "facebook", "wiki-vote", "karate"]
fig, ax = plt.subplots(figsize=(3.4, 2.4)); w = 0.8 / len(feats)
for i, ft in enumerate(feats):
    ax.bar(np.arange(len(order)) + i * w, [im.loc[d, ft] for d in order], w, label=ft)
ax.set_xticks(np.arange(len(order)) + 3 * w); ax.set_xticklabels([NAME[d] for d in order], rotation=30, ha="right", fontsize=6.5)
ax.set_ylabel("Random-Forest feature importance"); ax.grid(axis="y", alpha=.3, lw=.4)
ax.legend(ncol=4, frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.2), columnspacing=.8, handletextpad=.3)
plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig5.png"), dpi=300); plt.close()


print("saved fig1-fig5 in", a.out)