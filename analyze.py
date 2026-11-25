"""
analyze.py  --  turns results/*.csv into the tables, statistics and figures of the paper.

  python analyze.py --results results

Creates in results/:
  tables/*.csv|.md|.tex    graph stats, main results, ranks, significance, density correlation
  figures/*.png            bar charts, density curves, scatter, feature importance
  summary.md               everything in one readable file
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

HEUR = ["CN", "Jaccard", "AA", "RA", "PA", "LP", "RA+"]
ALL_METHODS = HEUR + ["LogReg", "RandForest"]
NEGS = ["easy", "hop2", "far"]


# ------------------------------------------------------------- helpers
def df_to_md(df, floatfmt="{:.4f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(x) if isinstance(x, (float, np.floating)) else str(x) for x in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def df_to_latex(df, caption, label, floatfmt="{:.3f}"):
    cols = list(df.columns)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
    out = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", r"\hline",
           " & ".join(esc(c) for c in cols) + r" \\", r"\hline"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(x) if isinstance(x, (float, np.floating)) else esc(x) for x in r]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def save_table(df, name, outdir, caption="", floatfmt="{:.4f}"):
    os.makedirs(f"{outdir}/tables", exist_ok=True)
    df.to_csv(f"{outdir}/tables/{name}.csv", index=False)
    open(f"{outdir}/tables/{name}.md", "w").write(df_to_md(df, floatfmt))
    open(f"{outdir}/tables/{name}.tex", "w").write(
        df_to_latex(df, caption or name.replace("_", " "), f"tab:{name}", "{:.3f}"))
    return df_to_md(df, floatfmt)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# -------------------------------------------------------------- tables
def table_graph_stats(stats):
    df = stats[stats.variant == "full"].copy()
    cols = ["dataset", "nodes", "edges", "density", "avg_degree", "max_degree",
            "avg_clustering", "assortativity", "degree_heterogeneity"]
    df = df[cols].sort_values("density").reset_index(drop=True)
    for c in ["nodes", "edges", "max_degree"]:
        df[c] = df[c].astype(int)
    return df


def table_main(res, neg_type, metric="auc"):
    d = res[(res.variant == "full") & (res.neg_type == neg_type)]
    g = d.groupby(["dataset", "method"])[metric].agg(["mean", "std"]).reset_index()
    g["txt"] = g.apply(lambda r: f"{r['mean']:.3f} ± {0 if np.isnan(r['std']) else r['std']:.3f}", axis=1)
    wide = g.pivot(index="dataset", columns="method", values="txt")
    wide = wide[[m for m in ALL_METHODS if m in wide.columns]].reset_index()
    return wide


def table_ranks(res, neg_type, metric="auc", methods=HEUR):
    d = res[(res.variant == "full") & (res.neg_type == neg_type) & (res.method.isin(methods))]
    p = d.pivot_table(index=["dataset", "seed"], columns="method", values=metric)
    ranks = p.rank(axis=1, ascending=False, method="average")
    out = ranks.mean().sort_values().reset_index()
    out.columns = ["method", "avg_rank (1=best)"]
    out["mean_" + metric] = [p[m].mean() for m in out["method"]]
    return out


def table_wilcoxon(res, neg_type, metric="auc", methods=HEUR):
    d = res[(res.variant == "full") & (res.neg_type == neg_type) & (res.method.isin(methods))]
    p = d.pivot_table(index=["dataset", "seed"], columns="method", values=metric)
    rows = []
    for a, b in itertools.combinations(methods, 2):
        diff = (p[a] - p[b]).dropna()
        try:
            pv = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
        except ValueError:
            pv = 1.0
        rows.append({"A": a, "B": b, "mean_diff(A-B)": diff.mean(), "n_pairs": len(diff), "p_raw": pv})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"].values)
    out["significant(0.05)"] = np.where(out["p_holm"] < 0.05, "yes", "no")
    return out


def table_density_corr(res, metric="auc"):
    rows = []
    for neg in NEGS:
        d = res[res.neg_type == neg].groupby(["dataset", "variant", "method"]).agg(
            density=("density", "first"), val=(metric, "mean")).reset_index()
        for m in ALL_METHODS:
            x = d[d.method == m]
            if len(x) >= 4:
                rho, pv = spearmanr(x["density"], x["val"])
                rows.append({"neg_type": neg, "method": m, "spearman_rho": rho, "p_value": pv,
                             "n_graphs": len(x)})
    return pd.DataFrame(rows)


def table_coverage(res, stats):
    d = res[(res.neg_type == "easy") & (res.method == "CN")].groupby(["dataset", "variant"]).agg(
        density=("density", "first"), hop2=("cov2", "mean"), hop3=("cov3", "mean"),
        hop4plus=("cov4p", "mean")).reset_index()
    st = stats[["dataset", "variant", "avg_clustering"]]
    d = d.merge(st, on=["dataset", "variant"], how="left")
    return d.sort_values("density").reset_index(drop=True)


def table_coverage_corr(cov):
    rows = []
    for col in ["hop2", "hop3", "hop4plus"]:
        for xcol in ["density", "avg_clustering"]:
            if len(cov) >= 4:
                rho, pv = spearmanr(cov[xcol], cov[col])
                rows.append({"share_of_test_links": col, "vs": xcol, "spearman_rho": rho,
                             "p_value": pv, "n_graphs": len(cov)})
    return pd.DataFrame(rows)


def table_cov_vs_auc(res, cov):
    """Does the share of 2-hop links explain heuristic accuracy on ALL links (easy setting)?"""
    d = res[res.neg_type == "easy"].groupby(["dataset", "variant", "method"])["auc"].mean().reset_index()
    d = d.merge(cov[["dataset", "variant", "hop2"]], on=["dataset", "variant"])
    rows = []
    for m in ALL_METHODS:
        x = d[d.method == m]
        if len(x) >= 4:
            rho, pv = spearmanr(x["hop2"], x["auc"])
            rows.append({"method": m, "spearman(hop2 share, AUC_easy)": rho, "p_value": pv, "n_graphs": len(x)})
    return pd.DataFrame(rows)


def table_fallback(res, cov):
    """What does a 3-hop fallback add?  RA vs RA+ vs LP, on all links and on far links."""
    g = res[res.variant == "full"].groupby(["dataset", "neg_type", "method"])["auc"].mean().reset_index()
    rows = []
    for ds, gd in g.groupby("dataset"):
        r = {"dataset": ds}
        c = cov[(cov.dataset == ds) & (cov.variant == "full")]
        r["share_beyond_2hop"] = float(1 - c["hop2"].iloc[0]) if len(c) else np.nan
        for neg in ["easy", "far"]:
            for m in ["RA", "RA+", "LP"]:
                v = gd[(gd.neg_type == neg) & (gd.method == m)]["auc"]
                r[f"{m}_{neg}"] = float(v.iloc[0]) if len(v) else np.nan
        r["gain_easy(RA+ - RA)"] = r["RA+_easy"] - r["RA_easy"]
        rows.append(r)
    return pd.DataFrame(rows)


def table_ml_gain(res, metric="auc"):
    d = res[res.variant == "full"].groupby(["dataset", "neg_type", "method"])[metric].mean().reset_index()
    rows = []
    for (ds, neg), g in d.groupby(["dataset", "neg_type"]):
        h = g[g.method.isin(HEUR)].sort_values(metric, ascending=False).iloc[0]
        r = {"dataset": ds, "neg_type": neg, "best_heuristic": h["method"], "heur_" + metric: h[metric]}
        for m in ["LogReg", "RandForest"]:
            v = g[g.method == m][metric]
            r[m] = float(v.iloc[0]) if len(v) else np.nan
        r["RF_gain"] = r["RandForest"] - r["heur_" + metric]
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- figures
def fig_main(res, outdir, metric="auc"):
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    methods = [m for m in ALL_METHODS if m in res.method.unique()]
    fig, axes = plt.subplots(1, len(NEGS), figsize=(7.5 * len(NEGS), 5), sharey=True)
    for ax, neg in zip(axes, NEGS):
        d = res[(res.variant == "full") & (res.neg_type == neg)]
        g = d.groupby(["dataset", "method"])[metric].agg(["mean", "std"]).reset_index()
        dss = list(g.dataset.unique())
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            gm = g[g.method == m].set_index("dataset").reindex(dss)
            ax.bar(np.arange(len(dss)) + i * w, gm["mean"], w, yerr=gm["std"], label=m, capsize=1.5)
        ax.set_xticks(np.arange(len(dss)) + 0.4 - w / 2)
        ax.set_xticklabels(dss, rotation=40, ha="right")
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.set_title(f"{metric.upper()} with {neg} negatives")
        ax.set_ylim(0.0, 1.02)
    axes[0].set_ylabel(metric.upper())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="upper center", fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(f"{outdir}/figures/fig_main_{metric}.png", dpi=200)
    plt.close()


def fig_density_curves(res, outdir, metric="auc"):
    sweep = [d for d, g in res.groupby("dataset") if g[g.variant.str.startswith("edge")].shape[0] > 0]
    if not sweep:
        return
    fig, axes = plt.subplots(len(NEGS), len(sweep), figsize=(4.6 * len(sweep), 3.6 * len(NEGS)), squeeze=False)
    for j, ds in enumerate(sweep):
        for i, neg in enumerate(NEGS):
            d = res[(res.dataset == ds) & (res.neg_type == neg) & (res.variant.str.match(r"^(full|edge)"))]
            g = d.groupby(["method", "density"])[metric].mean().reset_index()
            for m in [x for x in ALL_METHODS if x in g.method.unique()]:
                gm = g[g.method == m].sort_values("density")
                axes[i, j].plot(gm.density, gm[metric], marker="o", ms=3,
                                ls="--" if m in ("LogReg", "RandForest") else "-", label=m)
            axes[i, j].set_xscale("log")
            axes[i, j].set_title(f"{ds} ({neg} neg.)")
            axes[i, j].set_xlabel("density (edge-sampled versions)")
            axes[i, j].set_ylabel(metric.upper())
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{outdir}/figures/fig_density_curves.png", dpi=200)
    plt.close()


def fig_density_scatter(res, outdir, metric="auc"):
    fig, axes = plt.subplots(1, len(NEGS), figsize=(6.5 * len(NEGS), 4.6), sharey=True)
    for ax, neg in zip(axes, NEGS):
        d = res[res.neg_type == neg].groupby(["dataset", "variant", "method"]).agg(
            density=("density", "first"), val=(metric, "mean")).reset_index()
        for m in HEUR:
            x = d[d.method == m]
            ax.scatter(x.density, x.val, s=14, alpha=0.7, label=m)
        ax.set_xscale("log")
        ax.set_xlabel("network density")
        ax.set_title(f"all graphs, {neg} negatives")
    axes[0].set_ylabel(metric.upper())
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outdir}/figures/fig_density_scatter.png", dpi=200)
    plt.close()


def fig_coverage(res, stats, outdir):
    cov = table_coverage(res, stats)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for col, lab in [("hop2", "2-hop"), ("hop3", "3-hop"), ("hop4plus", "4+ hops")]:
        ax[0].scatter(cov.density, cov[col], s=16, alpha=0.75, label=lab)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("network density")
    ax[0].set_ylabel("share of held-out links")
    ax[0].set_title("How far do new links reach?")
    ax[0].legend()
    ax[1].scatter(cov.avg_clustering, cov.hop2, s=16, c="tab:blue")
    ax[1].set_xlabel("average clustering coefficient")
    ax[1].set_ylabel("share of 2-hop links")
    ax[1].set_title("2-hop coverage vs clustering")
    plt.tight_layout()
    plt.savefig(f"{outdir}/figures/fig_coverage.png", dpi=200)
    plt.close()


def fig_importance(imp, outdir):
    if imp is None or imp.empty:
        return
    d = imp[imp.variant == "full"].groupby("dataset")[HEUR].mean()
    d.plot(kind="bar", figsize=(10, 4.2))
    plt.ylabel("Random-Forest feature importance")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(f"{outdir}/figures/fig_feature_importance.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    out = a.results
    res = pd.read_csv(f"{out}/raw_results.csv")
    stats = pd.read_csv(f"{out}/graph_stats.csv")
    impf = f"{out}/feature_importance.csv"
    imp = pd.read_csv(impf) if os.path.exists(impf) and os.path.getsize(impf) > 5 else None

    md = ["# Results summary (auto-generated)\n"]
    md.append("## Table 1. Graph statistics\n" + save_table(table_graph_stats(stats), "table1_graph_stats", out,
              "Statistics of the datasets used (largest connected component).", "{:.4f}"))
    for neg in NEGS:
        for metric in ["auc", "ap", "p_at_k"]:
            t = table_main(res, neg, metric)
            md.append(f"\n## Main results: {metric.upper()} ({neg} negatives), mean ± std over seeds\n"
                      + save_table(t, f"main_{metric}_{neg}", out,
                                   f"{metric.upper()} ({neg} negatives), mean $\\pm$ std."))
        md.append(f"\n## Average rank of heuristics by AUC ({neg} negatives)\n"
                  + save_table(table_ranks(res, neg), f"ranks_auc_{neg}", out))
        md.append(f"\n## Wilcoxon signed-rank tests on AUC ({neg} negatives, Holm-corrected)\n"
                  + save_table(table_wilcoxon(res, neg), f"wilcoxon_auc_{neg}", out))
    md.append("\n## Density vs AUC (Spearman correlation across all graph variants)\n"
              + save_table(table_density_corr(res), "density_correlation", out))
    cov = table_coverage(res, stats)
    md.append("\n## Coverage: share of held-out links by hop distance in the training graph\n"
              + save_table(cov, "coverage_by_hops", out))
    md.append("\n## Coverage vs density / clustering (Spearman over all graph variants)\n"
              + save_table(table_coverage_corr(cov), "coverage_correlation", out))
    md.append("\n## Does 2-hop coverage explain accuracy on all links? (easy setting)\n"
              + save_table(table_cov_vs_auc(res, cov), "coverage_vs_auc", out))
    md.append("\n## 3-hop fallback: RA vs RA+ vs LP (AUC; far = links beyond 2 hops)\n"
              + save_table(table_fallback(res, cov), "fallback_gain", out))
    ml = table_ml_gain(res)
    if ml["RandForest"].notna().any():
        md.append("\n## ML extension: heuristics-as-features vs best single heuristic (AUC)\n"
                  + save_table(ml, "ml_gain", out))
    open(f"{out}/summary.md", "w").write("\n".join(md))

    fig_main(res, out, "auc")
    fig_main(res, out, "ap")
    fig_density_curves(res, out)
    fig_density_scatter(res, out)
    fig_coverage(res, stats, out)
    fig_importance(imp, out)
    print(f"Done. See {out}/summary.md, {out}/tables/ and {out}/figures/")


if __name__ == "__main__":
    main()
