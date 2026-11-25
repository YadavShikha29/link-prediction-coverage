"""
run_experiments.py  --  runs the whole experimental study.

Examples
  python run_experiments.py --preset demo                 # built-in + synthetic (no internet)
  python run_experiments.py --preset snap --seeds 10      # the real paper experiments
  python run_experiments.py --datasets facebook ca-grqc --seeds 5

Outputs (in --out, default results/):
  graph_stats.csv          one row per graph variant (n, m, density, clustering, ...)
  raw_results.csv          one row per (dataset, variant, seed, neg_type, method)
  feature_importance.csv   Random-Forest importances of CN/Jaccard/AA/RA/PA
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from data_loader import get_graph
from lp_core import edge_array, graph_stats, kcore_graph, run_one, sparsify

PRESETS = {
    "demo": dict(
        datasets=["karate", "lesmis", "syn-er-sparse", "syn-er-dense", "syn-ba-sparse",
                  "syn-ba-dense", "syn-ws-sparse", "syn-ws-dense", "syn-plc"],
        sweep=["syn-plc", "syn-ws-dense"]),
    "snap": dict(
        datasets=["karate", "ca-grqc", "ca-hepth", "email-eu-core", "facebook", "wiki-vote"],
        sweep=["ca-grqc", "email-eu-core", "facebook"]),
}


def build_variants(G, name, do_sweep, fracs, kcores, rng):
    variants = [("full", G)]
    if do_sweep:
        for f in fracs:
            if f < 1.0:
                variants.append((f"edge{f:g}", sparsify(G, f, rng)))
        seen = {G.number_of_edges()}          # skip k-cores identical to a previous graph
        for k in kcores:
            try:
                H = kcore_graph(G, k)
            except Exception:
                continue
            m = H.number_of_edges()
            if m in seen or m < 0.02 * G.number_of_edges() or H.number_of_nodes() <= 50:
                continue
            seen.add(m)
            variants.append((f"kcore{k}", H))
    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--sweep", nargs="*", default=None, help="datasets that also get density variants")
    ap.add_argument("--fracs", nargs="*", type=float, default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--kcores", nargs="*", type=int, default=[3, 5, 10])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--no_ml", action="store_true")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    preset = PRESETS[a.preset] if a.preset else {"datasets": [], "sweep": []}
    datasets = a.datasets if a.datasets else preset["datasets"]
    sweep = set(a.sweep if a.sweep is not None else preset["sweep"])
    if not datasets:
        ap.error("give --preset or --datasets")
    os.makedirs(a.out, exist_ok=True)

    stats_rows, res_rows, imp_rows = [], [], []
    for name in datasets:
        t0 = time.time()
        G = get_graph(name, a.data_dir)
        vrng = np.random.default_rng(12345)
        variants = build_variants(G, name, name in sweep, a.fracs, a.kcores, vrng)
        print(f"\n=== {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"{len(variants)} variant(s)")
        for vname, H in variants:
            if H.number_of_edges() < 40:
                print(f"   skip {vname}: too few edges")
                continue
            st = graph_stats(H)
            st.update(dataset=name, variant=vname)
            stats_rows.append(st)
            n, E = H.number_of_nodes(), edge_array(H)
            for seed in range(a.seeds):
                rows, imp = run_one(E, n, seed, a.test_frac, ml=not a.no_ml)
                for r in rows:
                    r.update(dataset=name, variant=vname, density=st["density"])
                res_rows.extend(rows)
                if imp:
                    imp_rows.append({"dataset": name, "variant": vname, "seed": seed, **imp})
            print(f"   {vname:9s} density={st['density']:.5f} clustering={st['avg_clustering']:.3f} done")
        # save after every dataset so a crash never loses finished work
        pd.DataFrame(stats_rows).to_csv(f"{a.out}/graph_stats.csv", index=False)
        pd.DataFrame(res_rows).to_csv(f"{a.out}/raw_results.csv", index=False)
        pd.DataFrame(imp_rows).to_csv(f"{a.out}/feature_importance.csv", index=False)
        print(f"   [{name}] finished in {time.time() - t0:.1f}s")
    print("\nDone. Now run:  python analyze.py --results", a.out)


if __name__ == "__main__":
    main()
