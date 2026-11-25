import pandas as pd
from scipy.stats import spearmanr
st  = pd.read_csv("results/graph_stats.csv")
raw = pd.read_csv("results/raw_results.csv")
cov = (raw[(raw.neg_type=="easy") & (raw.method=="CN")]
       .groupby(["dataset","variant"])[["cov2","cov3","cov4p"]].mean().reset_index())
d = cov.merge(st[["dataset","variant","nodes","avg_degree","avg_clustering"]],
              on=["dataset","variant"])
for x in ["avg_degree","avg_clustering"]:
    print(x, spearmanr(d[x], d["cov2"]))
d.to_csv("results/coverage_vs_avgdegree.csv", index=False)