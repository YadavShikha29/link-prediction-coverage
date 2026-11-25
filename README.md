# How Far Do New Links Reach? — Coverage study of neighbourhood link prediction (full project)

**Final topic:** *Hop-distance coverage of neighbourhood link prediction across network density, with a bounded 3-hop fallback (RA+).*

Kya measure hota hai:
1. Held-out links ka kitna share **2-hop / 3-hop / 4+ hop** par hai, aur ye **density aur clustering** ke saath kaise badalta hai.
2. Kya ye 2-hop share heuristics (CN, Jaccard, AA, RA, PA, LP, RA+) ki accuracy **explain** karta hai?
3. Teen evaluation settings: **easy** (saare links vs random negatives), **hop2** (2-hop candidate ranking), **far** (2-hop se door ke links).
4. **RA+** (RA + bounded 3-hop tie-break) vs **LP** (Local Path index) vs RA, aur ML (LogReg/RandForest on scores).

> Uniqueness ka guarantee nahi hai. Related work (LP 2009, Mao et al. 2023, Sarkar et al. 2015 ...) `paper/REFERENCES.md` me hai.
> Submit karne se pehle Google Scholar par khud verify karo (paper draft Section 2 me steps hain).

## Files
| File | Kaam |
|---|---|
| `data_loader.py` | Datasets: built-in, synthetic, SNAP auto-download, custom edge list |
| `lp_core.py` | Split, negatives, 7 scores (fast sparse), coverage, 3 settings, metrics, ML |
| `test_core.py` | Tests: scores == networkx, 3-hop scores == dense formulas, no leakage, valid negatives |
| `run_experiments.py` | Poora experiment -> `results/*.csv` |
| `analyze.py` | Tables (md/csv/LaTeX), Wilcoxon+Holm, Spearman, figures, `summary.md` |
| `paper/PAPER_DRAFT.md` | Poora paper draft (FILL placeholders) |
| `paper/REFERENCES.md` | 28 references (verify details) |
| `demo_results/` | Pipeline demo (built-in + synthetic). **Paper ke numbers nahi.** |

## Steps
```bash
pip install -r requirements.txt
python test_core.py                                    # 1. tests pass hone chahiye

python run_experiments.py --preset snap --seeds 10 --out results   # 2. real experiments
python analyze.py --results results                    # 3. tables + figures
```
SNAP download fail ho to files `data/` me daalo: `facebook_combined.txt.gz`, `email-Eu-core.txt.gz`, `ca-GrQc.txt.gz`, `ca-HepTh.txt.gz`, `wiki-Vote.txt.gz` (URLs `data_loader.py` me).

Time kam karna ho: `--seeds 5`, ya `--datasets ca-grqc email-eu-core facebook`. Facebook/wiki-vote heavy hain (3-hop scores ki wajah se); results har dataset ke baad save hote hain.
No-internet quick check: `python run_experiments.py --preset demo --seeds 5 --out demo_results`.

## Output tables (results/tables/) aur paper section
| Table | Section |
|---|---|
| `table1_graph_stats` | 4.1 |
| `coverage_by_hops`, `coverage_correlation` + `fig_coverage.png` | 5.1 (RQ1) |
| `coverage_vs_auc` | 5.2 (RQ2) |
| `main_auc_*`, `ranks_auc_hop2`, `wilcoxon_auc_*` | 5.3 (RQ3) |
| `main_auc_far`, `fallback_gain` | 5.4 (RQ4) |
| `ml_gain`, `fig_feature_importance.png` | 5.5 (RQ5) |
| `density_correlation`, `fig_density_*.png` | supporting |

## Protocol short me
20% edges hide (spanning tree train me), scores sirf train graph par, AUC/AP/Precision@k, 10 seeds, mean ± std.
Density variants: edge sampling (20-80%) + k-core + cross-network Spearman.

## Known limitations (paper me likhna)
- Random split (temporal nahi); undirected unweighted; no attributes; GNN baselines nahi.
- Edge sampling me density ke saath clustering bhi badalti hai -> `coverage_correlation` me dono ke saath correlation.
- `far` setting me CN/Jaccard/AA/RA ka AUC by construction exactly 0.5 hota hai (sab scores 0).
- hop2 me distance-2 negatives ke saath saare positives rakhna degenerate hota (AUC << 0.5), isliye sirf 2-hop positives.
- PA kabhi 0.5 se neeche aa sakta hai (test edges hide hone se endpoint degree kam hota hai).
- RA+ ek simple tie-break hai, naya SOTA model nahi.
