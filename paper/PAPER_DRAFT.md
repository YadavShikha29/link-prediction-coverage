# How Far Do New Links Reach? Hop-Distance Coverage of Neighbourhood Link Prediction across Network Density, and a Bounded 3-Hop Fallback

**Author:** Shreya  |  **Affiliation:** [college / department]  |  **Email:** [email]

> HOW TO USE THIS DRAFT
> - `[...]` placeholder hai. Numbers sirf `results/summary.md` se lo (script ne generate kiye hain).
> - "FILL" blocks me wahi likho jo tumhare actual results me sach nikle. Jo match na kare, delete.
> - Uniqueness claim mat karo. Sahi framing: "we quantify ... and evaluate ..." (Section 1 aur 2 dekho).

---

## Abstract

Neighbourhood heuristics such as Common Neighbours, Adamic–Adar and Resource Allocation (RA) are standard baselines for link prediction, but they can only score a pair that shares a neighbour, i.e. a pair at distance two. Whether this is a real limitation depends on how many *future* links actually lie within two hops, which should vary with network density and clustering. We measure, on [N] real and [N] synthetic networks and on density-controlled variants of them (edge sampling, k-cores), the share of held-out links at 2, 3 and 4+ hops in the training graph, and relate it to network density, clustering and to the accuracy of seven heuristics. We evaluate in three settings: all links against random negatives, 2-hop candidate ranking, and links beyond two hops. Finally we test RA+, a simple variant of RA whose bounded 3-hop tie-break never changes the ranking among 2-hop pairs, against the Local Path index. [FILL: 2-3 sentences with your actual findings: coverage vs density, how well coverage explains AUC, gain of RA+ / LP on far links, ML gain.]

**Keywords:** link prediction, social networks, network density, hop distance, coverage, Resource Allocation, local path

---

## 1. Introduction

Given a snapshot of a social network, link prediction ranks currently unconnected pairs by how likely they are to become connected [1], [5]. The most used scores (CN, Jaccard, AA [2], RA [3]) are *neighbourhood-based*: they are zero for any pair without a common neighbour. In many evaluations this is hidden because the pairs are compared against random negatives, and because test links in dense, clustered networks are mostly 2-hop pairs.

Prior work has shown that local heuristics such as CN lose efficacy on sparse networks [24], [27]; that higher-order paths can help with data sparsity, e.g. the Local Path index [23] and path/correlation-based combinations [25]; that CN-based prediction is consistent only when the graph is dense enough and that not every node is reachable in two hops [26]; and that evaluation with easy negatives can mislead [9]. What we found less explicit in the sources we read is a direct **measurement of the hop-distance profile of held-out links as a function of density and clustering across many networks, and how that profile explains heuristic accuracy**, together with an evaluation that separates links the heuristic can see (2-hop) from links it cannot (far links). [Update this sentence after your own literature search. Do not claim novelty beyond what you verified.]

**Research questions**
- **RQ1 (coverage):** What fraction of held-out links is at 2, 3 and 4+ hops, and how does it vary with density and clustering?
- **RQ2 (explanation):** Does the 2-hop share explain the accuracy of neighbourhood heuristics on all links?
- **RQ3 (ranking quality in sight):** On 2-hop candidates only, which heuristic is best and does the ranking change from the random-negative setting?
- **RQ4 (fallback):** On links beyond two hops, do 3-hop scores (LP, RA+) beat chance, and does RA+ keep the 2-hop quality of RA?
- **RQ5 (learning):** Do Logistic Regression / Random Forest on the heuristic scores add anything?

**Contributions**
1. A hop-distance **coverage analysis** of held-out links across [N] networks and their density variants.
2. A **three-setting evaluation** (all / 2-hop / far) that separates "what heuristics can see" from "what they cannot".
3. **RA+**, a one-line modification of RA: 3-hop resource-allocation tie-break bounded so that 2-hop ordering is unchanged; compared with LP [23].
4. Open, tested code. [repo link]

---

## 2. Related Work
- **Classical heuristics and evaluation.** [1], [2], [3], [4], [5]; AUC under imbalance [8].
- **Sparsity and higher-order paths.** LP index [23] augments CN with 3-hop path counts to handle sparse data; [25] studies sparsity by moving a growing fraction of links to the probe set and combines a correlation score with RA; [24] analyses data factors and reports CN weaker on sparse graphs; [27] targets sparse link prediction with attributes.
- **Theory.** [26] shows common neighbours are effective in sufficiently dense stochastic blockmodels, and notes not all nodes are reachable within two hops; [28] argues higher-order relations skew heuristic evaluation.
- **Learning approaches.** [7], [21], [22]; benchmark pitfalls and hard negatives [9]; heuristics remain competitive with GNNs [10].
- **Gap (write in your own words after your search):** [FILL: e.g. "None of these works reports the full hop-distance distribution of held-out links as a function of density and clustering across many networks and links it to accuracy in a three-way (all / 2-hop / far) evaluation."] Verify this against Google Scholar before submitting. If a paper already does it, cite it and reposition your contribution as a replication/extension.

---

## 3. Preliminaries
G = (V, E) undirected, unweighted, n = |V|, m = |E|, Γ(x) neighbours, k_x = |Γ(x)|, density ρ = 2m/(n(n−1)).

| Score | Definition |
|---|---|
| CN | \|Γ(x)∩Γ(y)\| |
| Jaccard | \|Γ(x)∩Γ(y)\| / \|Γ(x)∪Γ(y)\| |
| AA | Σ_{z∈Γ(x)∩Γ(y)} 1/log k_z |
| RA | Σ_{z∈Γ(x)∩Γ(y)} 1/k_z |
| PA | k_x · k_y |
| LP [23] | CN(x,y) + ε · W3(x,y), ε = 10⁻³, W3 = number of length-3 walks |
| **RA+** (ours, simple) | RA(x,y) + δ · RA3/(1+RA3), δ = 10⁻⁶, RA3(x,y) = Σ_{a∈Γ(x), b∈Γ(y), a~b} 1/(k_a k_b) |

**Hop distance.** For a held-out link (x,y) that is not in the training graph: *2-hop* if CN ≥ 1; *3-hop* if CN = 0 and W3 ≥ 1; *4+ hops* otherwise. The training graph always contains a spanning tree, so all pairs are connected.

**Property of RA+.** Since 0 ≤ RA3/(1+RA3) < 1, the added term is below δ. For pairs with CN ≥ 1 the order of RA is unchanged except for ties or differences smaller than δ (practically never); for pairs with CN = 0 it breaks the tie among zero scores using 3-hop evidence. Hence RA+ cannot be worse than RA on 2-hop candidate ranking (up to ties), while adding signal on far links. LP changes the 2-hop ordering by W3 and can therefore help or hurt.

---

## 4. Experimental Setup

### 4.1 Datasets
Largest connected component, undirected, simple (Table 1 → `table1_graph_stats`). Real: [Facebook ego [12], Email-Eu-core [14], CA-GrQc [13], CA-HepTh [13], Wiki-Vote [15], Zachary karate [16]]; synthetic (2 000 nodes; sparse and dense): ER [18] (no structure, sanity check, expected AUC ≈ 0.5), BA [4] (hubs), WS [17] (clustered), power-law-cluster. Source: SNAP [11].

### 4.2 Density-controlled variants
Edge-sampled versions (20/40/60/80% of edges) and k-cores (k = 3, 5, 10) of selected networks. Sampling reduces density but also clustering; k-cores raise density by removing peripheral nodes. Because density and clustering are partly confounded, we report Spearman correlations of coverage with both (`coverage_correlation`).

### 4.3 Protocol
1. 20% of edges hidden as positives; a random spanning tree always stays in training.
2. **Three settings** (each with equal numbers of positives and negatives):
   - *easy:* all held-out links vs random non-adjacent pairs (classic).
   - *hop2:* 2-hop positives vs non-adjacent distance-2 pairs (friend-of-friend ranking). Keeping all positives while sampling distance-2 negatives would be degenerate (neighbourhood scores fall below chance by construction), so only 2-hop positives are used.
   - *far:* held-out links with CN = 0 vs random non-adjacent pairs with CN = 0.
3. Scores use the training graph only. Metrics: AUC, AP, Precision@k (k = #positives, random tie-breaking).
4. [10] seeds; mean ± std; Wilcoxon signed-rank with Holm correction; Spearman correlations.
5. ML extension: Logistic Regression and Random Forest on the 7 scores, trained from a further hidden 20% of training edges (negatives half random, half distance-2).

### 4.4 Implementation
NumPy/SciPy sparse; scores verified against NetworkX [19] (CN, Jaccard, AA, RA, PA) and against dense-matrix formulas (LP, RA+, W3). scikit-learn [20].

---

## 5. Results (fill from `results/`)

### 5.1 Coverage (RQ1)
Table `coverage_by_hops`, `coverage_correlation`; figure `fig_coverage.png`.
[FILL: report the 2-hop / 3-hop / 4+ shares for each real network; the Spearman ρ of 2-hop share vs density and vs clustering (with p). State clearly whether density or clustering is the stronger correlate in YOUR data. Mention the confounding of edge sampling.]

### 5.2 Does coverage explain accuracy? (RQ2)
Table `coverage_vs_auc`, `main_auc_easy`.
[FILL: Spearman between 2-hop share and easy-setting AUC for CN, AA, RA etc.]
*Reading guide:* For a neighbourhood score, every held-out link without a common neighbour scores 0 and is tied with most random negatives. So easy-setting AUC is bounded by roughly cov2 · (AUC on 2-hop links) + (1 − cov2) · 0.5. Check whether your numbers follow this.

### 5.3 2-hop candidate ranking (RQ3)
Tables `main_auc_hop2`, `ranks_auc_hop2`, `wilcoxon_auc_hop2`. [FILL: best method per network type; does the ranking differ from the easy setting?] Note: PA may fall below 0.5 on near-regular graphs (hiding test edges reduces endpoint degrees); say so explicitly if you see it.

### 5.4 Links beyond two hops and the fallback (RQ4)
Tables `main_auc_far`, `fallback_gain`. CN, Jaccard, AA, RA are exactly 0.5 on far links by construction (all scores tie at zero). [FILL: AUC of LP, RA+, PA on far links per network; for which networks is it clearly above 0.5; and RA+ vs RA on the easy setting (gain), plus the Wilcoxon result for RA+ vs RA on hop2, which should show no loss.]

### 5.5 Learning extension (RQ5)
Table `ml_gain`, figure `fig_feature_importance.png`. [FILL]

---

## 6. Discussion
[FILL: (1) when are neighbourhood heuristics enough (high 2-hop share) and when not; (2) why reporting coverage and a 2-hop/far split matters for evaluation; (3) whether a cheap 3-hop fallback is worth it.]

## 7. Limitations
- Random edge split, not temporal; undirected, unweighted; no node attributes.
- Edge sampling confounds density with clustering; k-cores change the node set.
- Hard setting only uses distance-2 negatives (simplified HeaRT [9]); "far" negatives are random pairs with CN = 0.
- RA+ is a deliberately simple tie-break, not a new state-of-the-art model; no GNN baselines.
- [N] real networks, mostly collaboration and communication.

## 8. Conclusion and Future Work
[FILL: 4-5 sentences.] Future work: temporal splits, GNN/SEAL baselines [21], [22], other hop-aware scores, attributes, larger networks.

## References
`paper/REFERENCES.md` ([1]–[28]; remove unused).

### Title options
1. How Far Do New Links Reach? Hop-Distance Coverage of Neighbourhood Link Prediction across Network Density, and a Bounded 3-Hop Fallback
2. Beyond Two Hops: Coverage Limits of Neighbourhood Link Prediction in Sparse and Dense Social Networks
3. What Neighbourhood Heuristics Cannot See: A Coverage Study of Link Prediction

### Realistic venue
Student/undergraduate research tracks, regional or international data-science/SNA conferences, arXiv cs.SI. Claim: "empirical coverage study with a simple fallback".
