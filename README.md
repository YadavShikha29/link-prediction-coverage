# How Far Do New Links Reach? Hop-Distance Coverage of Neighbourhood Link Prediction

Code and results for the paper "How Far Do New Links Reach? A Hop-Distance Coverage Study of Neighbourhood Link Prediction".

For links hidden from a graph, we measure how many hops apart their endpoints are (2-hop, 3-hop, 4+ hops) and relate this to network density, average degree and clustering. We then compare seven link-prediction scores in three settings: all links, 2-hop candidates only, and links beyond two hops.

*Scores:* Common Neighbours (CN), Jaccard, Adamic–Adar (AA), Resource Allocation (RA), Preferential Attachment (PA), Local Path (LP), and RA+ (RA with a bounded 3-hop tie-break). A Logistic Regression and a Random Forest on these scores are also tested.

## Project files

| File | Purpose |
|---|---|
| data_loader.py | Loads datasets (SNAP files are downloaded automatically) |
| lp_core.py | Edge split, negative sampling, the seven scores, metrics, ML extension |
| run_experiments.py | Runs all experiments and writes CSV files to results/ |
| analyze.py | Creates tables, statistical tests and figures from the CSV files |
| coverage_avgdeg.py | Correlation of 2-hop share with average degree and clustering |
| make_figures.py | Creates the figures used in the paper |
| test_core.py | Sanity tests (scores match NetworkX, no data leakage) |
| requirements.txt | Python packages |                                                            
                                                                                                                                ## Datasets
All datasets come from [SNAP](http://snap.stanford.edu/data) and are downloaded into data/ on first use (the folder is not included in this repository):

- ego-Facebook, Email-Eu-core, CA-GrQc, CA-HepTh, Wiki-Vote
- Zachary Karate Club (included in NetworkX, used only as a sanity check)

Directed graphs are made undirected, self-loops are removed, and the largest connected component is kept. If the automatic download fails, download the files manually from the SNAP website and place them in data/.
