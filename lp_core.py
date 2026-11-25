"""
lp_core.py
----------
Core of the study: edge split, negative sampling, the 5 classical heuristics
(vectorised with scipy sparse, verified against networkx in test_core.py),
metrics, graph statistics and the ML extension.

Protocol (important for the paper's "Experimental Setup" section)
  1. Hide `test_frac` of the edges as positive test links. A random spanning
     tree is always kept in the training graph, so no node becomes isolated.
  2. Two evaluation settings:
       easy : all test positives vs uniformly random non-adjacent pairs
       hop2 : 2-hop candidate ranking (friend-of-friend recommendation).
              Positives = test edges whose endpoints share >=1 neighbour in the
              training graph; negatives = non-adjacent distance-2 pairs (same count).
              This is a hard-negative setting in the spirit of HeaRT, but fair to
              neighbourhood heuristics (positives AND negatives are 2-hop pairs).
              `cov2` = fraction of test positives that are 2-hop pairs (coverage).
       far  : links beyond 2 hops. Positives = test edges with NO common neighbour
              in the training graph; negatives = random non-adjacent pairs that
              also have no common neighbour. CN/Jaccard/AA/RA/PA-free zone: tests
              what a 3-hop fallback (LP, RA+) can add.
     Coverage: every test link is classified as 2-hop (cov2), 3-hop (cov3) or
     4+ hops (cov4p) in the training graph.
     NOTE: sampling distance-2 negatives while keeping ALL positives would be
     degenerate (AUC << 0.5 for CN/AA/RA by construction), so it is not used.
  3. ALL scores are computed on the TRAINING graph only (no leakage).
  4. Metrics: ROC-AUC, Average Precision, Precision@k (k = #positives).
"""
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSIC = ["CN", "Jaccard", "AA", "RA", "PA"]
METHODS = CLASSIC + ["LP", "RA+"]          # + 3-hop-aware scores
ML_METHODS = ["LogReg", "RandForest"]
LP_EPS = 1e-3          # Local Path index (Lu, Jin, Zhou 2009): CN + eps * #paths of length 3
RAPLUS_EPS = 1e-6      # RA+ : RA with a bounded 3-hop tie-break (2-hop pairs ALWAYS rank first)


# ----------------------------------------------------------------- basics
def edge_array(G):
    E = np.array([(min(u, v), max(u, v)) for u, v in G.edges()], dtype=np.int64)
    return E[np.lexsort((E[:, 1], E[:, 0]))]


def encode(E, n):
    return E[:, 0] * n + E[:, 1]          # requires E[:,0] < E[:,1]


def decode(codes, n):
    return np.column_stack([codes // n, codes % n])


def build_adj(E, n):
    r = np.r_[E[:, 0], E[:, 1]]
    c = np.r_[E[:, 1], E[:, 0]]
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


# ------------------------------------------------------------------ split
def split_edges(E, n, test_frac, rng):
    """Return (train_edges, test_edges). A random spanning tree stays in train."""
    m = len(E)
    w = rng.random(m) + 1e-9
    W = sp.coo_matrix((w, (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    T = minimum_spanning_tree(W).tocoo()
    tree_codes = np.minimum(T.row, T.col).astype(np.int64) * n + np.maximum(T.row, T.col)
    is_tree = np.isin(encode(E, n), tree_codes)
    cand = np.flatnonzero(~is_tree)
    n_test = min(int(round(test_frac * m)), len(cand))
    chosen = rng.choice(cand, size=n_test, replace=False)
    mask = np.zeros(m, dtype=bool)
    mask[chosen] = True
    return E[~mask], E[mask]


# ------------------------------------------------------ negative sampling
def sample_easy_negatives(n, full_codes, count, rng):
    max_possible = n * (n - 1) // 2 - len(full_codes)
    count = min(count, max_possible)
    out = np.empty(0, dtype=np.int64)
    while len(out) < count:
        k = (count - len(out)) * 2 + 10
        u = rng.integers(0, n, size=k)
        v = rng.integers(0, n, size=k)
        ok = u != v
        a, b = np.minimum(u, v)[ok], np.maximum(u, v)[ok]
        c = a.astype(np.int64) * n + b
        c = c[~np.isin(c, full_codes)]
        out = np.unique(np.r_[out, c])
    rng.shuffle(out)
    return decode(out[:count], n)


def sample_hard_negatives(A, n, exclude_codes, count, rng):
    """Non-adjacent pairs having >=1 common neighbour in A (distance 2)."""
    A2 = (A @ A).tocoo()
    mask = A2.row < A2.col
    c = A2.row[mask].astype(np.int64) * n + A2.col[mask]
    c = np.unique(c[~np.isin(c, exclude_codes)])
    if len(c) > count:
        c = rng.choice(c, size=count, replace=False)
    return decode(c, n)


# ----------------------------------------------------------------- scores
def cn_count(A, P, batch=50000):
    out = np.zeros(len(P))
    for s0 in range(0, len(P), batch):
        u, v = P[s0:s0 + batch, 0], P[s0:s0 + batch, 1]
        out[s0:s0 + len(u)] = np.asarray(A[u].multiply(A[v]).sum(axis=1)).ravel()
    return out


def score_pairs(A, P, batch=4000):
    """
    CN, Jaccard, AA, RA, PA, LP, RA+ for every pair in P (k x 2 int array) on graph A.
    Also returns "_W3" = number of length-3 walks (= 3-hop paths for pairs with CN=0).
    """
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    w_aa = np.zeros(n)
    m = deg > 1
    w_aa[m] = 1.0 / np.log(deg[m])
    inv = np.zeros(n)
    m = deg > 0
    inv[m] = 1.0 / deg[m]
    Dinv = sp.diags(inv)

    keys = METHODS + ["_W3"]
    out = {k: np.zeros(len(P)) for k in keys}
    for s0 in range(0, len(P), batch):
        u, v = P[s0:s0 + batch, 0], P[s0:s0 + batch, 1]
        Ru, Rv = A[u], A[v]
        C = Ru.multiply(Rv).tocsr()
        cn = np.asarray(C.sum(axis=1)).ravel()
        union = deg[u] + deg[v] - cn
        sl = slice(s0, s0 + len(u))
        ra = C @ inv
        # 3-hop information: walks u-a-b-v  and  resource-allocation over them
        w3 = np.asarray((Ru @ A).multiply(Rv).sum(axis=1)).ravel()
        ra3 = np.asarray(((Ru @ Dinv @ A).multiply(Rv @ Dinv)).sum(axis=1)).ravel()
        out["CN"][sl] = cn
        out["Jaccard"][sl] = np.divide(cn, union, out=np.zeros_like(cn), where=union > 0)
        out["AA"][sl] = C @ w_aa
        out["RA"][sl] = ra
        out["PA"][sl] = deg[u] * deg[v]
        out["LP"][sl] = cn + LP_EPS * w3
        out["RA+"][sl] = ra + RAPLUS_EPS * ra3 / (1.0 + ra3)
        out["_W3"][sl] = w3
    return out


def feature_matrix(A, P):
    sc = score_pairs(A, P)
    return np.column_stack([sc[k] for k in METHODS])


# ---------------------------------------------------------------- metrics
def evaluate_scores(y, s, rng):
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    k = int(y.sum())
    perm = rng.permutation(len(s))                 # random tie-breaking
    order = np.lexsort((perm, -s))
    return auc, ap, float(y[order[:k]].mean())


# ------------------------------------------------------------ graph stats
def graph_stats(G):
    deg = np.array([d for _, d in G.degree()], dtype=float)
    try:
        assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        assort = float("nan")
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": deg.mean(),
        "max_degree": deg.max(),
        "degree_heterogeneity": (deg ** 2).mean() / deg.mean(),   # <k^2>/<k>
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "assortativity": assort,
    }


# ------------------------------------------------------- graph variants
def sparsify(G, frac, rng):
    from data_loader import clean_graph
    E = list(G.edges())
    idx = rng.choice(len(E), size=int(round(frac * len(E))), replace=False)
    H = nx.Graph([E[i] for i in idx])
    return clean_graph(H)


def kcore_graph(G, k):
    from data_loader import clean_graph
    return clean_graph(nx.k_core(G, k))


# ------------------------------------------------------------ ML extension
def train_ml(E_train, n, rng, seed, val_frac=0.2):
    """
    Train classifiers on the 5 heuristic scores used as features.
    Training data is built ONLY from the training graph: a further `val_frac`
    of training edges is hidden (positives), features are computed on the
    reduced graph; negatives = half random, half distance-2 pairs.
    """
    E_sub, E_val = split_edges(E_train, n, val_frac, rng)
    A_sub = build_adj(E_sub, n)
    train_codes = np.sort(encode(E_train, n))
    n_val = len(E_val)
    hard = sample_hard_negatives(A_sub, n, train_codes, n_val // 2, rng)
    easy = sample_easy_negatives(n, train_codes, n_val - len(hard), rng)
    P = np.vstack([E_val, hard, easy])
    y = np.r_[np.ones(n_val), np.zeros(len(hard) + len(easy))]
    X = feature_matrix(A_sub, P)

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(np.log1p(X), y)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                random_state=seed, n_jobs=1)
    rf.fit(X, y)
    return {"LogReg": lr, "RandForest": rf}


def predict_ml(models, X):
    return {
        "LogReg": models["LogReg"].predict_proba(np.log1p(X))[:, 1],
        "RandForest": models["RandForest"].predict_proba(X)[:, 1],
    }


# ------------------------------------------------------------ one experiment
def run_one(E, n, seed, test_frac=0.2, ml=True):
    rng = np.random.default_rng(seed)
    E_train, E_test = split_edges(E, n, test_frac, rng)
    A_train = build_adj(E_train, n)
    full_codes = np.sort(encode(E, n))
    n_pos = len(E_test)
    models = train_ml(E_train, n, rng, seed) if ml else None

    # hop-distance of every test link in the training graph (coverage analysis)
    sc_pos = score_pairs(A_train, E_test)
    has2 = sc_pos["CN"] >= 1
    has3 = (~has2) & (sc_pos["_W3"] >= 1)
    cov = {"cov2": float(has2.mean()), "cov3": float(has3.mean()),
           "cov4p": float((~has2 & ~has3).mean())}

    settings = {"easy": (E_test, sample_easy_negatives(n, full_codes, n_pos, rng))}

    pos2 = E_test[has2]
    if len(pos2) >= 10:
        neg2 = sample_hard_negatives(A_train, n, full_codes, len(pos2), rng)
        if len(neg2) >= 10:
            k = min(len(pos2), len(neg2))
            settings["hop2"] = (pos2[:k], neg2[:k])

    pos_far = E_test[~has2]
    if len(pos_far) >= 10:
        cand = sample_easy_negatives(n, full_codes, 5 * len(pos_far) + 50, rng)
        cand = cand[cn_count(A_train, cand) == 0][:len(pos_far)]
        if len(cand) >= 10:
            k = min(len(pos_far), len(cand))
            settings["far"] = (pos_far[:k], cand[:k])

    rows = []
    for neg_type, (pos, neg) in settings.items():
        P = np.vstack([pos, neg])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        X = feature_matrix(A_train, P)
        scores = {k_: X[:, i] for i, k_ in enumerate(METHODS)}
        if ml:
            scores.update(predict_ml(models, X))
        for method, sc in scores.items():
            auc, ap, pk = evaluate_scores(y, sc, rng)
            rows.append({"seed": seed, "neg_type": neg_type, "method": method,
                         "auc": auc, "ap": ap, "p_at_k": pk,
                         "n_pos": len(pos), "n_neg": len(neg), **cov})
    importances = dict(zip(METHODS, models["RandForest"].feature_importances_)) if ml else None
    return rows, importances
