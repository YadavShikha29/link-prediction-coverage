"""Sanity tests: run `python test_core.py` before trusting any result."""
import numpy as np
import networkx as nx
from data_loader import get_graph
from lp_core import (CLASSIC, METHODS, edge_array, encode, build_adj, split_edges,
                     sample_easy_negatives, sample_hard_negatives, score_pairs)


def test_scores_match_networkx():
    G = get_graph("syn-plc")
    n = G.number_of_nodes()
    A = build_adj(edge_array(G), n)
    rng = np.random.default_rng(0)
    P = np.array([(u, v) for u, v in (rng.choice(n, 2, replace=False) for _ in range(500))])
    P = np.sort(P, axis=1)
    ours = score_pairs(A, P)
    pairs = [tuple(p) for p in P]
    ref = {
        "Jaccard": [s for _, _, s in nx.jaccard_coefficient(G, pairs)],
        "AA": [s for _, _, s in nx.adamic_adar_index(G, pairs)],
        "RA": [s for _, _, s in nx.resource_allocation_index(G, pairs)],
        "PA": [s for _, _, s in nx.preferential_attachment(G, pairs)],
        "CN": [len(list(nx.common_neighbors(G, u, v))) for u, v in pairs],
    }
    for k in CLASSIC:
        assert np.allclose(ours[k], ref[k]), f"{k} mismatch vs networkx"
    print("OK  scores identical to networkx (CN, Jaccard, AA, RA, PA)")


def test_split_no_leakage():
    G = get_graph("syn-ws-sparse")
    n = G.number_of_nodes()
    E = edge_array(G)
    rng = np.random.default_rng(1)
    E_tr, E_te = split_edges(E, n, 0.2, rng)
    assert len(E_tr) + len(E_te) == len(E)
    assert not np.isin(encode(E_te, n), encode(E_tr, n)).any()
    Gtr = nx.Graph(list(map(tuple, E_tr)))
    assert Gtr.number_of_nodes() == n and nx.is_connected(Gtr)
    print("OK  split: disjoint, training graph still connected, no isolated nodes")


def test_negatives_valid():
    G = get_graph("syn-plc")
    n = G.number_of_nodes()
    E = edge_array(G)
    rng = np.random.default_rng(2)
    E_tr, E_te = split_edges(E, n, 0.2, rng)
    A = build_adj(E_tr, n)
    full = np.sort(encode(E, n))
    easy = sample_easy_negatives(n, full, len(E_te), rng)
    hard = sample_hard_negatives(A, n, full, len(E_te), rng)
    for neg in (easy, hard):
        assert (neg[:, 0] < neg[:, 1]).all()
        assert not np.isin(encode(neg, n), full).any(), "negative is a real edge!"
    assert (score_pairs(A, hard)["CN"] >= 1).all(), "hard negatives must have a common neighbour"
    print("OK  negatives: never real edges; hard negatives are at distance 2")


def test_three_hop_scores():
    G = get_graph("lesmis")
    n = G.number_of_nodes()
    A = build_adj(edge_array(G), n)
    Ad = A.toarray()
    deg = Ad.sum(1)
    Dinv = np.diag(1.0 / deg)
    W3 = Ad @ Ad @ Ad
    RA3 = Ad @ Dinv @ Ad @ Dinv @ Ad
    iu = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    sc = score_pairs(A, iu)
    assert np.allclose(sc["_W3"], W3[iu[:, 0], iu[:, 1]]), "W3 mismatch"
    cn = sc["CN"]
    ra = sc["RA"]
    ra3 = RA3[iu[:, 0], iu[:, 1]]
    assert np.allclose(sc["RA+"], ra + 1e-6 * ra3 / (1 + ra3))
    assert np.allclose(sc["LP"], cn + 1e-3 * W3[iu[:, 0], iu[:, 1]])
    print("OK  3-hop scores (LP, RA+) match dense matrix formulas")


if __name__ == "__main__":
    test_three_hop_scores()
    test_scores_match_networkx()
    test_split_no_leakage()
    test_negatives_valid()
    print("\nAll tests passed.")
