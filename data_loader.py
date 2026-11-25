"""
data_loader.py
--------------
Datasets for the link-prediction study.

  * built-in   : karate, lesmis            (come with networkx, no download)
  * synthetic  : syn-er-*, syn-ba-*, syn-ws-*, syn-plc  (generated, fixed seed)
  * SNAP       : facebook, email-eu-core, ca-grqc, ca-hepth, wiki-vote
                 (auto-downloaded to ./data on first use)
  * custom     : "file:path/to/edgelist.txt"  (any whitespace separated edge list)

Every graph is returned as a simple, undirected, unweighted, connected
(largest connected component) networkx Graph with nodes relabelled 0..n-1.
"""
import gzip
import os
import urllib.request

import networkx as nx

SNAP_URLS = {
    "facebook": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "email-eu-core": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
    "ca-grqc": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "ca-hepth": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
    "wiki-vote": "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
}

BUILTIN = ["karate", "lesmis"]
SYNTHETIC = [
    "syn-er-sparse", "syn-er-dense",
    "syn-ba-sparse", "syn-ba-dense",
    "syn-ws-sparse", "syn-ws-dense",
    "syn-plc",
]
SNAP = list(SNAP_URLS.keys())


def clean_graph(G):
    """Simple undirected graph -> largest connected component -> labels 0..n-1."""
    H = nx.Graph()
    H.add_edges_from((u, v) for u, v in G.edges() if u != v)
    if H.number_of_nodes() == 0:
        raise ValueError("Empty graph after cleaning")
    lcc = max(nx.connected_components(H), key=len)
    H = H.subgraph(lcc)
    H = nx.convert_node_labels_to_integers(H, ordering="sorted")
    return nx.Graph(list(H.edges()))


def _read_edgelist(path):
    opener = gzip.open if path.endswith(".gz") else open
    G = nx.Graph()
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "%")):
                continue
            parts = line.split()
            G.add_edge(parts[0], parts[1])
    return G


def _download(name, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    url = SNAP_URLS[name]
    path = os.path.join(data_dir, os.path.basename(url))
    if not os.path.exists(path):
        print(f"[data] downloading {name} from {url}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Could not download {name}: {e}\n"
                f"Download it manually from {url} and put the file in '{data_dir}/'."
            )
    return path


def _synthetic(name, seed=42):
    n = 2000
    if name == "syn-er-sparse":
        return nx.gnp_random_graph(n, 6 / (n - 1), seed=seed)
    if name == "syn-er-dense":
        return nx.gnp_random_graph(n, 40 / (n - 1), seed=seed)
    if name == "syn-ba-sparse":
        return nx.barabasi_albert_graph(n, 2, seed=seed)
    if name == "syn-ba-dense":
        return nx.barabasi_albert_graph(n, 15, seed=seed)
    if name == "syn-ws-sparse":
        return nx.watts_strogatz_graph(n, 6, 0.1, seed=seed)
    if name == "syn-ws-dense":
        return nx.watts_strogatz_graph(n, 40, 0.1, seed=seed)
    if name == "syn-plc":
        return nx.powerlaw_cluster_graph(n, 5, 0.5, seed=seed)
    raise KeyError(name)


def get_graph(name, data_dir="data"):
    if name == "karate":
        G = nx.karate_club_graph()
    elif name == "lesmis":
        G = nx.les_miserables_graph()
    elif name in SYNTHETIC:
        G = _synthetic(name)
    elif name in SNAP_URLS:
        G = _read_edgelist(_download(name, data_dir))
    elif name.startswith("file:"):
        G = _read_edgelist(name[5:])
    else:
        raise KeyError(f"Unknown dataset '{name}'. Options: {BUILTIN + SYNTHETIC + SNAP}")
    return clean_graph(G)
