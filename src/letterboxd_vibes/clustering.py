"""Clustering algorithms behind one interface, plus the layout and neighbour search."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)


def _umap(points: np.ndarray, dims: int, seed: int, neighbors: int = 30, min_dist: float = 0.0) -> np.ndarray:
    import umap

    return umap.UMAP(
        n_components=dims,
        n_neighbors=min(neighbors, len(points) - 1),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    ).fit_transform(points)


def _hdbscan(points: np.ndarray, k: int, seed: int) -> np.ndarray:
    import hdbscan

    # HDBSCAN has no "number of groups" setting; the smallest group it may form
    # is sized so that k groups could cover about half the films.
    reduced = _umap(points, 10, seed)
    return hdbscan.HDBSCAN(min_cluster_size=max(5, len(points) // (2 * k)), min_samples=5).fit_predict(
        reduced
    )


def _leiden(points: np.ndarray, k: int, seed: int, neighbors: int = 15) -> np.ndarray:
    """Leiden community detection on the nearest-neighbour graph of the films.

    Leiden takes a resolution, not a number of groups, so the resolution is
    searched until it gives about `k` communities. That keeps it comparable
    with the other methods at the same group count.
    """
    import igraph
    import leidenalg

    neighbors = min(neighbors, len(points) - 1)
    distance, nearest = (
        NearestNeighbors(n_neighbors=neighbors + 1, metric="cosine").fit(points).kneighbors(points)
    )
    source = np.repeat(np.arange(len(points)), neighbors)
    graph = igraph.Graph(n=len(points), edges=np.column_stack([source, nearest[:, 1:].ravel()]).tolist())
    graph.es["weight"] = (1 - distance[:, 1:].ravel()).clip(min=1e-6).tolist()
    graph = graph.simplify(combine_edges="max")

    low, high = 0.05, 50.0
    labels = np.zeros(len(points), dtype=int)
    for _ in range(14):
        resolution = float(np.sqrt(low * high))
        labels = np.array(
            leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                seed=seed,
            ).membership
        )
        found = labels.max() + 1
        if found == k:
            break
        low, high = (resolution, high) if found < k else (low, resolution)
    return labels


# name -> f(points, k, seed) -> a label per point (-1 = unassigned)
ALGORITHMS: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
    "kmeans": lambda x, k, seed: KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x),
    "ward": lambda x, k, seed: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x),
    "average-linkage": lambda x, k, seed: AgglomerativeClustering(
        n_clusters=k, linkage="average", metric="cosine"
    ).fit_predict(x),
    "gaussian-mixture": lambda x, k, seed: GaussianMixture(
        n_components=k, covariance_type="diag", random_state=seed
    ).fit_predict(x),
    "birch": lambda x, k, seed: Birch(n_clusters=k, threshold=0.3).fit_predict(x),
    "spectral": lambda x, k, seed: SpectralClustering(
        n_clusters=k, affinity="nearest_neighbors", n_neighbors=30, assign_labels="kmeans", random_state=seed
    ).fit_predict(x),
    "umap-kmeans": lambda x, k, seed: KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(
        _umap(x, 10, seed)
    ),
    "umap-hdbscan": _hdbscan,
    "leiden": _leiden,
}


def cluster(points: np.ndarray, method: str, k: int, seed: int = 42) -> np.ndarray:
    """A group id per row of `points` (unit vectors), or -1 where the method leaves a row out."""
    if method not in ALGORITHMS:
        raise ValueError(f"Unknown cluster_method {method!r}; choose from {sorted(ALGORITHMS)}")
    return np.asarray(ALGORITHMS[method](points, k, seed))


def bootstrap_stability(
    assign, points: np.ndarray, reps: int = 4, share: float = 0.8, seed: int = 0
) -> float:
    """How repeatable a clustering is: mean adjusted Rand index between runs on
    different random `share` samples of the points, compared on the points
    they have in common. 1.0 means the same groups every time, 0 means chance.

    `assign(points, run)` returns a label per point.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    runs = []
    for run in range(reps):
        idx = np.sort(rng.choice(n, int(share * n), replace=False))
        runs.append((idx, np.asarray(assign(points[idx], run))))
    scores = []
    for a in range(reps):
        for b in range(a + 1, reps):
            _, ia, ib = np.intersect1d(runs[a][0], runs[b][0], return_indices=True)
            scores.append(adjusted_rand_score(runs[a][1][ia], runs[b][1][ib]))
    return float(np.mean(scores))


def layout_2d(points: np.ndarray, cfg: dict) -> np.ndarray:
    """Map coordinates. Display only: nothing is clustered in 2-D."""
    return _umap(points, 2, seed=42, neighbors=cfg["umap_n_neighbors"], min_dist=cfg["umap_min_dist"])


def neighbours(points: np.ndarray, genres: np.ndarray, k: int = 6, pool: int = 30) -> list[list[int]]:
    """For each film, k of its nearest films.

    Half are simply the nearest. The other half are the nearest that share no
    genre with the film, taken from its `pool` closest: the neighbours a genre
    shelf would never put next to it, which is the point of the map.
    """
    pool = min(pool + 1, len(points))
    _, nearest = NearestNeighbors(n_neighbors=pool, metric="cosine").fit(points).kneighbors(points)

    out = []
    for i, row in enumerate(nearest):
        row = row[row != i]
        unrelated = [j for j in row if genres[j] @ genres[i] == 0][: k // 2]
        closest = [j for j in row if j not in unrelated][: k - len(unrelated)]
        out.append([int(j) for j in closest + unrelated])
    return out


def central_members(points: np.ndarray, members: np.ndarray, n: int = 14) -> list[int]:
    """The `n` members nearest the group's mean direction."""
    centre = points[members].mean(axis=0)
    return [int(i) for i in members[np.argsort(-(points[members] @ centre))[:n]]]
