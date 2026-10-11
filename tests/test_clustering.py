import numpy as np
import pytest

from letterboxd_vibes.clustering import (
    ALGORITHMS,
    bootstrap_stability,
    central_members,
    cluster,
    neighbours,
    unit_rows,
)


def three_blobs(per_blob=60, dims=10, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.eye(dims)[:3] * 6
    return unit_rows(np.vstack([c + rng.standard_normal((per_blob, dims)) for c in centres]))


@pytest.mark.parametrize("method", sorted(set(ALGORITHMS) - {"umap-hdbscan", "umap-kmeans"}))
def test_every_algorithm_recovers_three_obvious_blobs(method):
    labels = cluster(three_blobs(), method, k=3)
    assert len(labels) == 180
    for blob in range(3):
        block = labels[blob * 60 : (blob + 1) * 60]
        assert np.bincount(block).max() >= 55  # nearly all of a blob lands in one group


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown cluster_method"):
        cluster(three_blobs(), "magic", k=3)


def test_stability_is_high_for_clear_groups_and_low_for_noise():
    def kmeans(points, run):
        return cluster(points, "kmeans", 3, seed=run)

    noise = unit_rows(np.random.default_rng(1).standard_normal((180, 10)))
    assert bootstrap_stability(kmeans, three_blobs()) > 0.9
    assert bootstrap_stability(kmeans, noise) < 0.5


def test_half_the_neighbours_share_no_genre():
    points = three_blobs()
    genres = np.zeros((len(points), 2))
    genres[::2, 0] = 1  # even rows are genre A, odd rows genre B
    genres[1::2, 1] = 1
    picked = neighbours(points, genres, k=6)
    assert all(len(row) == 6 and i not in row for i, row in enumerate(picked))
    assert sum(genres[j] @ genres[0] == 0 for j in picked[0]) == 3


def test_central_members_are_the_ones_nearest_the_group_mean():
    points = unit_rows(np.array([[1, 0.0], [1, 0.1], [1, -0.1], [1, 2.0], [0, 1.0]]))
    assert set(central_members(points, np.array([0, 1, 2, 3]), n=2)) <= {0, 1, 2}
