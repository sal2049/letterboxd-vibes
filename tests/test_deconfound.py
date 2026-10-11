import numpy as np
import pandas as pd

from letterboxd_vibes.deconfound import confound_matrix, residualize


def test_residual_no_longer_tracks_the_confound():
    rng = np.random.default_rng(0)
    year = rng.uniform(1950, 2020, 400)
    vectors = rng.standard_normal((400, 8))
    vectors[:, 0] += (year - 1985) / 10  # one dimension drifts with release year
    films = pd.DataFrame({"year": year})

    confounds = confound_matrix(films, ["year"])
    before = abs(np.corrcoef(vectors[:, 0], year)[0, 1])
    after = abs(np.corrcoef(residualize(vectors, confounds)[:, 0], year)[0, 1])
    assert before > 0.7 and after < 0.1


def test_no_signals_leaves_vectors_alone():
    vectors = np.eye(3)
    assert residualize(vectors, confound_matrix(pd.DataFrame({"year": [1, 2, 3]}), [])) is vectors
