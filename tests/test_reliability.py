import numpy as np

from letterboxd_vibes.reliability import split_half_agreement


def reviews(signal: float, films: int = 60, per_film: int = 8, dim: int = 16):
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(films, dim))
    film_of_review = np.repeat(np.arange(films), per_film)
    return film_of_review, signal * centres[film_of_review] + rng.normal(size=(films * per_film, dim))


def test_reviews_that_share_a_film_signal_agree():
    agreement, top10 = split_half_agreement(*reviews(signal=2.0))
    assert agreement > 0.95 and top10 > 0.9


def test_pure_noise_scores_chance():
    agreement, _ = split_half_agreement(*reviews(signal=0.0))
    assert 0.4 < agreement < 0.6
