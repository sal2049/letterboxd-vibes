import numpy as np
import pandas as pd

from letterboxd_vibes.films import aggregate_films, genre_matrix


def review(url, vector, likes):
    return {
        "film_url": url,
        "title": url,
        "year": 2000,
        "genres": ["Drama"],
        "directors": ["A", "B", "C"],
        "rating": 3.5,
        "poster_url": "",
        "review_text": "text",
        "likes": likes,
        "embedding": vector,
    }


def test_film_vector_is_the_mean_of_its_reviews():
    reviews = pd.DataFrame(
        [
            review("a", [1.0, 0.0], likes=10),
            review("a", [0.0, 1.0], likes=0),
            review("b", [0.0, 1.0], likes=3),
        ]
    )
    films, vectors, film_of_review = aggregate_films(reviews)
    assert list(films["film_url"]) == ["a", "b"]
    assert list(film_of_review) == [0, 0, 1]
    assert np.allclose(vectors[0], [0.7071, 0.7071], atol=1e-3)
    assert np.allclose(vectors[1], [0.0, 1.0])
    assert films.loc[0, "director"] == "A, B" and films.loc[0, "n_reviews"] == 2


def test_genre_matrix():
    matrix, names = genre_matrix([["Drama", "War"], [], ["War"]])
    assert names == ["Drama", "War"]
    assert matrix.tolist() == [[1, 1], [0, 0], [0, 1]]
