#!/usr/bin/env python3
"""Compare embedding models on the text the map is built from.

Every model in config/models.yaml embeds the masked reviews of the same sample
of films. With names gone, the question is whether a model still reads the
same film out of different people's reviews, which is what
letterboxd_vibes.reliability measures. The table also reports how much of
that agreement is explained by the dataset's genre labels, and the speed on
this machine.

Writes outputs/benchmark_embeddings.json and .md. Vectors are cached under
data/benchmark/, so a re-run only pays for models it has not seen.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from letterboxd_vibes.clustering import unit_rows
from letterboxd_vibes.config import DATA, OUTPUTS, load_config, masked_dir
from letterboxd_vibes.deconfound import confound_matrix, residualize
from letterboxd_vibes.embedding import Embedder, known_models, model_slug
from letterboxd_vibes.films import aggregate_films
from letterboxd_vibes.reliability import split_half_agreement

MIN_REVIEWS = 6  # so each half of a film has at least three


def sample_reviews(lang: str, n_films: int, seed: int) -> pd.DataFrame:
    reviews = pd.concat(
        [pd.read_parquet(p) for p in sorted(masked_dir(lang).glob("part-*.parquet"))], ignore_index=True
    )
    sizes = reviews.groupby("film_url").size()
    films = sizes[sizes >= MIN_REVIEWS].sample(n_films, random_state=seed).index
    return reviews[reviews["film_url"].isin(films)].reset_index(drop=True)


def genre_free(reviews: pd.DataFrame, vectors: np.ndarray) -> np.ndarray:
    """Review vectors with the part predictable from their film's genres removed."""
    films, _, film_of_review = aggregate_films(reviews.assign(embedding=list(vectors)))
    genres = confound_matrix(films, ["genre"])[film_of_review]
    return unit_rows(residualize(vectors, genres))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument("--films", type=int, default=1500)
    parser.add_argument("--models", nargs="*", help="default: every model in config/models.yaml")
    args = parser.parse_args()
    cfg = load_config(args.lang)
    reviews = sample_reviews(args.lang, args.films, seed=0)
    film_of_review = pd.factorize(reviews["film_url"])[0]
    print(f"{args.films} films, {len(reviews)} masked reviews")

    rows = []
    for name in args.models or list(known_models()):
        path = DATA / "benchmark" / model_slug(name) / f"masked_{args.lang}_{args.films}_{len(reviews)}.npy"
        timing = path.with_suffix(".json")
        if not path.exists():
            embedder = Embedder(name, cfg["embedding"])
            started = time.time()
            vectors = embedder.encode(reviews["masked_text"].tolist())
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
            timing.write_text(json.dumps({"reviews_per_second": len(reviews) / (time.time() - started)}))
            del embedder
        vectors = np.load(path)
        agreement, top10 = split_half_agreement(film_of_review, vectors)
        without_genre, _ = split_half_agreement(film_of_review, genre_free(reviews, vectors))
        rows.append(
            {
                "model": name,
                "dimensions": int(vectors.shape[1]),
                "agreement": round(agreement, 3),
                "top10": round(top10, 3),
                "agreement_without_genre": round(without_genre, 3),
                "reviews_per_second": round(json.loads(timing.read_text())["reviews_per_second"]),
            }
        )
        print(rows[-1], flush=True)

    rows.sort(key=lambda r: -r["agreement_without_genre"])
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "benchmark_embeddings.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Embedding models compared",
        "",
        f"{args.films} films, {len(reviews)} reviews with names removed. Agreement is the split-half",
        "measure in `letterboxd_vibes/reliability.py`: 0.5 is chance, 1.0 is perfect. The models are",
        "ranked on agreement with the dataset's genre labels regressed out of the vectors, since the",
        "map removes genre too.",
        "",
        "| Model | Dimensions | Agreement without genre | Agreement | Own half in top 10 | Reviews / s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['model']} | {r['dimensions']} | {r['agreement_without_genre']:.3f} | {r['agreement']:.3f} "
        f"| {r['top10']:.1%} | {r['reviews_per_second']} |"
        for r in rows
    ]
    (OUTPUTS / "benchmark_embeddings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
