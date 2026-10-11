#!/usr/bin/env python3
"""Pick the films and reviews for one language track and write them as Parquet shards.

One streaming pass over the dump. Each review is cleaned and language-checked;
a film qualifies with enough reviews in the target language; a bounded heap
keeps the most-discussed films, so memory stays flat however large the dump is.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from letterboxd_vibes.config import LANGS, load_config, raw_dump, reviews_dir
from letterboxd_vibes.lang_id import LanguageIdentifier
from letterboxd_vibes.text import (
    clean_review,
    is_valid_review,
    parse_likes,
    parse_rating,
    parse_year,
)


def film_reviews(film: dict, lang: str, lid: LanguageIdentifier, cfg: dict) -> list[dict]:
    """The film's usable reviews in `lang`."""
    kept = []
    for idx, review in enumerate(film.get("reviews") or []):
        text = clean_review(review.get("review_text"))
        if not is_valid_review(text, cfg["min_review_chars"]):
            continue
        label, confidence = lid.predict(text)
        if label != lang or confidence < cfg["lang_confidence"]:
            continue
        kept.append(
            {
                "review_idx": idx,
                "review_text": text,
                "likes": parse_likes(review.get("likes")),
                "username": review.get("username") or "",
            }
        )
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=LANGS, required=True)
    parser.add_argument("--input", type=Path, help="JSONL to read instead of the full dump")
    parser.add_argument("--limit", type=int, help="stop after this many input films")
    args = parser.parse_args()

    cfg = load_config(args.lang)
    ex = cfg["extract"]
    source = args.input or raw_dump(cfg)
    lid = LanguageIdentifier()

    # Min-heap of (total likes, tiebreak, film): the least-discussed film is on
    # top and is the one evicted when a more-discussed film arrives.
    heap: list[tuple[float, int, dict]] = []
    seen_urls: set[str] = set()  # the dump repeats some films
    stats = {"films_read": 0, "films_eligible": 0}

    with source.open(encoding="utf-8") as f:
        for line in tqdm(f, desc="films", unit=" films"):
            if args.limit and stats["films_read"] >= args.limit:
                break
            stats["films_read"] += 1
            film = json.loads(line)
            url = film.get("url") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            reviews = film_reviews(film, args.lang, lid, ex)
            if len(reviews) < ex["min_reviews_per_film"]:
                continue
            stats["films_eligible"] += 1

            record = {
                "film_url": url,
                "title": film.get("title") or "",
                "year": parse_year(film.get("year")),
                "genres": [g for g in film.get("genres") or [] if g],
                "directors": [d for d in film.get("directors") or [] if d],
                # kept so the names can be stripped from the reviews before embedding
                "cast": [c for c in film.get("cast") or [] if c],
                "rating": parse_rating(film.get("rating")),
                "poster_url": film.get("poster_url") or "",
                "reviews": reviews,
            }
            entry = (sum(r["likes"] for r in reviews), stats["films_eligible"], record)
            if len(heap) < ex["max_films"]:
                heapq.heappush(heap, entry)
            else:
                heapq.heappushpop(heap, entry)

    # Most-discussed first, so an interrupted embedding run has still covered
    # the films people are most likely to look for.
    films = [record for _, _, record in sorted(heap, key=lambda e: (-e[0], e[1]))]
    rows = [
        {**{k: v for k, v in film.items() if k != "reviews"}, **review}
        for film in films
        for review in film["reviews"]
    ]

    out_dir = reviews_dir(args.lang)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    shard_rows = ex["shard_rows"]
    for n, start in enumerate(range(0, len(rows), shard_rows)):
        pd.DataFrame(rows[start : start + shard_rows]).to_parquet(
            out_dir / f"part-{n:05d}.parquet", index=False
        )

    stats.update(
        lang=args.lang,
        films_kept=len(films),
        reviews_kept=len(rows),
        shards=-(-len(rows) // shard_rows),
    )
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
