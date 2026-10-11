#!/usr/bin/env python3
"""Strip names out of the reviews, so the embedding model reads how a film is talked about.

Reads the extracted reviews, works out which words to remove (see
letterboxd_vibes.masking for the three rules), and writes the same shards
with a `masked_text` column. outputs/vocabulary_<lang>.json lists what was
removed, so the rules can be checked by reading it.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from letterboxd_vibes.config import LANGS, OUTPUTS, load_config, masked_dir, reviews_dir
from letterboxd_vibes.masking import (
    build_vocabulary,
    credit_words,
    everyday_words,
    mask,
    word_count,
    word_counts,
)

EXAMPLES = 150  # removed words listed per rule, most widely used first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=LANGS, required=True)
    args = parser.parse_args()
    cfg = load_config(args.lang)
    mcfg = cfg["masking"]

    shards = sorted(reviews_dir(args.lang).glob("part-*.parquet"))
    if not shards:
        raise SystemExit(f"No reviews in {reviews_dir(args.lang)}. Run 01_extract_reviews.py first.")
    reviews = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)

    counts = word_counts(reviews["film_url"], reviews["review_text"])
    vocabulary = build_vocabulary(counts, mcfg["min_films"], mcfg["capital_share"])

    first = reviews.groupby("film_url", sort=False).first()
    everyday = everyday_words(counts, len(first), mcfg["everyday_share"])
    credits = {
        url: credit_words(row.title, [*row.directors, *list(row.cast)[: mcfg["top_billed"]]], everyday)
        for url, row in first.iterrows()
    }
    reviews["masked_text"] = [
        mask(text, vocabulary, credits[url])
        for url, text in zip(reviews["film_url"], reviews["review_text"], strict=True)
    ]

    # A review with almost nothing left says nothing; a film needs a few reviews to be placed.
    reviews = reviews[reviews["masked_text"].map(word_count) >= mcfg["min_words"]]
    per_film = reviews.groupby("film_url")["masked_text"].transform("size")
    reviews = reviews[per_film >= cfg["extract"]["min_reviews_per_film"]].reset_index(drop=True)
    words_before = int(reviews["review_text"].map(word_count).sum())
    words_after = int(reviews["masked_text"].map(word_count).sum())

    out_dir = masked_dir(args.lang)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    shard_rows = cfg["extract"]["shard_rows"]
    for n, start in enumerate(range(0, len(reviews), shard_rows)):
        reviews.iloc[start : start + shard_rows].to_parquet(out_dir / f"part-{n:05d}.parquet", index=False)

    def most_used(words: frozenset[str]) -> list[str]:
        return sorted(words, key=lambda w: -counts[w][0])[:EXAMPLES]

    report = {
        "lang": args.lang,
        "films": int(reviews["film_url"].nunique()),
        "reviews": len(reviews),
        "words_kept_share": round(words_after / words_before, 3),
        "rules": mcfg,
        **vocabulary.summary(
            {"capitalised": most_used(vocabulary.capitalised), "rare": most_used(vocabulary.rare)}
        ),
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / f"vocabulary_{args.lang}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
