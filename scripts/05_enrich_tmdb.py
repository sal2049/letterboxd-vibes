#!/usr/bin/env python3
"""Look up each film on TMDB and write its poster and id into the web JSON.

Runs at build time so the credential never reaches the browser. Results are
cached in data/tmdb_cache.json, so a re-run only fetches films that are new.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

from letterboxd_vibes.config import LANGS, ROOT, WEB_DATA
from letterboxd_vibes.export import TMDB_CACHE, apply_posters, load_tmdb_cache, write_web_json
from letterboxd_vibes.tmdb import lookup, make_client

SAVE_EVERY = 250


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=LANGS, nargs="*", default=list(LANGS))
    parser.add_argument("--limit", type=int, help="look up at most this many new films")
    parser.add_argument("--retry-misses", action="store_true", help="search again for films not found before")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    payloads = {}
    for lang in args.lang:
        path = WEB_DATA / f"movies_{lang}.json"
        if path.exists():
            payloads[lang] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"Skipping {lang}: run 04_build_map.py --lang {lang} first.")

    cache = load_tmdb_cache()
    films = {p["id"]: (p["title"], p["year"]) for payload in payloads.values() for p in payload["points"]}
    todo = [f for f in films if f not in cache or (args.retry_misses and not cache[f].get("id"))]
    todo = todo[: args.limit]

    if todo:
        client = make_client()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            hits = pool.map(lambda film_id: lookup(client, *films[film_id]), todo)
            for done, (film_id, hit) in enumerate(zip(todo, tqdm(hits, total=len(todo), desc="tmdb")), 1):
                cache[film_id] = hit
                if done % SAVE_EVERY == 0:
                    TMDB_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        TMDB_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    else:
        print("Nothing new to look up.")

    for lang, payload in payloads.items():
        apply_posters(payload, cache)
        write_web_json(payload, lang)
        matched = sum(1 for p in payload["points"] if p["tmdb"])
        print(f"movies_{lang}.json: {matched} of {len(payload['points'])} films matched on TMDB")


if __name__ == "__main__":
    main()
