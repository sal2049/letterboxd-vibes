#!/usr/bin/env python3
"""Embed every masked review with the configured model, one output shard per input shard.

A finished shard is never redone, so the run can be stopped and restarted.
Vectors are stored per model, so trying another model never overwrites them.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from letterboxd_vibes.config import LANGS, embeddings_dir, load_config, masked_dir
from letterboxd_vibes.embedding import Embedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=LANGS, required=True)
    parser.add_argument("--max-shards", type=int, help="stop after this many new shards")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: best available")
    args = parser.parse_args()

    cfg = load_config(args.lang)
    model = cfg["embedding"]["model"]
    shards = sorted(masked_dir(args.lang).glob("part-*.parquet"))
    if not shards:
        raise SystemExit(f"No reviews in {masked_dir(args.lang)}. Run 02_mask_reviews.py first.")

    out_dir = embeddings_dir(args.lang, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in shards if not (out_dir / s.name).exists()]
    if args.max_shards:
        todo = todo[: args.max_shards]
    print(f"{len(shards) - len(todo)} of {len(shards)} shards already embedded with {model}")

    if todo:
        embedder = Embedder(model, cfg["embedding"], device=args.device)
        print(f"{embedder.name} on {embedder.device}, {embedder.dim} dimensions")
        for shard in todo:
            started = time.time()
            df = pd.read_parquet(shard)
            df["embedding"] = list(embedder.encode(df["masked_text"].tolist()))
            # Write under a temporary name so a killed run never leaves a
            # half-written shard that looks finished.
            tmp = out_dir / f"{shard.name}.tmp"
            df.to_parquet(tmp, index=False)
            tmp.rename(out_dir / shard.name)
            rate = len(df) / (time.time() - started)
            print(f"{shard.name}: {len(df)} reviews, {rate:.0f}/s", flush=True)
        meta = {
            "lang": args.lang,
            "model": model,
            "dim": embedder.dim,
            "max_tokens": cfg["embedding"]["max_tokens"],
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    done = len(list(out_dir.glob("part-*.parquet")))
    print(f"{done} of {len(shards)} shards embedded")


if __name__ == "__main__":
    main()
