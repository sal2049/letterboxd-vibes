#!/usr/bin/env python3
"""Download the Letterboxd dump (one 1.1 GB JSONL file) from the Hugging Face Hub.

A local copy beats streaming here: every later step re-reads the file, a pass
over it on disk takes about two minutes, and the download resumes if interrupted.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from letterboxd_vibes.config import load_config, raw_dump


def main() -> None:
    cfg = load_config("en")
    dest = raw_dump(cfg)
    path = hf_hub_download(
        repo_id=cfg["dataset"]["repo"],
        filename=cfg["dataset"]["file"],
        repo_type="dataset",
        local_dir=dest.parent,
    )
    print(f"{path} ({Path(path).stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
