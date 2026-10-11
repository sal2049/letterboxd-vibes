from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
WEB_DATA = ROOT / "web" / "public" / "data"
# One track for now. Adding a language means a config/<lang>.yaml and a model that covers it.
LANGS = ("en",)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(lang: str) -> dict[str, Any]:
    """config/base.yaml with config/<lang>.yaml laid over it."""
    with (ROOT / "config" / "base.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with (ROOT / "config" / f"{lang}.yaml").open(encoding="utf-8") as f:
        cfg = _merge(cfg, yaml.safe_load(f) or {})
    cfg["lang"] = lang
    return cfg


def raw_dump(cfg: dict) -> Path:
    return DATA / "raw" / cfg["dataset"]["file"]


def reviews_dir(lang: str) -> Path:
    return DATA / f"reviews_{lang}"


def masked_dir(lang: str) -> Path:
    return DATA / f"masked_{lang}"


def embeddings_dir(lang: str, model: str) -> Path:
    """Vectors are kept per model, so switching models never mixes them."""
    return DATA / f"embeddings_{lang}" / model.replace("/", "--")
