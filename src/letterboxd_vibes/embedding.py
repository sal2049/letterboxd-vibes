"""Local text embeddings through sentence-transformers."""

from __future__ import annotations

import numpy as np
import yaml

from letterboxd_vibes.config import ROOT


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_slug(name: str) -> str:
    """'unsloth/embeddinggemma-2' -> 'unsloth--embeddinggemma-2', safe as a folder name."""
    return name.replace("/", "--")


def known_models() -> dict[str, dict]:
    with (ROOT / "config" / "models.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class Embedder:
    """One embedding model with the input prefix its documentation asks for."""

    def __init__(self, name: str, cfg: dict, device: str | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.batch_size = cfg["batch_size"]
        self.device = device or pick_device()
        prefix = known_models().get(name) or {}
        self._prompt = {k: prefix[k] for k in ("prompt", "prompt_name") if k in prefix}

        kwargs = {}
        if "embeddinggemma-2" in name:
            # The checkpoint also carries vision and audio towers (740M parameters
            # in total); only the 270M text model is needed here.
            kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
        self.model = SentenceTransformer(
            name,
            device=self.device,
            # float32: the EmbeddingGemma model card warns that float16 can produce NaNs.
            model_kwargs={"torch_dtype": torch.float32},
            **kwargs,
        )
        self.model.max_seq_length = cfg["max_tokens"]

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-length vectors, one row per text, in the order given."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # Shortest first, so each batch is padded to a similar length. With
        # reviews running from one line to several paragraphs this is a few
        # times faster than embedding them in file order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vectors = self.model.encode(
            [texts[i] for i in order],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            **self._prompt,
        ).astype(np.float32)
        out = np.empty_like(vectors)
        out[order] = vectors
        return out
