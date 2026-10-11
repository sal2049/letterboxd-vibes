from __future__ import annotations

import urllib.request
from pathlib import Path

from letterboxd_vibes.config import ROOT

# The compressed fastText language-ID model (917 KB, 176 languages).
LID_PATH = ROOT / "models" / "lid.176.ftz"
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


def ensure_lid_model() -> Path:
    if not LID_PATH.exists():
        LID_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading fastText language ID model to {LID_PATH}")
        urllib.request.urlretrieve(LID_URL, LID_PATH)
    return LID_PATH


class LanguageIdentifier:
    def __init__(self) -> None:
        import fasttext

        fasttext.FastText.eprint = lambda x: None  # silence the load-time warning
        self._model = fasttext.load_model(str(ensure_lid_model()))

    def predict(self, text: str) -> tuple[str, float]:
        """Most likely language code and its probability."""
        # Call the binding directly: fasttext's Python wrapper builds its result
        # with np.array(copy=False), which raises on NumPy 2.
        pairs = self._model.f.predict(text.replace("\n", " ") + "\n", 1, 0.0, "strict")
        if not pairs:
            return "unknown", 0.0
        prob, label = pairs[0]
        return label.replace("__label__", ""), float(prob)
