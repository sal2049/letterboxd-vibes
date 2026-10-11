"""Colours that follow similarity: groups near each other on the map look alike."""

from __future__ import annotations

import numpy as np


def _oklch_to_srgb(lightness: float, chroma: float, hue: float) -> np.ndarray:
    """OKLCH -> linear-light sRGB (may fall outside 0..1 when out of gamut)."""
    a, b = chroma * np.cos(hue), chroma * np.sin(hue)
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return np.array(
        [
            4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
            -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
            -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
        ]
    )


def _hex(lightness: float, chroma: float, hue: float) -> str:
    rgb = _oklch_to_srgb(lightness, chroma, hue)
    while (rgb.min() < 0 or rgb.max() > 1) and chroma > 0.01:  # pull into gamut, keeping the hue
        chroma *= 0.95
        rgb = _oklch_to_srgb(lightness, chroma, hue)
    rgb = np.clip(rgb, 0, 1)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return "#" + "".join(f"{round(float(c) * 255):02x}" for c in srgb)


def position_hues(centres: np.ndarray) -> np.ndarray:
    """Angle of each group's map position around the middle of the map, 0..2π."""
    offset = centres - centres.mean(axis=0)
    return np.arctan2(offset[:, 1], offset[:, 0]) % (2 * np.pi)


def similarity_colours(centres: np.ndarray) -> list[str]:
    """One hex colour per group, from where the group sits on the 2-D map.

    Hue is the direction from the middle of the map, so neighbours around the
    map shade into each other. Lightness rises with distance from the middle,
    which separates an inner group from an outer one in the same direction.
    All colours share one chroma and stay light enough to read on the dark map.
    """
    offset = centres - centres.mean(axis=0)
    radius = np.linalg.norm(offset, axis=1)
    reach = radius / (radius.max() + 1e-9)
    return [_hex(0.62 + 0.16 * r, 0.15, h) for h, r in zip(position_hues(centres), reach, strict=True)]
