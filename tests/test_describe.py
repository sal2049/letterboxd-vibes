import numpy as np

from letterboxd_vibes.colours import similarity_colours
from letterboxd_vibes.describe import clip_quote, distinctive_words


def test_a_word_repeated_in_one_review_counts_once():
    spam = ["bobby " * 50 + "was there"] + ["a calm and quiet film"] * 30
    other = ["loud fast and thrilling ride"] * 30
    words = distinctive_words({0: spam, 1: other}, max_share=1.0)
    assert "bobby" not in words[0]
    assert "calm" in words[0] and "thrilling" in words[1]


def test_excluded_tokens_never_appear():
    groups = {0: ["kubrick is cold and precise"] * 20, 1: ["warm and funny"] * 20}
    words = distinctive_words(groups, exclude={"kubrick"}, max_share=1.0)
    assert not any("kubrick" in w for w in words[0])
    assert any("cold" in w for w in words[0])


def test_clip_quote_cuts_on_a_word():
    assert clip_quote("short") == "short"
    assert clip_quote("word " * 100, limit=22) == "word word word word…"


def test_neighbouring_groups_get_similar_colours():
    centres = np.array([[1.0, 0.0], [0.98, 0.2], [-1.0, 0.0]])
    colours = similarity_colours(centres)
    rgb = np.array([[int(c[i : i + 2], 16) for i in (1, 3, 5)] for c in colours])
    assert all(len(c) == 7 and c.startswith("#") for c in colours)
    assert np.abs(rgb[0] - rgb[1]).sum() < np.abs(rgb[0] - rgb[2]).sum()
