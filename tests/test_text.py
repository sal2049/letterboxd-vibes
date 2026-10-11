from letterboxd_vibes.text import (
    clean_review,
    film_slug,
    is_valid_review,
    parse_likes,
    parse_rating,
    parse_year,
)


def test_spoiler_gate_is_removed():
    assert clean_review("This review may contain spoilers.I can handle the truth.") == ""
    assert clean_review("This review may contain spoilers. I can handle the truth. Wow.") == "Wow."


def test_urls_and_whitespace_are_collapsed():
    assert clean_review("see  https://example.com/x \n now") == "see now"


def test_short_or_symbol_heavy_reviews_are_rejected():
    assert not is_valid_review("too short")
    assert not is_valid_review("!!! 10/10 ★★★★★ 😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭")
    assert is_valid_review("It left me hollow for days and I still think about the ending.")


def test_field_parsers():
    assert parse_likes("11,662") == 11662.0
    assert parse_likes(None) == 0.0
    assert parse_likes("n/a") == 0.0
    assert parse_rating("4.62 out of 5") == 4.62
    assert parse_rating(None) is None
    assert parse_year("1985") == 1985
    assert parse_year("") is None
    assert film_slug("https://letterboxd.com/film/come-and-see/") == "come-and-see"
