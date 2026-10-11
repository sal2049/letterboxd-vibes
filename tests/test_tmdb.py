from letterboxd_vibes.tmdb import pick

MONSTERS = [
    {
        "id": 1,
        "title": "Monster",
        "original_title": "Monster",
        "release_date": "2003-12-24",
        "poster_path": "/a.jpg",
    },
    {
        "id": 2,
        "title": "Monster",
        "original_title": "怪物",
        "release_date": "2023-06-02",
        "poster_path": "/b.jpg",
    },
    {"id": 3, "title": "Monsters, Inc.", "original_title": "Monsters, Inc.", "release_date": "2001-11-01"},
]


def test_same_title_is_separated_by_year():
    assert pick(MONSTERS, "Monster", 2023, "movie")["id"] == 2
    assert pick(MONSTERS, "Monster", 2004, "movie")["id"] == 1  # festival year vs release year


def test_wrong_year_is_not_a_match():
    assert pick(MONSTERS, "Monster", 1980, "movie") is None


def test_accents_and_tv_results():
    amelie = [
        {"id": 5, "title": "Amélie", "original_title": "Le Fabuleux Destin", "release_date": "2001-04-25"}
    ]
    assert pick(amelie, "Amelie", 2001, "movie")["id"] == 5
    chernobyl = [{"id": 9, "name": "Chernobyl", "original_name": "Chernobyl", "first_air_date": "2019-05-06"}]
    assert pick(chernobyl, "Chernobyl", 2019, "tv")["id"] == 9
