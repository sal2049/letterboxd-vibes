from letterboxd_vibes.naming import build_prompt, fallback, parse_answer

EVIDENCE = {"words": ["twist", "confusing"], "films": ["Tenet (2020)"], "reviews": ["what did i just watch"]}


def test_fallback_uses_the_groups_own_words():
    name = fallback(["twist", "confusing", "rewatch", "ending"])
    assert name == {"name": "twist, confusing, rewatch", "by": "words"}


def test_prompt_carries_the_evidence_and_the_labels_already_used():
    prompt = build_prompt(EVIDENCE, taken=["Slow-burning dread"])
    assert "twist, confusing" in prompt and "Tenet (2020)" in prompt and "what did i just watch" in prompt
    assert "do not repeat them: Slow-burning dread" in prompt
    assert "already used" not in build_prompt(EVIDENCE, taken=[])


def test_answer_is_read_out_of_surrounding_text():
    answer = parse_answer('```json\n{"name": "Mind-bending puzzles that reward a rewatch."}\n```')
    assert answer == {"name": "Mind-bending puzzles that reward a rewatch", "by": "model"}


def test_a_bad_answer_is_rejected():
    assert parse_answer("no json here") is None
    assert parse_answer('{"name": ""}') is None
    assert parse_answer('{"name": "Thrillers"}') is None
    assert parse_answer('{"name": "' + "word " * 12 + '"}') is None
