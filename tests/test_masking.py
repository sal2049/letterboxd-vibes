from letterboxd_vibes.masking import build_vocabulary, credit_words, everyday_words, mask, word_counts


def corpus():
    films, texts = [], []
    for film in range(30):
        films += [film, film]
        texts += [
            "A slow and tender film. I think Nolan made it for Disney, and it is tender.",
            "so slow. honestly i loved how tender it felt, nolan or not",
        ]
    films.append(99)
    texts.append("Only this one mentions the zorblax, twice: zorblax.")
    return films, texts


def test_words_usually_capitalised_are_names():
    vocabulary = build_vocabulary(word_counts(*corpus()), min_films=5, capital_share=0.5)
    assert "disney" in vocabulary.capitalised
    assert {"slow", "tender", "loved"} <= vocabulary.keep


def test_a_capital_at_the_start_of_a_sentence_is_not_evidence():
    counts = word_counts([1, 2], ["Slow film. Slow start.", "it is slow"])
    assert counts["slow"] == (2, 1, 0)


def test_the_pronoun_is_kept():
    vocabulary = build_vocabulary(word_counts(*corpus()), min_films=5, capital_share=0.5)
    assert "i" in vocabulary.keep


def test_words_used_for_few_films_are_dropped():
    vocabulary = build_vocabulary(word_counts(*corpus()), min_films=5, capital_share=0.5)
    assert "zorblax" in vocabulary.rare


def test_a_name_written_in_lower_case_half_the_time_needs_the_credits():
    vocabulary = build_vocabulary(word_counts(*corpus()), min_films=5, capital_share=0.5)
    assert "nolan" in vocabulary.keep  # one capital, one lower-case use per film
    credits = credit_words("Tenet", ["Christopher Nolan"])
    assert mask("nolan made it slow", vocabulary, credits) == "made it slow"


def test_mask_leaves_the_rest_as_written():
    vocabulary = build_vocabulary(word_counts(*corpus()), min_films=5, capital_share=0.5)
    assert (
        mask("A slow film, made for Disney. I loved it!", vocabulary) == "A slow film, made for. I loved it!"
    )


def test_short_credit_words_are_ignored():
    assert credit_words("Up", ["Ed Asner", "Bo Li"]) == frozenset({"asner"})


def test_everyday_title_words_stay():
    counts = word_counts(*corpus())
    everyday = everyday_words(counts, n_films=31, share=0.5)
    assert credit_words("A Slow Zorblax Film", [], everyday) == frozenset({"zorblax"})
