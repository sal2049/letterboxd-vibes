"""Give each group a theme label.

An embedding model cannot name anything; it only turns text into numbers. So a
group's label comes from a second, generative model. It is shown what the
pipeline measured about the group (the words its reviews over-use, the films
nearest its centre, a few of its reviews) and asked for one short phrase in
the style of the "Themes" on a Letterboxd film page: the mood paired with what
the stories are about. The prompt below is the same for every group and is the
only place where a person tells a model what to write. The style examples in
it are Letterboxd's own theme labels.

A label is a reading of the evidence, not a measurement. The UI and the group
sheet always show the evidence next to it.

Without a naming model the label falls back to the group's most distinctive
words.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from letterboxd_vibes.config import DATA

CACHE = DATA / "names_cache.json"

PROMPT = """\
An algorithm grouped films whose reviews are written in a similar way. Here is one group.

Words its reviews use more than other reviews do:
{words}

Films nearest the centre of the group:
{films}

Reviews of films in the group:
{reviews}

Write a theme label for this group, in the style of the "Themes" on a Letterboxd film page: one \
short phrase that pairs the mood with what the stories are about. Examples of the style:
- Moving relationship stories
- Humanity and the world around us
- Brutal, violent prison drama
- Powerful stories of heartbreak and suffering
- Heartfelt and sentimental family stories
- Gripping, intense violent crime
- Surreal and thought-provoking visions of life and death
Three to nine words, in plain words a film fan would use: say stories, drama, comedy, romance, \
horror, not explorations, narratives, experiences or analysis. Describe this group from the \
evidence above; do not reuse an example. No names of people or countries and no film titles.
{taken}
Answer with JSON only: {{"name": "..."}}"""

MIN_WORDS, MAX_WORDS = 2, 10


def fallback(words: list[str]) -> dict:
    """A label made of the group's own most distinctive words."""
    return {"name": ", ".join(words[:3]) or "unnamed", "by": "words"}


def build_prompt(evidence: dict, taken: list[str]) -> str:
    return PROMPT.format(
        words=", ".join(evidence["words"]),
        films="; ".join(evidence["films"]),
        reviews="\n".join(f"- {r}" for r in evidence["reviews"]),
        taken=f"Names already used for other groups, do not repeat them: {', '.join(taken)}.\n"
        if taken
        else "",
    )


def parse_answer(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        answer = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    name = str(answer.get("name", "")).strip().rstrip(".")
    if not MIN_WORDS <= len(name.split()) <= MAX_WORDS:
        return None
    return {"name": name, "by": "model"}


class Namer:
    """A local chat model, loaded once. Answers are cached by prompt so a rebuild is free and repeatable."""

    def __init__(self, model: str, cache: Path = CACHE) -> None:
        from mlx_lm import load

        self.name = model
        self.model, self.tokenizer = load(model)
        self.cache_path = cache
        self.cache = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}

    def ask(self, prompt: str) -> str:
        from mlx_lm import generate

        key = hashlib.sha256(f"{self.name}\n{prompt}".encode()).hexdigest()
        if key not in self.cache:
            chat = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
            )
            # mlx_lm samples greedily unless told otherwise, so the same prompt gives the same name.
            self.cache[key] = generate(self.model, self.tokenizer, prompt=chat, max_tokens=60)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False), encoding="utf-8")
        return self.cache[key]


def name_groups(evidence: list[dict], model: str | None) -> list[dict]:
    """A {"name", "by"} per group, in the order given."""
    if not model:
        return [fallback(e["words"]) for e in evidence]
    namer = Namer(model)
    names: list[dict] = []
    for group in evidence:
        taken = [n["name"] for n in names if n["by"] == "model"]
        prompt = build_prompt(group, taken)
        answer = parse_answer(namer.ask(prompt))
        if answer and answer["name"].lower() in {t.lower() for t in taken}:
            # The model repeated a label despite being told; ask once more, then give up on it.
            retry = f'{prompt}\n\n"{answer["name"]}" is already used. Write a different label.'
            answer = parse_answer(namer.ask(retry))
            if answer and answer["name"].lower() in {t.lower() for t in taken}:
                answer = None
        names.append(answer or fallback(group["words"]))
    return names
