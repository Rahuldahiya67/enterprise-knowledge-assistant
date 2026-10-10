"""Small text utilities shared by the offline embedder, BM25 and evaluation."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its of on or our
    the their them there these this to was we what when where which who whom why will with
    you your may must should would could shall any all not no than then so such into about
    per also other each more most only own same too very s t just over under up down out
    get gets got need needs much many""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _stem(token: str) -> str:
    # A tiny suffix stripper: good enough to match "approvals" with "approval".
    for suffix in ("ing", "ies", "es", "s", "ed"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def content_tokens(text: str) -> list[str]:
    """Lower-cased, stop-word-free, lightly stemmed tokens."""
    return [_stem(t) for t in tokenize(text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split into sentences, dropping markdown headings."""
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    parts = re.split(r"(?<=[.!?])\s+|\n+", "\n".join(lines))
    return [p.strip(" -*\t") for p in parts if len(p.strip(" -*\t")) > 3]
