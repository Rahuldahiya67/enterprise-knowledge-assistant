"""Answer generators.

All generators take the question plus the retrieved chunks and yield the answer
as a stream of text pieces, so the API can forward tokens as they arrive.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Protocol

from .models import ScoredChunk
from .retrieval import LexicalReranker
from .text import split_sentences

NO_ANSWER = "I couldn't find this in the available documents."

SYSTEM_PROMPT = f"""You are an internal knowledge assistant for company employees.
Answer the question using ONLY the numbered sources provided.

Rules:
- Cite every factual sentence with the source number in square brackets, e.g. [2]. Use several when needed: [1][3].
- If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"
- Do not use outside knowledge, do not guess numbers, dates or names.
- Be concise: 1-4 sentences, or a short list when the answer is a procedure.
"""


def format_sources(chunks: list[ScoredChunk]) -> str:
    blocks = []
    for i, sc in enumerate(chunks, start=1):
        c = sc.chunk
        header = f"[{i}] {c.title}" + (f" ({c.department})" if c.department else "")
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[ScoredChunk]) -> str:
    return f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"


class Generator(Protocol):
    name: str

    def stream(self, question: str, chunks: list[ScoredChunk]) -> Iterator[str]: ...

    def complete(self, system: str, user: str) -> str: ...


class OfflineExtractiveGenerator:
    """No-API-key answerer. Picks the source sentences that best match the question
    and returns them with citations. It cannot synthesise or paraphrase, but it is
    grounded by construction, which makes it a useful baseline for evaluation."""

    name = "offline-extractive"

    def __init__(self, max_sentences: int = 3, min_sentence_score: float = 0.30):
        self.max_sentences = max_sentences
        self.min_sentence_score = min_sentence_score
        self.scorer = LexicalReranker()

    def _answer(self, question: str, chunks: list[ScoredChunk]) -> str:
        # Candidate units are single sentences and pairs of adjacent sentences, so a
        # sentence like "It is Rs 2,500 per day" is scored together with the sentence
        # that tells us what "it" is.
        units: list[tuple[int, tuple[int, ...], str]] = []     # (source no., sentence ids, text)
        for src, sc in enumerate(chunks, start=1):
            sentences = [s for s in split_sentences(sc.chunk.text) if len(s.split()) >= 4]
            for j, s in enumerate(sentences):
                units.append((src, (j,), s))
                if j + 1 < len(sentences):
                    units.append((src, (j, j + 1), f"{s} {sentences[j + 1]}"))
        if not units:
            return NO_ANSWER

        raw = self.scorer.score(question, [u[2] for u in units])
        scored = []
        for (src, ids, text), score in zip(units, raw):
            score *= 1.0 - 0.05 * (src - 1)          # prefer higher-ranked chunks
            if len(ids) > 1:
                score *= 0.97                        # prefer the shorter unit on ties
            scored.append((score, src, ids, text))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Decline unless at least one *single* sentence matches the question well.
        # (Pairs pool the words of two sentences, which would let unrelated questions through.)
        best_single = max((score for score, _, ids, _ in scored if len(ids) == 1), default=0.0)
        if best_single < self.min_sentence_score:
            return NO_ANSWER
        best = scored[0][0]
        floor = max(self.min_sentence_score, 0.6 * best)

        picked, used, seen_text, n_sentences = [], set(), set(), 0
        for score, src, ids, text in scored:
            if score < floor or n_sentences >= self.max_sentences:
                break
            keys = {(src, i) for i in ids}
            norm = " ".join(text.lower().split())
            # skip units already used, or repeated across chunks because of chunk overlap
            if keys & used or norm in seen_text or any(norm in t or t in norm for t in seen_text):
                continue
            used |= keys
            seen_text.add(norm)
            n_sentences += len(ids)
            picked.append((text.rstrip(), src))

        parts = []
        for text, src in picked:
            if not re.search(r"[.!?:]$", text):
                text += "."
            parts.append(f"{text} [{src}]")
        return " ".join(parts)

    def stream(self, question: str, chunks: list[ScoredChunk]) -> Iterator[str]:
        text = self._answer(question, chunks)
        for word in re.findall(r"\S+\s*", text):
            yield word

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - not used offline
        raise NotImplementedError("The offline generator only answers from retrieved chunks.")


class OpenAIGenerator:
    def __init__(self, api_key: str | None, model: str):
        from openai import OpenAI

        if not api_key:
            raise ValueError("EKA_OPENAI_API_KEY is required for the openai provider")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai:{model}"

    def stream(self, question: str, chunks: list[ScoredChunk]) -> Iterator[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, chunks)},
            ],
        )
        for event in resp:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield delta

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


class AnthropicGenerator:
    def __init__(self, api_key: str | None, model: str):
        from anthropic import Anthropic

        if not api_key:
            raise ValueError("EKA_ANTHROPIC_API_KEY is required for the anthropic provider")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.name = f"anthropic:{model}"

    def stream(self, question: str, chunks: list[ScoredChunk]) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=800,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(question, chunks)}],
        ) as stream:
            yield from stream.text_stream

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


def build_generator(settings) -> Generator:
    if settings.llm_provider == "openai":
        return OpenAIGenerator(settings.openai_api_key, settings.openai_chat_model)
    if settings.llm_provider == "anthropic":
        return AnthropicGenerator(settings.anthropic_api_key, settings.anthropic_model)
    return OfflineExtractiveGenerator()
