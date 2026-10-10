"""Evaluation suite.

Each test case in a JSONL file looks like:

    {"id": "hr-01", "question": "...", "expected_sources": ["leave_policy.md"],
     "key_facts": ["18 days"], "answerable": true, "department": null}

Metrics
-------
Retrieval
  hit_rate@k     share of answerable questions where any expected source is in the top-k
  recall@k       share of expected sources found in the top-k (averaged)
  mrr            mean reciprocal rank of the first relevant chunk
Generation
  answer_accuracy   share of key facts present in the answer (answerable questions)
  faithfulness      share of answer sentences supported by the retrieved context
  hallucination_rate share of answers containing at least one unsupported sentence
  refusal_accuracy  unanswerable questions correctly declined, answerable ones not declined

Faithfulness uses an LLM judge when a real provider is configured, and a
token-overlap check otherwise.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .llm import NO_ANSWER
from .models import Answer, SearchFilters
from .pipeline import RAGPipeline
from .text import content_tokens, split_sentences

_CITE = re.compile(r"\s*\[\d+\]")


@dataclass
class EvalCase:
    id: str
    question: str
    expected_sources: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    answerable: bool = True
    department: str | None = None
    doc_type: str | None = None


@dataclass
class CaseResult:
    id: str
    question: str
    answerable: bool
    answer: str
    grounded: bool
    retrieved_sources: list[str]
    hit: bool | None
    recall: float | None
    reciprocal_rank: float | None
    fact_accuracy: float | None
    faithfulness: float | None
    unsupported_sentences: list[str]
    refusal_correct: bool


def load_cases(path: str | Path) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(EvalCase(**json.loads(line)))
    return cases


# ---------------------------------------------------------------- faithfulness judges
def _answer_sentences(answer: str) -> list[str]:
    return [_CITE.sub("", s).strip() for s in split_sentences(answer) if _CITE.sub("", s).strip()]


def overlap_judge(sentences: list[str], contexts: list[str], threshold: float = 0.7) -> list[bool]:
    """A sentence is supported if >= `threshold` of its content words occur in one context chunk."""
    ctx_tokens = [set(content_tokens(c)) for c in contexts]
    verdicts = []
    for s in sentences:
        toks = set(content_tokens(s))
        if not toks:
            verdicts.append(True)
            continue
        best = max((len(toks & c) / len(toks) for c in ctx_tokens), default=0.0)
        verdicts.append(best >= threshold)
    return verdicts


JUDGE_PROMPT = """You check whether each sentence of an answer is supported by the context.
Return ONLY a JSON list of booleans, one per sentence, in order. true = fully supported by the context."""


def llm_judge(generator, sentences: list[str], contexts: list[str]) -> list[bool]:
    user = "Context:\n" + "\n---\n".join(contexts) + "\n\nSentences:\n" + "\n".join(
        f"{i}. {s}" for i, s in enumerate(sentences, start=1)
    )
    raw = generator.complete(JUDGE_PROMPT, user)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    try:
        verdicts = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        verdicts = []
    if len(verdicts) != len(sentences):  # judge misbehaved: fall back to the overlap check
        return overlap_judge(sentences, contexts)
    return [bool(v) for v in verdicts]


# ---------------------------------------------------------------- evaluation
def _fact_present(fact: str, answer: str) -> bool:
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
    return norm(fact) in norm(answer)


def evaluate_case(rag: RAGPipeline, case: EvalCase, use_llm_judge: bool) -> CaseResult:
    filters = SearchFilters(department=case.department, doc_type=case.doc_type)
    ans: Answer = rag.ask(case.question, filters)
    sources = [sc.chunk.source for sc in ans.retrieved]

    hit = recall = rr = facts = None
    if case.answerable and case.expected_sources:
        expected = set(case.expected_sources)
        hit = bool(expected & set(sources))
        recall = len(expected & set(sources)) / len(expected)
        rr = next((1.0 / i for i, s in enumerate(sources, start=1) if s in expected), 0.0)
    if case.answerable and case.key_facts:
        facts = sum(_fact_present(f, ans.answer) for f in case.key_facts) / len(case.key_facts)

    faith, unsupported = None, []
    if ans.grounded:
        sentences = _answer_sentences(ans.answer)
        contexts = [sc.chunk.text for sc in ans.retrieved]
        verdicts = llm_judge(rag.generator, sentences, contexts) if use_llm_judge else overlap_judge(sentences, contexts)
        faith = sum(verdicts) / len(verdicts) if verdicts else 1.0
        unsupported = [s for s, ok in zip(sentences, verdicts) if not ok]

    declined = (not ans.grounded) or NO_ANSWER in ans.answer
    refusal_correct = declined if not case.answerable else not declined

    return CaseResult(
        id=case.id,
        question=case.question,
        answerable=case.answerable,
        answer=ans.answer,
        grounded=ans.grounded,
        retrieved_sources=sources,
        hit=hit,
        recall=recall,
        reciprocal_rank=rr,
        fact_accuracy=facts,
        faithfulness=faith,
        unsupported_sentences=unsupported,
        refusal_correct=refusal_correct,
    )


def _mean(values: list[float | bool | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(results: list[CaseResult], k: int) -> dict:
    answered = [r for r in results if r.faithfulness is not None]
    return {
        "cases": len(results),
        f"hit_rate@{k}": _mean([r.hit for r in results]),
        f"recall@{k}": _mean([r.recall for r in results]),
        "mrr": _mean([r.reciprocal_rank for r in results]),
        "answer_accuracy": _mean([r.fact_accuracy for r in results]),
        "faithfulness": _mean([r.faithfulness for r in answered]),
        "hallucination_rate": _mean([bool(r.unsupported_sentences) for r in answered]),
        "refusal_accuracy": _mean([r.refusal_correct for r in results]),
    }


def run_evaluation(rag: RAGPipeline, cases_path: str | Path, report_dir: str | Path | None = None) -> dict:
    cases = load_cases(cases_path)
    use_llm_judge = rag.settings.llm_provider != "offline"
    results = [evaluate_case(rag, c, use_llm_judge) for c in cases]
    summary = summarize(results, rag.settings.top_k)
    summary["llm"] = rag.generator.name
    summary["judge"] = "llm" if use_llm_judge else "token-overlap"

    if report_dir:
        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_results.json").write_text(
            json.dumps({"summary": summary, "cases": [asdict(r) for r in results]}, indent=2), encoding="utf-8"
        )
        (out / "eval_report.md").write_text(render_markdown(summary, results), encoding="utf-8")
    return {"summary": summary, "results": results}


def render_markdown(summary: dict, results: list[CaseResult]) -> str:
    lines = ["# Evaluation report", "", f"Generator: `{summary['llm']}` · Judge: `{summary['judge']}`", ""]
    lines += ["| Metric | Value |", "|---|---|"]
    for key, value in summary.items():
        if key not in {"llm", "judge"}:
            lines.append(f"| {key} | {value} |")
    lines += ["", "## Cases", "", "| id | answerable | hit | facts | faithful | refusal ok |", "|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    for r in results:
        lines.append(
            f"| {r.id} | {r.answerable} | {fmt(r.hit)} | {fmt(r.fact_accuracy)} | {fmt(r.faithfulness)} | {r.refusal_correct} |"
        )
    failures = [r for r in results if not r.refusal_correct or (r.fact_accuracy is not None and r.fact_accuracy < 1)]
    if failures:
        lines += ["", "## Misses to look at", ""]
        for r in failures:
            lines += [f"**{r.id}**: {r.question}", "", f"> {r.answer}", ""]
    return "\n".join(lines) + "\n"
