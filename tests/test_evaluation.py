import json

from eka.evaluation import load_cases, llm_judge, overlap_judge, run_evaluation

from .conftest import EVAL_CASES


def test_overlap_judge_flags_unsupported_sentences():
    ctx = ["Employees receive 18 days of paid annual leave per calendar year."]
    verdicts = overlap_judge(["Employees receive 18 days of annual leave.", "Employees also get a company car."], ctx)
    assert verdicts == [True, False]


def test_llm_judge_parses_json_and_falls_back_when_malformed():
    class Judge:
        def __init__(self, reply):
            self.reply = reply

        def complete(self, system, user):
            return self.reply

    sentences = ["Leave is 18 days.", "Cars are provided."]
    ctx = ["Leave is 18 days per year."]
    assert llm_judge(Judge("Here you go: [true, false]"), sentences, ctx) == [True, False]
    assert llm_judge(Judge("not json"), sentences, ctx) == overlap_judge(sentences, ctx)


def test_eval_cases_file_is_valid():
    cases = load_cases(EVAL_CASES)
    assert len(cases) >= 25
    assert any(not c.answerable for c in cases)
    assert len({c.id for c in cases}) == len(cases)


def test_full_evaluation_meets_quality_bar(rag, tmp_path):
    out = run_evaluation(rag, EVAL_CASES, tmp_path)
    s = out["summary"]
    # Regression guard for the offline baseline. Real providers should beat these numbers.
    assert s["hit_rate@5"] >= 0.95
    assert s["mrr"] >= 0.9
    assert s["answer_accuracy"] >= 0.9
    assert s["faithfulness"] >= 0.95
    assert s["hallucination_rate"] <= 0.05
    assert s["refusal_accuracy"] >= 0.9
    saved = json.loads((tmp_path / "eval_results.json").read_text())
    assert saved["summary"]["cases"] == len(saved["cases"])
    assert (tmp_path / "eval_report.md").read_text().startswith("# Evaluation report")
