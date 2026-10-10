from types import SimpleNamespace

from eka.llm import NO_ANSWER, AnthropicGenerator, OpenAIGenerator, SYSTEM_PROMPT
from eka.models import SearchFilters
from eka.pipeline import RAGPipeline


def test_answer_contains_fact_and_citation(rag):
    ans = rag.ask("How many days of paid annual leave do permanent employees get?")
    assert ans.grounded
    assert "18 days" in ans.answer
    assert ans.citations and ans.citations[0].source == "leave_policy.md"
    assert "[1]" in ans.answer


def test_unanswerable_question_is_declined(rag):
    ans = rag.ask("What is the cafeteria menu on Fridays?")
    assert not ans.grounded
    assert ans.answer == NO_ANSWER
    assert ans.citations == []


def test_empty_index_declines(settings):
    ans = RAGPipeline(settings).ask("How many days of leave?")
    assert ans.answer == NO_ANSWER and not ans.grounded


def test_stream_emits_tokens_then_done(rag):
    events = list(rag.stream("What time is the dispatch plan frozen?"))
    assert events[-1]["type"] == "done"
    assert all(e["type"] == "token" for e in events[:-1])
    text = "".join(e["text"] for e in events[:-1])
    assert "17:30" in text
    assert events[-1]["citations"][0]["source"] == "dispatch_sop.md"


def test_filters_change_the_answer(rag):
    ans = rag.ask("How many days?", SearchFilters(department="Quality"))
    assert all(c.department == "Quality" for c in ans.citations)


def test_citations_ignore_out_of_range_numbers(rag):
    retrieved = rag.retrieve("annual leave")
    cites = RAGPipeline.build_citations("Fact [1]. Other [99].", retrieved)
    assert [c.index for c in cites] == [1]


# ---- provider adapters, tested with fake SDK clients (no network) -------------
def _chunks(rag):
    return rag.retrieve("annual leave days")


def test_openai_generator_streams_and_sends_grounding_prompt(rag):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        if kwargs.get("stream"):
            return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))]) for p in ["18 days ", "[1]"]]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[true]"))])

    gen = OpenAIGenerator.__new__(OpenAIGenerator)
    gen.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    gen.model, gen.name = "gpt-test", "openai:gpt-test"

    assert "".join(gen.stream("How much leave?", _chunks(rag))) == "18 days [1]"
    assert captured["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "[1] Leave Policy (HR)" in captured["messages"][1]["content"]
    assert captured["temperature"] == 0
    assert gen.complete("sys", "user") == "[true]"


def test_anthropic_generator_streams(rag):
    class FakeStream:
        text_stream = iter(["Employees get ", "18 days [1]"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    captured = {}

    def stream(**kwargs):
        captured.update(kwargs)
        return FakeStream()

    gen = AnthropicGenerator.__new__(AnthropicGenerator)
    gen.client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    gen.model, gen.name = "claude-test", "anthropic:claude-test"
    assert "".join(gen.stream("How much leave?", _chunks(rag))) == "Employees get 18 days [1]"
    assert captured["system"] == SYSTEM_PROMPT
