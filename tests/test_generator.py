from src.generator import llm
from src.generator.llm import GroundedRAGService, extract_cited_indices, is_fallback_answer
from src.vector_stores.base import SearchResult


def test_citation_parsing_is_exact():
    assert extract_cited_indices("As stated [Chunk-1] and [Chunk-3].", 4) == [0, 2]
    assert extract_cited_indices("See [Chunk-10]", 4) == []            # out of range, and not a match for Chunk-1
    assert extract_cited_indices("[Chunk-2] [Chunk-2]", 4) == [1]
    assert extract_cited_indices("no citations", 4) == []


def test_fallback_detection():
    assert is_fallback_answer("I do not have sufficient information in the provided context to answer this question.")
    assert not is_fallback_answer("HNSW is a graph index [Chunk-1].")
    assert is_fallback_answer('"I do not have sufficient information in the provided context to answer this question."')
    # quoting the sentence inside a real answer is not a refusal
    assert not is_fallback_answer('The system must reply: "I do not have sufficient information in the provided context to answer this question." [Chunk-1]')


class FakeStore:
    def __init__(self, score):
        self.score = score
    def search(self, query_vector, top_k, metadata_filter=None):
        return [SearchResult(chunk_id=f"c{i}", doc_id="d", filename="f.md", format="md", section="S",
                             text=f"text {i}", score=self.score - i * 0.01, metadata={}) for i in range(top_k)]


def make_service(monkeypatch, score, llm_text=None, llm_error=None):
    monkeypatch.setattr(llm, "get_vector_store", lambda _t: FakeStore(score))
    svc = GroundedRAGService.__new__(GroundedRAGService)
    svc.store_type = "lancedb"
    svc.embedder = type("E", (), {"embed_query": lambda self, q: [0.0]})()
    svc.groq_client = object()
    def gen(prompt):
        if llm_error:
            raise llm_error
        return {"text": llm_text, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    svc._generate = gen
    return svc


def test_below_threshold_refuses_without_calling_llm(monkeypatch):
    svc = make_service(monkeypatch, score=0.5, llm_error=AssertionError("LLM must not be called"))
    r = svc.query("q", top_k=3, similarity_threshold=0.68)
    assert r.generation_mode == "no_context" and r.citations == [] and r.retrieved_chunks_count == 0


def test_cited_answer_returns_only_cited_chunks(monkeypatch):
    svc = make_service(monkeypatch, score=0.9, llm_text="Fact one [Chunk-2].")
    r = svc.query("q", top_k=3, similarity_threshold=0.68)
    assert [c.chunk_id for c in r.citations] == ["c1"] and r.token_usage["total_tokens"] == 15


def test_llm_refusal_has_no_citations(monkeypatch):
    svc = make_service(monkeypatch, score=0.9, llm_text="I do not have sufficient information in the provided context to answer this question.")
    assert svc.query("q", top_k=3).citations == []


def test_llm_failure_degrades_to_labelled_extractive_answer(monkeypatch):
    svc = make_service(monkeypatch, score=0.9, llm_error=RuntimeError("boom"))
    r = svc.query("q", top_k=3, similarity_threshold=0.68)
    assert r.generation_mode == "extractive_fallback" and "LLM unavailable" in r.answer


def test_blank_query_rejected(monkeypatch):
    import pytest
    with pytest.raises(ValueError):
        make_service(monkeypatch, 0.9).query("   ")


def test_unavailable_model_falls_back_to_an_available_one(monkeypatch):
    class NotFoundError(Exception):
        pass

    calls = []

    class Msg:
        content = "Fact [Chunk-1]."

    class Resp:
        choices = [type("C", (), {"message": Msg()})()]
        usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})()

    class Completions:
        def create(self, **kw):
            calls.append(kw["model"])
            if kw["model"] == "retired-model":
                raise NotFoundError("model not found")
            return Resp()

    client = type("G", (), {
        "chat": type("Ch", (), {"completions": Completions()})(),
        "models": type("M", (), {"list": lambda self: type("L", (), {"data": [type("X", (), {"id": "openai/gpt-oss-20b"})()]})()})(),
    })()
    monkeypatch.setattr(llm.settings, "groq_model", "retired-model")
    svc = GroundedRAGService.__new__(GroundedRAGService)
    svc.groq_client = client
    out = svc._generate("prompt")
    assert out["model"] == "openai/gpt-oss-20b" and calls[0] == "retired-model"
    assert svc._generate("again")["model"] == "openai/gpt-oss-20b"   # cached: no second failure
