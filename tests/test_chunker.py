import pytest
from src.ingestion.chunker import DocumentChunker

TEXT = ("Alpha sentence about vectors. " * 40 + "\n\n" + "Beta paragraph about storage costs. " * 40)

def chunk(size=200, overlap=20, text=TEXT, doc_id="d1"):
    return DocumentChunker(chunk_size=size, chunk_overlap=overlap).chunk_document(doc_id, "f.md", "md", text)

def test_chunks_respect_size():
    assert all(len(c.text) <= 200 + 20 + 1 for c in chunk())

def test_deterministic_hashes_make_reingest_idempotent():
    assert [c.chunk_hash for c in chunk()] == [c.chunk_hash for c in chunk()]

def test_hash_changes_with_content_and_doc():
    assert {c.chunk_hash for c in chunk()}.isdisjoint({c.chunk_hash for c in chunk(doc_id="d2")})

def test_zero_overlap_is_respected():
    assert DocumentChunker(chunk_size=100, chunk_overlap=0).chunk_overlap == 0

def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=-1)

def test_empty_and_whitespace_text_yield_no_chunks():
    assert chunk(text="") == []
    assert chunk(text="   \n\n  ") == []

def test_text_without_separators_is_hard_split():
    assert len(chunk(size=100, overlap=10, text="x" * 1000)) >= 10

def test_section_is_tracked():
    text = "## Section 1: Intro\n" + "word " * 30 + "\n\n## Section 2: Costs\n" + "money " * 30
    sections = {c.section for c in chunk(size=120, overlap=10, text=text)}
    assert any("Intro" in s for s in sections) and any("Costs" in s for s in sections)
