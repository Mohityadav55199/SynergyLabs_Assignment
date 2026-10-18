import numpy as np
import pytest
from src.ingestion.chunker import DocumentChunker
from src.vector_stores.lance_store import LanceDBVectorStore
from src.vector_stores.qdrant_store import QdrantVectorStore
from src.vector_stores.base import validate_metadata_filter
from src.config import settings

DIM = settings.embedding_dim

def vec(seed):
    v = np.random.default_rng(seed).normal(size=DIM)
    return (v / np.linalg.norm(v)).tolist()

def make_chunks(doc_id, filename, fmt, n=5):
    text = "\n\n".join(f"Paragraph {i} of {filename}. " + "filler " * 30 for i in range(n))
    return DocumentChunker(300, 30).chunk_document(doc_id, filename, fmt, text)

@pytest.fixture(params=["lancedb", "qdrant"])
def store(request, tmp_path):
    return LanceDBVectorStore(str(tmp_path)) if request.param == "lancedb" else QdrantVectorStore(str(tmp_path))

def test_insert_is_idempotent(store):
    chunks = make_chunks("d1", "a.md", "md")
    vectors = [vec(i) for i in range(len(chunks))]
    assert store.insert_chunks(chunks, vectors) == len(chunks)
    assert store.insert_chunks(chunks, vectors) == 0
    assert store.count() == len(chunks)

def test_metadata_filter_restricts_results(store):
    a, b = make_chunks("d1", "a.md", "md"), make_chunks("d2", "b.pdf", "pdf")
    store.insert_chunks(a + b, [vec(i) for i in range(len(a + b))])
    res = store.search(vec(0), top_k=10, metadata_filter={"format": "pdf"})
    assert res and all(r.format == "pdf" for r in res)

def test_filter_with_quote_does_not_break_or_inject(store):
    store.insert_chunks(make_chunks("d1", "a.md", "md"), [vec(i) for i in range(10)])
    assert store.search(vec(0), 3, {"filename": "x' OR '1'='1"}) == []

def test_unknown_filter_field_rejected(store):
    with pytest.raises(ValueError):
        store.search(vec(0), 3, {"text": "x"})

def test_search_on_empty_store(store):
    assert store.search(vec(0), 3) == []

def test_edited_file_replaces_stale_vectors(store):
    old = make_chunks("v1", "a.md", "md")
    store.insert_chunks(old, [vec(i) for i in range(len(old))])
    new = make_chunks("v2", "a.md", "md", n=3)
    removed = store.delete_stale_versions("a.md", "v2")
    store.insert_chunks(new, [vec(100 + i) for i in range(len(new))])
    assert removed == len(old)
    assert store.count() == len(new)

def test_filter_validation():
    assert validate_metadata_filter(None) == {}
    with pytest.raises(ValueError):
        validate_metadata_filter({"format": 3})
