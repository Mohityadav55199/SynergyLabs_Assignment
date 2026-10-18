import pytest
from fastapi.testclient import TestClient

import src.vector_stores as vs
from src.vector_stores.lance_store import LanceDBVectorStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(vs._store_cache, "lancedb", LanceDBVectorStore(str(tmp_path / "lance")))
    from src.api import main
    monkeypatch.setattr(main.settings, "corpus_dir", str(tmp_path / "corpus"))
    monkeypatch.setattr(main.settings, "admin_token", "")
    return TestClient(main.app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy" and body["embedding_dim"] == 384 and body["vectors_indexed"] == 0


def test_empty_store_returns_no_context_without_llm(client):
    r = client.post("/query", json={"query": "anything", "store_type": "lancedb"})
    body = r.json()
    assert r.status_code == 200 and body["generation_mode"] == "no_context"
    assert body["citations"] == [] and body["token_usage"]["total_tokens"] == 0
    assert "do not have sufficient information" in body["answer"]


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "   "}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 99},
                                     {"query": "x", "metadata_filter": {"text": "y"}}, {"query": "x", "store_type": "pinecone"}])
def test_invalid_queries_rejected(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_upload_rejects_bad_type_empty_and_traversal(client):
    assert client.post("/ingest/upload", files={"file": ("x.exe", b"abc")}).status_code == 415
    assert client.post("/ingest/upload", files={"file": ("a.md", b"")}).status_code == 422
    r = client.post("/ingest/upload", files={"file": ("../../evil.md", b"# Title\n" + b"hello world " * 40)})
    assert r.status_code == 200 and r.json()["filename"] == "evil.md"


def test_upload_is_idempotent_and_corrupt_pdf_is_a_clean_422(client):
    body = b"# Doc\n" + b"alpha beta gamma delta " * 80
    first = client.post("/ingest/upload", files={"file": ("d.md", body)}).json()
    second = client.post("/ingest/upload", files={"file": ("d.md", body)}).json()
    assert first["inserted_chunks"] > 0 and second["inserted_chunks"] == 0
    assert second["skipped_duplicates"] == first["total_chunks"]
    assert client.post("/ingest/upload", files={"file": ("bad.pdf", b"not a pdf")}).status_code == 422


def test_admin_token_protects_ingest(client, monkeypatch):
    from src.api import main
    monkeypatch.setattr(main.settings, "admin_token", "s3cret")
    assert client.post("/ingest/upload", files={"file": ("a.md", b"x" * 300)}).status_code == 401
    ok = client.post("/ingest/upload", files={"file": ("a.md", b"hello there " * 50)}, headers={"X-Admin-Token": "s3cret"})
    assert ok.status_code == 200


def test_documents_endpoint_reflects_uploads(client):
    assert client.get("/documents").json()["total_documents"] == 0
    client.post("/ingest/upload", files={"file": ("d.md", b"# Doc\n" + b"alpha beta gamma delta " * 80)})
    body = client.get("/documents").json()
    assert body["total_documents"] == 1 and body["documents"][0]["filename"] == "d.md"
    assert body["total_chunks"] == body["documents"][0]["chunks"] > 0
