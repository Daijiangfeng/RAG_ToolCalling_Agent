"""API integration tests using FastAPI TestClient."""

import io

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_markdown():
    content = b"# Test Doc\n\nThis is a test about widgets and gadgets."
    files = {"file": ("test.md", io.BytesIO(content), "text/markdown")}
    r = client.post("/api/upload", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "processed"
    assert body["chunks"] >= 1


def test_upload_rejects_unknown_type():
    files = {"file": ("bad.exe", io.BytesIO(b"x"), "application/octet-stream")}
    r = client.post("/api/upload", files=files)
    assert r.status_code == 400


def test_chat_rag():
    r = client.post("/api/chat", json={"question": "RAG 的优势是什么?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "sources" in body and "trace" in body


def test_chat_tool():
    r = client.post("/api/chat", json={"question": "12345*678"})
    body = r.json()
    assert body["tools"] and body["tools"][0]["tool"] == "calculator"


def test_chat_stream():
    with client.stream("POST", "/api/chat/stream", json={"question": "什么是 RAG?"}) as r:
        assert r.status_code == 200
        chunks = [line for line in r.iter_lines()]
    joined = "\n".join(chunks)
    assert "data:" in joined and "done" in joined


def test_evaluation_endpoint():
    r = client.get("/api/evaluation")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 20
    assert "retrieval" in body and "generation" in body and "safety" in body


def test_documents_endpoint():
    r = client.get("/api/documents")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
