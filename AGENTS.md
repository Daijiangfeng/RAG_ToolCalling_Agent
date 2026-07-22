# AGENTS.md

Guidance for coding agents working in this repository. Keep changes aligned with
the conventions below. This file is intentionally short; see `README.md` for the
full product overview.

## Setup / Run / Test (backend)

```bash
# from repo root
python -m venv backend/.venv
# Windows PowerShell: backend\.venv\Scripts\Activate.ps1
# Linux/macOS:        source backend/.venv/bin/activate
pip install -r backend/requirements.txt

# Run the API (from backend/)
cd backend
uvicorn main:app --reload --port 8000

# Run the offline test suite (from backend/) -- no API keys or heavy models
cd backend
pytest
```

`tests/conftest.py` forces fully-offline backends (hash embedding, lexical
reranker, in-memory vector store, mock LLM) at import time, so `pytest` is
deterministic and needs no environment setup.

## Directory Responsibilities (backend/)

- `app/` - configuration (`config.py`), DB (`db.py`), LLM client (`llm.py`),
  error contracts (`errors.py`), schemas, logging.
- `rag/` - retrieval pipeline: loader, splitter, embedding, vectorstore,
  retriever, reranker, generator, evaluator. Shared tokenizer lives in
  `rag/text.py`.
- `agent/` - LangGraph agent: `graph.py`, `nodes.py`, `router.py`, `state.py`,
  `memory.py`.
- `tools/` - callable tools plus `registry.py`.
- `api/` - FastAPI routers (chat, documents, upload, evaluation).

## Key Conventions

- New tools: add a module in `tools/` exposing `SCHEMA` (with a `name`) and a
  `run(**kwargs)` callable, then register it in `tools/registry.py`. The
  `SCHEMA["name"]` must match the intent name the router dispatches.
- Provider failures: never silently degrade to mock when credentials are
  configured. Raise `ProviderError` (see `app/errors.py`); `main.py` converts it
  to HTTP 502 with an actionable message. Missing credentials -> offline mock is
  the only sanctioned fallback.
- Offline fallbacks are selected by environment variables: `EMBEDDING_BACKEND`,
  `RERANKER_BACKEND`, `VECTOR_BACKEND` (see `.env.example`).
- Lexical tokenization must go through `rag/text.py::tokenize` so the reranker
  and evaluator never drift apart.

## Validation Before Handing Off

- Run `cd backend; pytest` and confirm the suite is green.
- If you touched evaluation, run `run_evaluation()` (offline) and check
  `evaluation_report.md`, including the "Regression vs Baseline" section.
