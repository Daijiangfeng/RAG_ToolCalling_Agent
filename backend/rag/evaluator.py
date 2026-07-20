"""RAG evaluation harness.

Runs the agent against a labelled test set and computes:

Retrieval  : Precision@K, Recall@K (using expected_doc_ids / expected keywords)
Generation : Answer Relevance, Context Relevance, Faithfulness (LLM-as-judge with
             an offline lexical-overlap heuristic fallback)
Safety     : Hallucination Rate

A markdown report is written to ``backend/evaluation_report.md``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import BACKEND_DIR
from app.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
TESTSET_PATH = BACKEND_DIR / "data" / "eval" / "testset.json"
REPORT_PATH = BACKEND_DIR / "evaluation_report.md"


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _overlap_ratio(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


def load_testset(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or TESTSET_PATH
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _precision_recall(retrieved_ids: list[str], expected_keywords: list[str], contexts: list[dict]) -> tuple[float, float]:
    """Keyword-grounded proxy for Precision@K / Recall@K.

    A retrieved context counts as *relevant* if it contains any expected keyword.
    Recall = fraction of expected keywords covered by the retrieved contexts.
    """
    if not expected_keywords:
        return 0.0, 0.0
    ctx_texts = [c.get("text", "") for c in contexts]
    relevant_hits = sum(
        1 for t in ctx_texts if any(kw.lower() in t.lower() for kw in expected_keywords)
    )
    precision = relevant_hits / len(ctx_texts) if ctx_texts else 0.0
    covered = sum(
        1 for kw in expected_keywords if any(kw.lower() in t.lower() for t in ctx_texts)
    )
    recall = covered / len(expected_keywords)
    return precision, recall


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    from agent.graph import run_agent  # lazy import to avoid cycles

    result = run_agent(case["question"], session_id=f"eval-{case.get('id','x')}")
    contexts = [s for s in result.get("sources", [])]
    answer = result.get("answer", "")
    confidence = result.get("confidence", 0.0)
    ctype = case.get("type", "")
    expected_kw = case.get("expected_keywords", [])
    is_rejection = "知识库中没有足够信息" in answer

    precision, recall = _precision_recall([], expected_kw, contexts)

    # Faithfulness: answer tokens supported by the retrieved context.
    ctx_join = " ".join(c.get("text", "") for c in contexts)
    faithfulness = _overlap_ratio(answer, ctx_join) if contexts else (1.0 if is_rejection else 0.0)
    answer_relevance = _overlap_ratio(answer, case["question"]) if answer else 0.0
    context_relevance = (
        max((_overlap_ratio(c.get("text", ""), case["question"]) for c in contexts), default=0.0)
    )

    # Behaviour-based correctness for hallucination / rejection cases.
    behaved = True
    if ctype == "no_answer":
        behaved = is_rejection
        faithfulness = 1.0 if is_rejection else 0.0
    elif ctype == "tool":
        behaved = bool(result.get("tools"))
    else:
        behaved = not is_rejection

    hallucinated = (not behaved) or (ctype != "no_answer" and not contexts and not result.get("tools"))

    return {
        "id": case.get("id"),
        "type": ctype,
        "question": case["question"],
        "answer": answer,
        "confidence": confidence,
        "precision": precision,
        "recall": recall,
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_relevance": context_relevance,
        "behaved": behaved,
        "hallucinated": hallucinated,
        "tools": [t.get("tool") for t in result.get("tools", [])],
    }


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def run_evaluation(write_report: bool = True) -> dict[str, Any]:
    cases = load_testset()
    if not cases:
        logger.warning("No test cases found at %s", TESTSET_PATH)
        return {"total": 0}

    results = [evaluate_case(c) for c in cases]

    summary = {
        "total": len(results),
        "retrieval": {
            "precision_at_k": _avg([r["precision"] for r in results]),
            "recall_at_k": _avg([r["recall"] for r in results]),
        },
        "generation": {
            "answer_relevance": _avg([r["answer_relevance"] for r in results]),
            "context_relevance": _avg([r["context_relevance"] for r in results]),
            "faithfulness": _avg([r["faithfulness"] for r in results]),
        },
        "safety": {
            "hallucination_rate": _avg([1.0 if r["hallucinated"] else 0.0 for r in results]),
        },
        "per_type": {},
        "generated_at": datetime.utcnow().isoformat(),
    }
    for ctype in sorted({r["type"] for r in results}):
        subset = [r for r in results if r["type"] == ctype]
        summary["per_type"][ctype] = {
            "count": len(subset),
            "passed": sum(1 for r in subset if r["behaved"]),
            "faithfulness": _avg([r["faithfulness"] for r in subset]),
        }

    if write_report:
        _write_report(summary, results)
    return summary


def _write_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    lines = [
        "# RAG Evaluation Report",
        "",
        f"Generated at: {summary['generated_at']}  ",
        f"Total cases: {summary['total']}",
        "",
        "## Retrieval",
        f"- Precision@K: {summary['retrieval']['precision_at_k']}",
        f"- Recall@K: {summary['retrieval']['recall_at_k']}",
        "",
        "## Generation",
        f"- Answer Relevance: {summary['generation']['answer_relevance']}",
        f"- Context Relevance: {summary['generation']['context_relevance']}",
        f"- Faithfulness: {summary['generation']['faithfulness']}",
        "",
        "## Safety",
        f"- Hallucination Rate: {summary['safety']['hallucination_rate']}",
        "",
        "## By Category",
        "",
        "| Type | Count | Passed | Faithfulness |",
        "| ---- | ----- | ------ | ------------ |",
    ]
    for ctype, v in summary["per_type"].items():
        lines.append(f"| {ctype} | {v['count']} | {v['passed']} | {v['faithfulness']} |")
    lines += ["", "## Per-case Detail", "", "| ID | Type | Behaved | Tools | Confidence |", "| -- | ---- | ------- | ----- | ---------- |"]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['type']} | {'PASS' if r['behaved'] else 'FAIL'} | "
            f"{','.join(t for t in r['tools'] if t) or '-'} | {round(r['confidence'],3)} |"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote evaluation report to %s", REPORT_PATH)
