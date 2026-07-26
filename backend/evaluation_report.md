# RAG Evaluation Report

Generated at: 2026-07-26T14:35:14.916196  
Total cases: 20

## Retrieval
- Precision@K: 0.1667
- Recall@K: 0.325

## Generation
- Answer Relevance: 0.2206
- Context Relevance: 0.0686
- Faithfulness: 0.3304

## Safety
- Hallucination Rate: 0.15

## By Category

| Type | Count | Passed | Faithfulness |
| ---- | ----- | ------ | ------------ |
| knowledge | 5 | 5 | 0.2854 |
| multi_hop | 5 | 4 | 0.4364 |
| no_answer | 5 | 3 | 0.6 |
| tool | 5 | 5 | 0.0 |

## Regression vs Baseline

WARNING: 2 metric(s) regressed beyond tolerance (advisory, non-blocking).

| Metric | Baseline | Current | Delta |
| ------ | -------- | ------- | ----- |
| generation.answer_relevance | 0.2441 | 0.2206 | -0.0235 |
| generation.context_relevance | 0.0815 | 0.0686 | -0.0129 |
| generation.faithfulness | 0.3719 | 0.3304 | -0.0415 |
| retrieval.precision_at_k (regressed) | 0.22 | 0.1667 | -0.0533 |
| retrieval.recall_at_k (regressed) | 0.4375 | 0.325 | -0.1125 |
| safety.hallucination_rate | 0.25 | 0.15 | -0.1000 |

## Per-case Detail

| ID | Type | Behaved | Tools | Confidence |
| -- | ---- | ------- | ----- | ---------- |
| kb-1 | knowledge | PASS | - | 0.314 |
| kb-2 | knowledge | PASS | - | 0.485 |
| kb-3 | knowledge | PASS | - | 0.894 |
| kb-4 | knowledge | PASS | - | 0.373 |
| kb-5 | knowledge | PASS | - | 0.332 |
| multi-1 | multi_hop | PASS | - | 0.637 |
| multi-2 | multi_hop | PASS | - | 0.356 |
| multi-3 | multi_hop | PASS | - | 0.32 |
| multi-4 | multi_hop | FAIL | - | 0.232 |
| multi-5 | multi_hop | PASS | - | 0.561 |
| noans-1 | no_answer | FAIL | calculator | 0.0 |
| noans-2 | no_answer | FAIL | web_search | 0.0 |
| noans-3 | no_answer | PASS | - | 0.21 |
| noans-4 | no_answer | PASS | - | 0.223 |
| noans-5 | no_answer | PASS | - | 0.187 |
| tool-1 | tool | PASS | calculator | 0.0 |
| tool-2 | tool | PASS | calculator | 0.0 |
| tool-3 | tool | PASS | web_search | 0.0 |
| tool-4 | tool | PASS | datetime | 0.0 |
| tool-5 | tool | PASS | web_search | 0.0 |
