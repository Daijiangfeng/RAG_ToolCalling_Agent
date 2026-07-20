# RAG Evaluation Report

Generated at: 2026-07-20T05:51:35.106272  
Total cases: 20

## Retrieval
- Precision@K: 0.1833
- Recall@K: 0.3

## Generation
- Answer Relevance: 0.1742
- Context Relevance: 0.0341
- Faithfulness: 0.2612

## Safety
- Hallucination Rate: 0.15

## By Category

| Type | Count | Passed | Faithfulness |
| ---- | ----- | ------ | ------------ |
| knowledge | 5 | 5 | 0.1279 |
| multi_hop | 5 | 4 | 0.3169 |
| no_answer | 5 | 3 | 0.6 |
| tool | 5 | 5 | 0.0 |

## Per-case Detail

| ID | Type | Behaved | Tools | Confidence |
| -- | ---- | ------- | ----- | ---------- |
| kb-1 | knowledge | PASS | - | 1.0 |
| kb-2 | knowledge | PASS | - | 1.0 |
| kb-3 | knowledge | PASS | - | 1.0 |
| kb-4 | knowledge | PASS | - | 1.0 |
| kb-5 | knowledge | PASS | - | 1.0 |
| multi-1 | multi_hop | PASS | - | 1.0 |
| multi-2 | multi_hop | PASS | - | 1.0 |
| multi-3 | multi_hop | FAIL | - | 0.0 |
| multi-4 | multi_hop | PASS | - | 1.0 |
| multi-5 | multi_hop | PASS | - | 1.0 |
| noans-1 | no_answer | FAIL | calculator | 0.0 |
| noans-2 | no_answer | FAIL | web_search | 0.0 |
| noans-3 | no_answer | PASS | - | 0.0 |
| noans-4 | no_answer | PASS | - | 0.0 |
| noans-5 | no_answer | PASS | - | 0.0 |
| tool-1 | tool | PASS | calculator | 0.0 |
| tool-2 | tool | PASS | calculator | 0.0 |
| tool-3 | tool | PASS | web_search | 0.0 |
| tool-4 | tool | PASS | datetime | 0.0 |
| tool-5 | tool | PASS | web_search | 0.0 |
