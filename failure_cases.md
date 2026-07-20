# Failure Case 分析

本文档记录 RAG + Agent 系统在评测与使用中暴露的三类典型失败场景、根因分析与已实现的缓解手段。所有结论可在离线兜底模式下复现（详见 `backend/evaluation_report.md`）。

---

## Case 1 · 检索错误（Retrieval Miss）

**现象**
- 用户问题与知识库文档使用了不同的表述（同义改写、缩写、跨语言），导致向量相似度检索的 Top-K 中缺失真正相关的 chunk，最终答案缺乏依据或答非所问。
- 评测中体现为 `Precision@K` / `Recall@K` 偏低、`Context Relevance` 低。

**根因**
- 单一 query 的稠密向量表示与文档表述存在语义鸿沟（vocabulary / phrasing mismatch）。
- 多轮对话中出现指代（"它""这个方法"），未做指代消解直接检索。

**缓解手段（已实现）**
- **Query Rewrite**（`rag/advanced.py::query_rewrite`）：结合会话历史（`agent/memory.py`）对 query 做指代消解与改写，再检索。
- **HyDE**（`rag/advanced.py::hyde`）：先让 LLM 生成"假设性答案"，用其向量检索，缩小 query↔doc 表述差异。
- **较大的初检窗口**：`TOP_K=20` 先召回，再由 Reranker 精排到 `RERANK_TOP_N=5`，降低漏召回概率。

**进一步改进方向**
- 引入 BM25 + 向量的 **混合检索（hybrid search）** 与 RRF 融合。
- 对文档做多粒度切分（父子块 / 句级）。

---

## Case 2 · 幻觉（Hallucination）

**现象**
- 知识库中没有充分证据时，模型仍"自信"地编造答案；评测中体现为 `Hallucination Rate > 0`、`Faithfulness` 低。
- `no_answer` 类问题本应拒答，却被强行作答（例如被误路由到工具后编造）。

**根因**
- 生成阶段未对"上下文是否足够支撑答案"做把关。
- 缺少置信度门槛，低质量检索结果也被送入生成。

**缓解手段（已实现）**
- **Confidence Threshold + 拒答**（`agent/nodes.py::rerank_node` / `reject_node`）：当最高 rerank 分数低于 `CONFIDENCE_THRESHOLD=0.30` 或无上下文时，路由到 `reject_node` 返回固定拒答语「知识库中没有足够信息回答该问题。」
- **强约束 System Prompt**（`rag/generator.py`）：只依据上下文作答、禁止编造、必须引用来源、信息不足即拒答。
- **Self-Critique**（`agent/nodes.py::critique_node`）：对生成结果做事实性/是否答题校验，判定 FAIL 时触发一次重生成。
- **来源引用**：答案附带 `file_name / page_number / score`，便于人工核验。

**进一步改进方向**
- 句级引用对齐（claim → evidence span）。
- 用更强的 LLM-as-judge 替代离线词法启发式判分。

---

## Case 3 · 复杂问题 / 多跳推理（Multi-hop）

**现象**
- 需要跨多个文档/段落综合推理的问题，单轮检索无法一次覆盖全部证据，答案不完整。
- 评测 `multi_hop` 类中出现个别 FAIL（见报告 `multi-3`）。

**根因**
- 固定 "检索一次 → 生成" 的 Chain 无法自适应问题复杂度。
- 单一 query 难以同时命中多个子主题。

**缓解手段（已实现）**
- **Agentic RAG（LangGraph 条件路由）**（`agent/graph.py`）：Agent 通过 `intent_router` 与条件边自主决定是否改写、是否检索、是否重生成，而非固定 Chain。
- **Rewrite → Retrieve → Rerank → Critique 回路**：Self-Critique 失败可回到 `generate` 重生成一次，提升复杂问题稳健性。

**进一步改进方向**
- 引入 **多步检索 / 子问题分解（decomposition）**：将复杂问题拆成子查询分别检索后聚合。
- 增加检索轮次上限与迭代式证据累积（iterative retrieval）。

---

## 附:离线基线说明

仓库内 `evaluation_report.md` 的指标是在**离线 mock 模式**（hash 向量 embedding + 词法启发式判分）下生成的**基线**，数值偏低属预期——它验证的是**评估框架与失败模式的可复现性**。接入真实 embedding / reranker / LLM 后，上述三类问题的指标均会显著改善。
