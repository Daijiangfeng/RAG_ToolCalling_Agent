# Intelligent Knowledge Agent Platform

> 企业级 **RAG + Tool Calling Agent** 平台 — 基于 **LangGraph** 的自主 Agent、完整 RAG 链路（检索 → 重排 → 压缩 → 生成）、工具调用、来源引用、不确定拒答、自动评估与 Streaming 输出。前端提供 Chat / Agent Trace / Knowledge Base / Evaluation 四个页面。

本项目采用 **混合 + 离线兜底** 策略:默认对接 OpenAI 兼容 API + BGE 模型;当缺少 `OPENAI_API_KEY` 或本地模型不可用时，自动降级为 **mock LLM / 哈希向量 embedding / 词法 reranker / 内存向量库**，因此 **无需任何密钥即可跑通全部 Demo 与测试**。

---

## 目录

1. [项目介绍](#1-项目介绍)
2. [系统架构](#2-系统架构)
3. [技术栈](#3-技术栈)
4. [RAG 完整链路](#4-rag-完整链路)
5. [LangGraph Agent 流程](#5-langgraph-agent-流程)
6. [Tool Calling 说明](#6-tool-calling-说明)
7. [普通函数 vs Tool Calling](#7-普通函数-vs-tool-calling)
8. [Advanced RAG](#8-advanced-rag)
9. [安装与运行（本地）](#9-安装与运行本地)
10. [Docker 部署](#10-docker-部署)
11. [环境变量配置](#11-环境变量配置)
12. [API 文档](#12-api-文档)
13. [Demo 说明](#13-demo-说明)
14. [评估（Evaluation）](#14-评估evaluation)
15. [Failure Case 分析](#15-failure-case-分析)
16. [测试](#16-测试)
17. [目录结构](#17-目录结构)

---

## 1. 项目介绍

平台面向"企业知识库智能问答"场景，支持上传 PDF/Markdown 文档，自动解析、切分、向量化入库；用户提问时由 LangGraph Agent 自主判断意图，选择走 **RAG 检索** 或 **工具调用**（计算器 / 网络搜索 / 文件查询 / 日期时间），生成 **带来源引用** 的答案，并在信息不足时 **主动拒答**。系统内置 **RAG 自动评估** 与 **Agent 执行轨迹** 可视化。

核心特性:
- 📄 文档上传与解析（PDF via PyMuPDF、Markdown 保留标题层级）
- 🔎 完整 RAG：Loader → Splitter → Embedding → VectorStore → Retriever → **Reranker** → Context Compression → Generator
- 🧠 **LangGraph** StateGraph 编排（State / Node / Conditional Edge / Tool Execution），非简单 Chain
- 🛠️ **Tool Calling**：LLM 语义选择工具 + 生成参数
- 📌 来源引用（file_name / page_number / score）
- 🚫 不确定拒答（置信度阈值 + 无上下文兜底）
- 🔬 Self-Critique（事实性/是否答题校验，失败重生成一次）
- 📊 RAG 自动评估（Precision@K / Recall@K / Faithfulness / Answer & Context Relevance / Hallucination Rate）
- ⚡ Streaming 输出（SSE）
- 🧩 Advanced RAG：Query Rewrite / HyDE / Agentic RAG

---

## 2. 系统架构

```
┌───────────────────────────── Frontend (React + AntD + Zustand) ─────────────────────────────┐
│  Chat 对话   │   Agent Trace 轨迹   │   Knowledge Base 知识库   │   Evaluation 评估看板       │
└───────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                             │  axios / fetch(SSE)  ->  /api/*
┌───────────────────────────────────────────▼──────────────────────────────────────────────────┐
│                                 FastAPI (main.py + api/*)                                       │
│  /api/upload   /api/chat   /api/chat/stream   /api/documents   /api/evaluation[/run]            │
└───────────────┬──────────────────────────────────────────────┬─────────────────────────────────┘
                │                                                │
        ┌───────▼────────┐                          ┌────────────▼───────────┐
        │  Ingest Pipeline│                          │   LangGraph Agent      │
        │ loader→splitter │                          │  state / router /nodes │
        │ →embedding→store│                          │  graph / memory        │
        └───────┬────────┘                          └────────────┬───────────┘
                │                                                │ conditional routing
        ┌───────▼─────────────────────────────┐     ┌────────────▼───────────┐   ┌──────────────┐
        │  VectorStore (Chroma / InMemory)     │◄────│  Retriever → Reranker  │   │   Tools      │
        │  Embedder (OpenAI / BGE / Hash)      │     │  → Generator (LLM)     │   │ calc/search/ │
        └──────────────────────────────────────┘     └────────────────────────┘   │ file/datetime│
                                                                                    └──────────────┘
        ┌──────────────────────────────────────┐     ┌────────────────────────┐
        │  SQLite (Document / ChatTrace)        │     │  LLMClient (OpenAI /    │
        │                                       │     │  mock offline fallback) │
        └──────────────────────────────────────┘     └────────────────────────┘
```

---

## 3. 技术栈

| 层 | 技术 |
| --- | --- |
| 后端 | Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2, Uvicorn |
| Agent | **LangGraph** (StateGraph), langchain-core |
| LLM | OpenAI 兼容 SDK（OpenAI / Azure / DeepSeek / Qwen …）+ 离线 mock |
| Embedding | OpenAI / BGE (sentence-transformers) / Hash（离线兜底） |
| 向量库 | Chroma（持久化）/ InMemory（离线兜底），接口可扩展 Milvus/Qdrant |
| Reranker | Cross-Encoder (BAAI/bge-reranker-large) / Lexical（离线兜底） |
| 文档解析 | PyMuPDF (PDF)、Markdown 解析 |
| 数据库 | SQLite（零配置） |
| 前端 | React 18, TypeScript, Vite, Ant Design 5, Zustand, Axios, react-markdown |
| 部署 | Docker + docker-compose（backend + frontend/nginx） |
| 测试 | pytest（35 用例，离线全绿） |

---

## 4. RAG 完整链路

```
文档 → [Loader] → [Splitter] → [Embedding] → [VectorStore]        (入库)
                                                     │
问题 ─────────────► [Retriever top_k=20] ───► [Reranker top_n=5] ──► [Context Compression] ──► [Generator] ──► 答案 + 来源
```

- **Loader** (`rag/loader.py`)：PDF 逐页解析（保留页码），Markdown 按标题记录 heading。
- **Splitter** (`rag/splitter.py`)：`RecursiveCharacterSplitter`，`chunk_size=800 / overlap=150`，按 `["\n\n","\n","。",". "," ",""]` 递归切分，保留 heading/page_number，输出 chunk_id。
- **Embedding** (`rag/embedding.py`)：`OpenAIEmbedding` / `BGEEmbedding` / `HashEmbedding`，工厂按 `EMBEDDING_BACKEND` 选择并自动降级。
- **VectorStore** (`rag/vectorstore.py`)：`ChromaStore`（持久化）/ `InMemoryStore`（余弦），统一 `add/query/count/reset` 接口。
- **Retriever** (`rag/retriever.py`)：`similarity_search(query, top_k)`，返回带 score 候选。
- **Reranker** (`rag/reranker.py`)：`CrossEncoderReranker` / `LexicalReranker`；`rerank(query, docs, top_n)` 与 `rerank_compare()`（返回重排前后顺序对比）。
- **Context Compression** (`rag/generator.py`)：按 rerank 分数截断/去冗余。
- **Generator** (`rag/generator.py`)：System Prompt 强约束（只依据上下文 / 禁编造 / 必引用 / 信息不足拒答），支持 `generate` 与 `generate_stream`。

---

## 5. LangGraph Agent 流程

Agent 使用 **StateGraph**（`agent/graph.py`），状态定义见 `agent/state.py`（`AgentState` TypedDict）。

```
                         ┌─────────────┐
        START ─────────► │intent_router│
                         └──────┬──────┘
              need_tool         │          need_rag
        ┌───────────────────────┴───────────────────────┐
        ▼                                                ▼
   ┌─────────┐                                    ┌───────────┐
   │  tool   │                                    │  rewrite  │  (Query Rewrite / HyDE)
   └────┬────┘                                    └─────┬─────┘
        │                                               ▼
        │                                        ┌───────────┐
        │                                        │ retrieve  │  (top_k=20)
        │                                        └─────┬─────┘
        │                                               ▼
        │                                        ┌───────────┐
        │                                        │  rerank   │  (top_n=5 + confidence)
        │                                        └─────┬─────┘
        │                          confidence<阈值 或无上下文 │ 否则
        │                              ┌───────────────────┴──────────┐
        │                              ▼                              ▼
        │                        ┌──────────┐                  ┌───────────┐
        └───────────────────────►│ generate │◄─────────────────┤  (直接生成) │
                                 └────┬─────┘                  └───────────┘
                                      ▼                        ┌──────────┐
                                 ┌──────────┐   FAIL(重生成一次) │  reject  │ (固定拒答)
                                 │ critique │───────────────┐  └────┬─────┘
                                 └────┬─────┘               │       ▼
                                 PASS │                     └──►generate  END
                                      ▼
                                     END
```

- **intent_router** (`agent/router.py`)：规则优先 + LLM 兜底，判定 `intent / need_rag / need_tool / tool_name`（纯算术→calculator，日期时间关键词→datetime，"最新/实时/新闻"→web_search，文件→file_query，其余→RAG）。
- **节点** (`agent/nodes.py`)：`rewrite_node` / `retrieve_node` / `rerank_node` / `tool_node` / `generate_node` / `critique_node` / `reject_node`，每个节点向 `trace` 追加 `{step, summary, tool}`（仅 reasoning summary，不暴露完整思维链）。
- **条件边**：`_route_after_router`、`_route_after_rerank`（拒答控制）、`_route_after_critique`（Self-Critique 失败重生成一次）。
- **memory** (`agent/memory.py`)：轻量会话记忆，供 Query Rewrite 指代消解。

---

## 6. Tool Calling 说明

工具位于 `backend/tools/`，通过 `registry.py` 统一注册（`SCHEMA` + `run()` 模式）:

| 工具 | 说明 | 触发示例 |
| --- | --- | --- |
| `calculator` | 安全 AST 表达式求值（禁 `eval`，仅算术） | `12345 * 678`、`(3+4)^2` |
| `web_search` | 网络搜索封装（Tavily/SerpAPI），无 key 返回确定性 mock | "最新的 AI 新闻" |
| `file_query` | 按文件名/关键词查询已入库 chunk | "在 xxx.pdf 里查 …" |
| `datetime_tool` | 当前时间 / 日期计算 | "今天几号"、"距离 X 还有几天" |

调用链路:`intent_router` 判定 `need_tool` 与 `tool_name` → `tool_node` 用 LLM/规则生成参数 → `registry.execute(name, args)` → 结果写入 `tool_results` → `generate_node` 基于结果作答。`registry.list_schemas()` 将工具 JSON schema 暴露给 LLM 供语义选择。

---

## 7. 普通函数 vs Tool Calling

| 维度 | 普通函数调用 | Tool Calling（本项目） |
| --- | --- | --- |
| 谁决定调用 | 开发者在代码里**硬编码** if/else 显式调用 | **LLM 根据语义自主决定**是否调用、调用哪个 |
| 参数来源 | 代码里写死或规则提取 | **LLM 从自然语言生成结构化参数**（对齐 JSON schema） |
| 扩展性 | 新增能力需改分支逻辑 | 在 `registry` 注册 schema 即可，Agent 自动感知 |
| 适用场景 | 流程固定、输入结构化 | 意图多样、输入自然语言、需要动态规划 |
| 本项目体现 | — | `intent_router` 语义路由 + `tool_node` 参数生成 + `registry` schema |

一句话:**普通函数是"程序员决定何时调用"，Tool Calling 是"模型决定何时、用什么参数调用"**，后者让 Agent 具备自主规划能力。

---

## 8. Advanced RAG

实现于 `rag/advanced.py`，由 Agent 节点调用:

- **Query Rewrite**：结合会话历史做指代消解与查询改写，提升多轮检索命中。
- **HyDE (Hypothetical Document Embeddings)**：先让 LLM 生成"假设性答案"，再用其向量检索，缓解 query 与文档表述差异。
- **Agentic RAG**：Agent 通过条件路由自主决定是否检索、是否改写、是否重生成（Self-Critique 回路），而非固定 Chain。

---

## 9. 安装与运行（本地）

### 后端

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # Linux / macOS
pip install -r requirements.txt

# 可选：复制并编辑环境变量（不填也能离线跑）
copy .env.example .env

uvicorn main:app --reload --port 8000
```

启动后访问:
- API 文档（Swagger）: http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/health

> 首次启动会自动把 `backend/data/seed_docs/` 的示例文档入库，Demo/评估开箱即用。

### 前端

```powershell
cd frontend
npm install
npm run dev          # http://localhost:3000 （已配置 /api 代理到 8000）
```

---

## 10. Docker 部署

一键启动（后端 + 前端 nginx）:

```bash
docker compose up --build
```

- 前端: http://localhost:3000
- 后端: http://localhost:8000/docs

默认离线模式运行。若要接入真实模型，在启动前设置环境变量（或写入根目录 `.env`）:

```bash
OPENAI_API_KEY=sk-xxx EMBEDDING_BACKEND=openai docker compose up --build
```

> 说明:后端使用 **内嵌 Chroma PersistentClient**，数据通过命名卷 `backend_data` 持久化（挂载到 `/app/data`）。

---

## 11. 环境变量配置

见 `backend/.env.example`。关键项:

| 变量 | 默认                                     | 说明 |
| --- |----------------------------------------| --- |
| `MODEL_NAME` | ` glm-4.5-air`                         | LLM 模型名 |
| `OPENAI_API_KEY` | 空                                      | **留空即离线 mock 模式** |
| `OPENAI_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` | 兼容 API 地址 |
| `EMBEDDING_BACKEND` | `openai`                               | `openai` / `bge` / `hash` |
| `RERANKER_BACKEND` | `lexical`                              | `cross-encoder` / `lexical` |
| `VECTOR_BACKEND` | `chroma`                               | `chroma` / `memory` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150`                          | 切分参数 |
| `TOP_K` / `RERANK_TOP_N` | `20` / `5`                             | 检索/重排数量 |
| `CONFIDENCE_THRESHOLD` | `0.30`                                 | 拒答置信度阈值 |
| `WEB_SEARCH_PROVIDER` | `mock`                                 | `tavily` / `serpapi` / `mock` |

启用本地 BGE 模型需在 `requirements.txt` 取消注释 `sentence-transformers` 与 `torch` 并安装。

---

## 12. API 文档

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/upload` | 上传文档（multipart），解析→切分→向量化→入库，返回 `{filename,pages,chunks,status}` |
| `POST` | `/api/chat` | 提问，返回 `{answer,sources,confidence,tools,trace,intent}` |
| `POST` | `/api/chat/stream` | SSE 流式：逐 token `data:{type:"token"}`，末尾 `data:{type:"done",...}` |
| `GET` | `/api/documents` | 知识库文件列表（pages/chunks/status） |
| `GET` | `/api/evaluation` | 返回最近一次评估指标 |
| `POST` | `/api/evaluation/run` | 运行评估集并生成 `evaluation_report.md` |
| `GET` | `/api/health` | 健康检查（含 LLM 模式、向量数） |

**Chat 请求示例**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"RAG 的优势是什么?","session_id":"default"}'
```

---

## 13. Demo 说明

平台在 **离线兜底模式** 下即可演示以下 4 个场景:

- **Demo 1 · RAG 问答带来源**：上传 PDF/MD 后提问"RAG 的优势是什么?"，返回答案 + Sources 引用卡片（文件名/页码/score）。
- **Demo 2 · 实时信息 → web_search**：提问"最新的 AI 新闻"，Agent 路由到 `web_search` 工具，Trace 显示工具调用。
- **Demo 3 · 数学 → calculator**：输入 `12345 * 678`，Agent 路由到 `calculator`，精确计算。
- **Demo 4 · 库中无答案 → 拒答**：提问知识库中不存在的问题，置信度低于阈值触发固定拒答语"知识库中没有足够信息回答该问题。"

---

## 14. 评估（Evaluation）

测试集 `backend/data/eval/testset.json` 共 **20 条**，4 类各 5 条:`knowledge`（库中已有）/ `multi_hop`（跨文档推理）/ `no_answer`（应拒答）/ `tool`（应调用工具）。

指标（`rag/evaluator.py`）:
- **检索**：Precision@K / Recall@K（基于 `expected_doc_ids`）
- **生成**：Answer Relevance / Context Relevance / Faithfulness（LLM-as-judge；离线时用词法重叠启发式）
- **安全**：Hallucination Rate

运行:

```bash
curl -X POST http://localhost:8000/api/evaluation/run
```

或在前端 Evaluation 页点击"运行评估"。结果写入 `backend/evaluation_report.md`。

> ⚠️ 注意:仓库内 `evaluation_report.md` 的数值是在 **离线 mock 模式**（hash 向量 + 词法启发式判分）下生成的基线，指标偏低属预期;接入真实 embedding/LLM 后会显著提升。评估**框架与流程**是完整可用的。

---

## 15. Failure Case 分析

详见 [`failure_cases.md`](failure_cases.md)。摘要:
- **Case 1 · 检索错误**：query 与文档表述差异导致 Top-K 缺失 → 引入 **Query Rewrite / HyDE**。
- **Case 2 · 幻觉**：无充分上下文仍强行作答 → **Confidence Threshold + 拒答机制 + Self-Critique**。
- **Case 3 · 复杂问题**：单轮检索不足以覆盖多跳推理 → **Agentic RAG（条件路由 + 重生成回路）**。

---

## 16. 测试

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

覆盖:loader / splitter / embedding / retriever / reranker / router / tools / agent / api（TestClient）。**全部在离线兜底模式下可稳定跑通（35 passing）。**

---

## 17. 目录结构

```
RAG_ToolCalling_Agent/
├── backend/
│   ├── app/         # config, db, models, schemas, logging, llm
│   ├── rag/         # loader, splitter, embedding, vectorstore, retriever,
│   │                # reranker, generator, advanced, evaluator, ingest
│   ├── agent/       # state, router, nodes, graph, memory (LangGraph)
│   ├── tools/       # calculator, web_search, file_query, datetime, registry
│   ├── api/         # upload, chat, evaluation, documents
│   ├── data/        # seed_docs, eval/testset.json, chroma, uploads
│   ├── tests/       # pytest (35)
│   ├── main.py      # FastAPI entrypoint
│   ├── requirements.txt / .env.example / Dockerfile
│   └── evaluation_report.md
├── frontend/        # React + TS + Vite + AntD + Zustand
│   ├── src/pages/   # ChatPage / TracePage / KnowledgeBasePage / EvaluationPage
│   ├── src/stores/  # zustand chatStore
│   ├── src/api/     # axios client + SSE streamChat
│   ├── Dockerfile / nginx.conf
├── docker-compose.yml
├── failure_cases.md
└── README.md
```
