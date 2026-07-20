"""RAG pipeline package.

Each stage of the pipeline lives in its own module so the pieces can be reused
and tested independently:

    loader -> splitter -> embedding -> vectorstore -> retriever -> reranker
           -> generator (with context compression) -> evaluator

``advanced`` implements Advanced-RAG techniques (Query Rewrite / HyDE) used by
the Agentic RAG workflow.
"""
