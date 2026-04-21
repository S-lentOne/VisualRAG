"""
rag/

Retrieval-Augmented Generation module for the object detection pipeline.

Public API
----------
    from rag import Embedder, KnowledgeBase, Retriever
    from rag import build_query, build_query_from_detections
    from rag import RetrievedChunk, QueryContext

Typical usage
-------------
    # 1. Build the knowledge base (run once, or when new documents are added)
    kb = KnowledgeBase(db_path="data/chroma_db")
    kb.build()

    # 2. Create a retriever
    retriever = Retriever(kb)

    # 3. At inference time, given detection results:
    detections = [{"label": "laptop", "score": 0.94}, {"label": "cup", "score": 0.72}]
    ctx = build_query_from_detections(detections)
    chunks = retriever.retrieve(ctx.query, top_k=5)
    context_str = retriever.format_context(chunks)

Adding new object classes
-------------------------
    1. Create  rag/documents/<new_class>.txt  following the existing format.
    2. Run  kb.build()  (or  kb.build(force_rebuild=False)  to upsert only).
    3. Add any relevant scene hints to query_builder._SCENE_HINTS  (optional).
    No other code changes are required.
"""

from rag.embedder import Embedder
from rag.knowledge_base import KnowledgeBase
from rag.retriever import Retriever, RetrievedChunk
from rag.query_builder import (
    build_query,
    build_query_from_detections,
    QueryContext,
)

__all__ = [
    "Embedder",
    "KnowledgeBase",
    "Retriever",
    "RetrievedChunk",
    "build_query",
    "build_query_from_detections",
    "QueryContext",
]