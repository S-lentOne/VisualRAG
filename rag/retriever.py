from dataclasses import dataclass
from typing import Optional

from rag.knowledge_base import KnowledgeBase
from rag.embedder import Embedder

@dataclass
class RetrievedChunk:
    text: str
    class_label: str
    doc_name: str
    chunk_index: int
    score: float          # cosine similarity (0–1, higher = more relevant)
    source_file: str

class Retriever:
    def __init__(self, knowledge_base: KnowledgeBase, embedder: Optional[Embedder] = None):
        self.collection = knowledge_base.get_collection()
        self.embedder = embedder or knowledge_base.embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_classes: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query).tolist()

        where_clause = None
        if filter_classes:
            if len(filter_classes) == 1:
                where_clause = {"class_label": {"$eq": filter_classes[0]}}
            else:
                where_clause = {"class_label": {"$in": filter_classes}}

        query_kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where_clause:
            query_kwargs["where"] = where_clause

        results = self.collection.query(**query_kwargs)

        chunks = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance → similarity: similarity = 1 - distance
            score = round(1.0 - distance, 4)
            if score < min_score:
                continue

            chunks.append(RetrievedChunk(
                text=text,
                class_label=meta["class_label"],
                doc_name=meta["doc_name"],
                chunk_index=meta["chunk_index"],
                score=score,
                source_file=meta["source_file"],
            ))

        # Already sorted by ChromaDB (nearest first), but re-sort after filtering
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    def retrieve_per_class(
        self,
        query: str,
        classes: list[str],
        chunks_per_class: int = 1,
    ) -> list[RetrievedChunk]:
        all_chunks = []
        for class_label in classes:
            chunks = self.retrieve(
                query=query,
                top_k=chunks_per_class,
                filter_classes=[class_label],
            )
            all_chunks.extend(chunks)

        all_chunks.sort(key=lambda c: c.score, reverse=True)
        return all_chunks

    def format_context(self, chunks: list[RetrievedChunk], dedupe: bool = True) -> str:
        seen = set()
        sections = {}

        for chunk in chunks:
            key = (chunk.doc_name, chunk.chunk_index)
            if dedupe and key in seen:
                continue
            seen.add(key)

            label = chunk.class_label
            if label not in sections:
                sections[label] = []
            sections[label].append(chunk.text.strip())

        lines = []
        for label, texts in sections.items():
            lines.append(f"[{label.upper()}]")
            for text in texts:
                lines.append(text)
            lines.append("")   # blank line between sections

        return "\n".join(lines).strip()