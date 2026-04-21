"""
rag/retriever.py

Queries the ChromaDB knowledge base to retrieve the most relevant
context chunks for a given semantic query.

Adding new classes: no changes needed — the retriever queries by
semantic similarity, not by hardcoded class names.
"""

from dataclasses import dataclass
from typing import Optional

from rag.knowledge_base import KnowledgeBase
from rag.embedder import Embedder


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its metadata and similarity score."""
    text: str
    class_label: str
    doc_name: str
    chunk_index: int
    score: float          # cosine similarity (0–1, higher = more relevant)
    source_file: str


class Retriever:
    """
    Retrieves the top-k most relevant knowledge chunks for a query.

    Usage:
        retriever = Retriever(knowledge_base)
        chunks = retriever.retrieve("laptop on a desk with coffee cup", top_k=4)
        context = retriever.format_context(chunks)
    """

    def __init__(self, knowledge_base: KnowledgeBase, embedder: Optional[Embedder] = None):
        """
        Args:
            knowledge_base: A built KnowledgeBase instance.
            embedder:       Embedder to use for query embedding.
                            Reuses the KB's embedder by default to avoid loading twice.
        """
        self.collection = knowledge_base.get_collection()
        self.embedder = embedder or knowledge_base.embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_classes: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks from the knowledge base.

        Args:
            query:          The semantic search string.
            top_k:          Maximum number of chunks to return.
            min_score:      Minimum similarity threshold (0–1). Chunks below
                            this score are filtered out.
            filter_classes: If provided, restrict results to these class labels only.
                            Useful when you want to retrieve only specific classes.

        Returns:
            List of RetrievedChunk objects, sorted by descending relevance score.
        """
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
        """
        Retrieve the best chunk for each detected class individually.
        This guarantees at least one relevant chunk per detected object,
        even if some classes score lower in a broad semantic search.

        Args:
            query:            The semantic search string (shared across all classes).
            classes:          List of class label strings to retrieve for.
            chunks_per_class: Number of best chunks to fetch per class.

        Returns:
            Flat list of RetrievedChunk objects, one or more per class,
            sorted by descending score.
        """
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
        """
        Format retrieved chunks into a clean context string for the LLM prompt.

        Args:
            chunks: List of RetrievedChunk objects.
            dedupe: If True, skip duplicate chunks (same doc_name + chunk_index).

        Returns:
            Formatted multi-line string with labelled sections per class.
        """
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