"""
rag/knowledge_base.py

Reads all .txt documents from the documents/ directory, chunks them,
embeds each chunk, and persists them into a ChromaDB collection.

Adding new classes: drop a new <classname>.txt into rag/documents/ and
re-run build(). The new class is picked up automatically — no code changes.
"""

import os
import hashlib
import chromadb
from chromadb.config import Settings
from typing import Optional

from rag.embedder import Embedder


_DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
_COLLECTION_NAME = "object_knowledge"


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Args:
        text:       Full document text.
        chunk_size: Target number of words per chunk.
        overlap:    Number of words to repeat at the start of the next chunk.

    Returns:
        List of text chunk strings.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def _make_chunk_id(doc_name: str, chunk_index: int, chunk_text: str) -> str:
    """
    Generate a stable, unique ID for a chunk.
    If the same document is re-ingested unchanged, IDs are identical —
    ChromaDB will upsert without duplicating.
    """
    content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
    return f"{doc_name}__chunk{chunk_index}__{content_hash}"


def _parse_class_label(text: str) -> str:
    """
    Extract the class name from the first line of a document.
    Expected format:  class: laptop
    Falls back to the raw first line if format is unexpected.
    """
    first_line = text.strip().splitlines()[0]
    if ":" in first_line:
        return first_line.split(":", 1)[1].strip()
    return first_line.strip()


class KnowledgeBase:
    """
    Manages the ChromaDB vector store for object knowledge documents.

    Usage:
        kb = KnowledgeBase(db_path="data/chroma_db")
        kb.build()          # ingest/update all documents
        # (retrieval is handled by Retriever, which takes a KnowledgeBase)
    """

    def __init__(
        self,
        db_path: str = "data/chroma_db",
        embedder: Optional[Embedder] = None,
        chunk_size: int = 80,
        chunk_overlap: int = 15,
        documents_dir: str = _DOCUMENTS_DIR,
    ):
        """
        Args:
            db_path:        Path where ChromaDB persists data on disk.
            embedder:       Embedder instance. Created automatically if not provided.
            chunk_size:     Words per chunk (default 80 — suitable for ~300-word docs).
            chunk_overlap:  Word overlap between adjacent chunks (helps retrieval continuity).
            documents_dir:  Path to the folder containing .txt knowledge documents.
        """
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents_dir = documents_dir
        self.embedder = embedder or Embedder()

        os.makedirs(db_path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def build(self, force_rebuild: bool = False) -> int:
        """
        Ingest all .txt files from the documents directory into ChromaDB.
        Uses upsert so re-running is safe — unchanged chunks are not duplicated.

        Args:
            force_rebuild: If True, deletes and recreates the collection first.

        Returns:
            Total number of chunks ingested.
        """
        if force_rebuild:
            print("[KnowledgeBase] Force rebuild — deleting existing collection.")
            self.client.delete_collection(_COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        doc_files = sorted(
            f for f in os.listdir(self.documents_dir) if f.endswith(".txt")
        )

        if not doc_files:
            print(f"[KnowledgeBase] Warning: no .txt files found in {self.documents_dir}")
            return 0

        print(f"[KnowledgeBase] Ingesting {len(doc_files)} document(s)...")
        total_chunks = 0

        for filename in doc_files:
            doc_name = filename.replace(".txt", "")
            filepath = os.path.join(self.documents_dir, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            class_label = _parse_class_label(text)
            chunks = _chunk_text(text, self.chunk_size, self.chunk_overlap)

            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                chunk_id = _make_chunk_id(doc_name, i, chunk)
                embedding = self.embedder.embed(chunk).tolist()

                ids.append(chunk_id)
                embeddings.append(embedding)
                documents.append(chunk)
                metadatas.append({
                    "class_label": class_label,
                    "doc_name": doc_name,
                    "chunk_index": i,
                    "source_file": filename,
                })

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            total_chunks += len(chunks)
            print(f"  [{doc_name}] {len(chunks)} chunk(s) ingested.")

        print(f"[KnowledgeBase] Done. Total chunks in store: {total_chunks}")
        return total_chunks

    def get_collection(self):
        """Return the underlying ChromaDB collection (used by Retriever)."""
        return self.collection

    def list_classes(self) -> list[str]:
        """
        Return a sorted list of all class labels currently in the knowledge base.
        Useful for verifying coverage after adding new documents.
        """
        results = self.collection.get(include=["metadatas"])
        labels = {m["class_label"] for m in results["metadatas"]}
        return sorted(labels)

    def stats(self) -> dict:
        """Return basic stats about the knowledge base."""
        count = self.collection.count()
        classes = self.list_classes()
        return {
            "total_chunks": count,
            "total_classes": len(classes),
            "classes": classes,
        }