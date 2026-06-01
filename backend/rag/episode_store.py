import ast
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings

from rag.embedder import Embedder

_EPISODE_COLLECTION = "scene_episodes"


@dataclass
class Episode:
    episode_id: str
    timestamp: float
    labels: list
    scene_text: str
    scores: dict
    score: float


def _labels_to_scene_text(labels: list, scores: dict) -> str:
    parts = sorted(scores.items(), key=lambda x: -x[1])
    desc = ", ".join(f"{l} ({s:.2f})" for l, s in parts if l in labels)
    return f"{desc} observed together"


class EpisodeStore:
    def __init__(
        self,
        db_path: str = "data/chroma_db",
        embedder: Optional[Embedder] = None,
        max_episodes: int = 500,
        # shared client so EpisodeStore and KnowledgeBase don't conflict
        chroma_client=None,
    ):
        self.db_path = db_path
        self.max_episodes = max_episodes
        self.embedder = embedder or Embedder()

        self.client = chroma_client or chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=_EPISODE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def record(
        self,
        labels: list,
        scores: Optional[dict] = None,
        timestamp: Optional[float] = None,
        scene_summary: Optional[str] = None,
    ) -> str:
        if not labels:
            return ""

        scores = scores or {l: 1.0 for l in labels}
        timestamp = timestamp or time.time()
        episode_id = str(uuid.uuid4())

        scene_text = scene_summary or _labels_to_scene_text(labels, scores)
        embedding = self.embedder.embed(scene_text).tolist()

        self.collection.add(
            ids=[episode_id],
            embeddings=[embedding],
            documents=[scene_text],
            metadatas=[
                {
                    "timestamp": float(timestamp),
                    "labels": ",".join(sorted(labels)),
                    "scores": str(scores),
                    "has_summary": int(scene_summary is not None),
                }
            ],
        )
        self._prune_if_needed()
        return episode_id

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
        since: Optional[float] = None,
        exclude_last_n: int = 1,
    ) -> list:
        total = self.collection.count()
        if total == 0:
            return []

        fetch_k = min(top_k + exclude_last_n + 5, total)
        query_embedding = self.embedder.embed_query(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        recent_timestamps = sorted([m["timestamp"] for m in all_meta], reverse=True)[
            :exclude_last_n
        ]
        exclude_ts = set(recent_timestamps)
        episodes = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            ts = meta["timestamp"]
            if ts in exclude_ts:
                continue
            if since and ts < since:
                continue

            score = round(1.0 - distance, 4)
            if score < min_score:
                continue

            labels = [l for l in meta["labels"].split(",") if l]
            try:
                label_scores = ast.literal_eval(meta["scores"])
            except Exception:
                label_scores = {}

            episodes.append(
                Episode(
                    episode_id="",
                    timestamp=ts,
                    labels=labels,
                    scene_text=text,
                    scores=label_scores,
                    score=score,
                )
            )

        episodes.sort(key=lambda e: e.score, reverse=True)
        return episodes[:top_k]

    def format_context(self, episodes: list) -> str:
        if not episodes:
            return ""
        now = time.time()
        lines = [
            f"- {_human_age(now - ep.timestamp)}: {ep.scene_text}" for ep in episodes
        ]
        return "\n".join(lines)

    def get_recent(self, n: int = 5) -> list:
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["documents", "metadatas"])
        paired = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda x: x[1]["timestamp"],
            reverse=True,
        )

        episodes = []
        for text, meta in paired[:n]:
            try:
                label_scores = eval(meta["scores"])
            except Exception:
                label_scores = {}
            episodes.append(
                Episode(
                    episode_id="",
                    timestamp=meta["timestamp"],
                    labels=[l for l in meta["labels"].split(",") if l],
                    scene_text=text,
                    scores=label_scores,
                    score=0.0,
                )
            )
        return episodes

    def detect_changes(self, current_labels: list, lookback: int = 3) -> dict:
        recent = self.get_recent(lookback)
        if not recent:
            return {"appeared": list(current_labels), "disappeared": [], "stable": []}

        past_labels = set()
        for ep in recent:
            past_labels.update(ep.labels)

        current_set = set(current_labels)
        return {
            "appeared": sorted(current_set - past_labels),
            "disappeared": sorted(past_labels - current_set),
            "stable": sorted(current_set & past_labels),
        }

    def clear(self):
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)
        print(f"[EpisodeStore] Cleared {len(ids)} episodes.")

    def stats(self) -> dict:
        total = self.collection.count()
        recent = self.get_recent(1)
        last_ts = recent[0].timestamp if recent else None
        return {
            "total_episodes": total,
            "last_recorded": last_ts,
            "last_recorded_human": (_human_age(time.time() - last_ts) + " ago")
            if last_ts
            else "never",
        }

    def _prune_if_needed(self):
        total = self.collection.count()
        if total <= self.max_episodes:
            return
        all_data = self.collection.get(include=["metadatas"])
        paired = sorted(
            zip(all_data["ids"], all_data["metadatas"]), key=lambda x: x[1]["timestamp"]
        )
        ids_to_delete = [id_ for id_, _ in paired[: total - self.max_episodes]]
        self.collection.delete(ids=ids_to_delete)


def _human_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s ago"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    elif seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    else:
        return f"{int(seconds // 86400)}d ago"
