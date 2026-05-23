import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

# stored as a simple JSON file — one file per user_id
# when we move to a website, this file path becomes a DB row key
_DEFAULT_DIR = "data/user_profiles"


@dataclass
class UserProfile:
    user_id: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    session_count: int = 0
    # flat list of past Q&A pairs the user had with the system
    chat_history: list = field(default_factory=list)
    # summary of recurring objects/patterns seen for this user (updated each session)
    scene_summary: str = ""
    # labels the user's environment commonly contains — used to bias future retrieval
    frequent_labels: dict = field(default_factory=dict)  # label -> seen count


class UserStore:
    """
    Persists per-user chat history and scene patterns across sessions.
    In terminal mode, user_id is just a name you type at startup.
    When we move to a website, user_id comes from the auth system.
    """

    def __init__(self, store_dir: str = _DEFAULT_DIR):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def _path(self, user_id: str) -> str:
        # sanitize user_id so it's safe as a filename
        safe_id = "".join(c for c in user_id if c.isalnum() or c in "-_").lower()
        return os.path.join(self.store_dir, f"{safe_id}.json")

    def load(self, user_id: str) -> UserProfile:
        path = self._path(user_id)
        if not os.path.exists(path):
            return UserProfile(user_id=user_id)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return UserProfile(**data)

    def save(self, profile: UserProfile):
        path = self._path(profile.user_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, indent=2)

    def record_session_start(self, profile: UserProfile) -> UserProfile:
        profile.session_count += 1
        profile.last_seen = time.time()
        return profile

    def record_exchange(self, profile: UserProfile, question: str, answer: str) -> UserProfile:
        # keeps the last 50 exchanges to avoid unbounded growth
        profile.chat_history.append({
            "ts": time.time(),
            "q": question,
            "a": answer,
        })
        profile.chat_history = profile.chat_history[-50:]
        return profile

    def record_detections(self, profile: UserProfile, labels: list) -> UserProfile:
        # tally label frequencies so the profile reflects what's common in this user's space
        for label in labels:
            profile.frequent_labels[label] = profile.frequent_labels.get(label, 0) + 1
        return profile

    def get_context_for_llm(self, profile: UserProfile, max_history: int = 6) -> str:
        """
        Build a context string from user history to inject into the LLM prompt.
        Tells the model what it knows about this specific user.
        """
        if not profile.chat_history and not profile.frequent_labels:
            return ""

        lines = [f"User '{profile.user_id}' — session #{profile.session_count}"]

        if profile.frequent_labels:
            top = sorted(profile.frequent_labels.items(), key=lambda x: -x[1])[:6]
            lines.append("Common objects in their environment: " + ", ".join(l for l, _ in top))

        if profile.scene_summary:
            lines.append(f"Previous session summary: {profile.scene_summary}")

        if profile.chat_history:
            recent = profile.chat_history[-max_history:]
            lines.append("Recent conversation history:")
            for entry in recent:
                lines.append(f"  Q: {entry['q']}")
                lines.append(f"  A: {entry['a'][:120]}{'...' if len(entry['a']) > 120 else ''}")

        return "\n".join(lines)

    def update_scene_summary(self, profile: UserProfile, summary: str) -> UserProfile:
        # overwrite with latest LLM scene summary — used as "last known context" next session
        profile.scene_summary = summary
        return profile

    def list_users(self) -> list:
        return [f.replace(".json", "") for f in os.listdir(self.store_dir) if f.endswith(".json")]