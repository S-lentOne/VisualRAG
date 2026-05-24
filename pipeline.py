import time
import cv2
import sys
from dataclasses import dataclass, field
from typing import Optional, Callable
from ultralytics import YOLO
import chromadb
from chromadb.config import Settings

from rag import KnowledgeBase, Retriever, EpisodeStore, build_query_from_detections
from rag.embedder import Embedder
from llm import OllamaClient, build_prompt
from llm.prompt_builder import build_followup_prompt

DEFAULT_CONFIG = {
    "model_path":         "models/tr-9.pt",
    # set to a COCO yolo model path (e.g. "models/yolov8n.pt") to enable dual detection
    # download with: from ultralytics import YOLO; YOLO("yolov8n.pt")
    "coco_model_path":    None,
    "confidence":         0.25,
    "db_path":            "data/chroma_db",
    "static_top_k":       4,
    "episodic_top_k":     3,
    "episodic_min_score": 0.3,
    "max_episodes":       500,
    "ollama_model":       "llama3.2:3b",
    "ollama_host":        "http://localhost:11434",
    "stream":             False,
    "frame_interval":     15,
    "show_cv_window":     True,
}

# COCO classes that overlap well with our 30 — used to merge dual-model results cleanly
_COCO_TO_OURS = {
    "laptop":       "laptop",
    "keyboard":     "keyboard",
    "mouse":        "mouse",
    "tv":           "monitor",
    "monitor":      "monitor",
    "cell phone":   "phone",
    "backpack":     "backpack",
    "handbag":      "backpack",
    "cup":          "cup",
    "bottle":       "bottle",
    "chair":        "chair",
    "book":         "notebook",
    "clock":        "watch",
    "scissors":     "ruler",
    "remote":       "controller",
}


@dataclass
class PipelineResult:
    detections:       list
    scene_query:      str
    static_context:   str
    episodic_context: str
    llm_response:     str
    changes:          dict
    timestamp:        float = field(default_factory=time.time)
    episode_id:       str = ""


class Pipeline:
    def __init__(self, config: Optional[dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model        = None
        self.coco_model   = None
        self.retriever    = None
        self.episode_store = None
        self.llm          = None
        self._ready       = False

    def setup(self):
        cfg = self.config

        print("[Pipeline] Loading YOLO model...")
        self.model = YOLO(cfg["model_path"])

        if cfg["coco_model_path"]:
            print(f"[Pipeline] Loading COCO model: {cfg['coco_model_path']}")
            self.coco_model = YOLO(cfg["coco_model_path"])
            print("[Pipeline] Dual-model detection enabled.")

        print("[Pipeline] Initializing RAG...")
        embedder = Embedder()

        # single shared ChromaDB client — fixes cross-session persistence bug
        chroma_client = chromadb.PersistentClient(
            path=cfg["db_path"],
            settings=Settings(anonymized_telemetry=False),
        )

        kb = KnowledgeBase(db_path=cfg["db_path"], embedder=embedder, chroma_client=chroma_client)
        kb.build()
        self.retriever = Retriever(kb, embedder=embedder)
        self.episode_store = EpisodeStore(
            db_path=cfg["db_path"],
            embedder=embedder,
            max_episodes=cfg["max_episodes"],
            chroma_client=chroma_client,
        )

        print("[Pipeline] Connecting to Ollama...")
        self.llm = OllamaClient(model=cfg["ollama_model"], host=cfg["ollama_host"])
        if not self.llm.is_reachable():
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve\n"
                f"Then pull a model: ollama pull {cfg['ollama_model']}"
            )

        self._ready = True
        ep_count = self.episode_store.stats()["total_episodes"]
        print(f"[Pipeline] Ready. ({ep_count} episodes in memory)\n")

    def run_image(self, path: str, on_result: Optional[Callable] = None) -> PipelineResult:
        self._check_ready()
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        detections = self._detect(frame)
        result = self._analyze(detections)
        if self.config["show_cv_window"]:
            cv2.imshow("Image", self._annotate_frame(frame, result))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if on_result:
            on_result(result)
        return result

    def run_video(self, path: str) -> list:
        """
        Two-pass video processing:
          Pass 1 — YOLO only, reads every frame at full speed, collects sampled detections
          Pass 2 — LLM analysis on the collected detection snapshots
        This prevents Ollama's response time from skewing which frames get analyzed.
        """
        self._check_ready()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        print(f"[Video] {total_frames} frames, {fps:.0f}fps, ~{duration:.1f}s")
        print(f"[Video] Pass 1: running YOLO on every {self.config['frame_interval']} frames...")

        # pass 1 — pure detection, no LLM, no blocking
        snapshots = []  # list of (timestamp, detections, frame)
        frame_count = 0
        last_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            last_frame = frame

            if frame_count % self.config["frame_interval"] == 0:
                t = frame_count / fps
                detections = self._detect(frame)
                snapshots.append((t, detections, frame.copy()))
                labels = [d["label"] for d in detections]
                print(f"  t={t:.1f}s  detected={labels or ['nothing']}")

        cap.release()
        print(f"[Video] Pass 1 done. {len(snapshots)} snapshots collected.")
        print(f"[Video] Pass 2: running RAG + LLM on each snapshot...\n")

        # pass 2 — RAG + LLM on each snapshot sequentially
        results = []
        last_result = None
        for i, (t, detections, frame) in enumerate(snapshots):
            result = self._analyze(detections)
            results.append(result)
            last_result = result

            labels = [d["label"] for d in result.detections]
            changes = []
            if result.changes["appeared"]:
                changes.append(f"+{','.join(result.changes['appeared'])}")
            if result.changes["disappeared"]:
                changes.append(f"-{','.join(result.changes['disappeared'])}")
            change_str = f"  [{' '.join(changes)}]" if changes else ""
            print(f"  [{i+1}/{len(snapshots)}] t={t:.1f}s  {labels or ['nothing']}{change_str}")
            print(f"  └─ {result.llm_response}\n")

            if self.config["show_cv_window"]:
                display = self._annotate_frame(frame, result)
                cv2.imshow("Video — VisualRAG", display)
                cv2.waitKey(1)

        cv2.destroyAllWindows()
        print(f"[Video] Done. {len(results)} scenes analyzed.")
        return results

    def run_live(self, camera_index: int = 0, on_result: Optional[Callable] = None):
        self._check_ready()
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(camera_index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {camera_index}")
        self._run_capture_loop(cap, "Live", on_result)
        cap.release()

    def followup(self, result: PipelineResult, question: str) -> str:
        self._check_ready()
        pkg = build_prompt(
            detections=result.detections,
            static_context=result.static_context,
            episodic_context=result.episodic_context,
        )
        messages = build_followup_prompt(pkg.messages, question)
        return self.llm.chat(messages)

    def _run_capture_loop(self, cap, source_label: str, on_result: Optional[Callable]):
        cfg = self.config
        frame_count = 0
        last_result = None

        print(f"[Pipeline] {source_label} running. Press Q to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            if frame_count % cfg["frame_interval"] == 0:
                detections = self._detect(frame)
                last_result = self._analyze(detections)
                if on_result:
                    on_result(last_result)

            if cfg["show_cv_window"]:
                display = self._annotate_frame(frame, last_result) if last_result else frame
                cv2.imshow(f"Pipeline — {source_label}", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def _detect(self, frame) -> list:
        detections = self._run_model(self.model, frame)

        # merge COCO detections if second model is loaded
        if self.coco_model:
            coco_raw = self._run_model(self.coco_model, frame)
            existing_labels = {d["label"] for d in detections}
            for d in coco_raw:
                mapped = _COCO_TO_OURS.get(d["label"])
                # add COCO detection only if it maps to one of our classes and isn't already detected
                if mapped and mapped not in existing_labels:
                    d["label"] = mapped
                    detections.append(d)
                    existing_labels.add(mapped)

        detections.sort(key=lambda d: -d["score"])
        return detections

    def _run_model(self, model, frame) -> list:
        results = model(frame, conf=self.config["confidence"])[0]
        out = []
        for box in results.boxes:
            label = model.names[int(box.cls)]
            score = float(box.conf)
            bbox  = [round(float(x), 3) for x in box.xyxyn[0].tolist()]
            out.append({"label": label, "score": score, "bbox": bbox})
        return out

    def _analyze(self, detections: list) -> PipelineResult:
        cfg = self.config
        ctx = build_query_from_detections(detections)

        static_chunks = self.retriever.retrieve(ctx.query, top_k=cfg["static_top_k"])
        static_context = self.retriever.format_context(static_chunks)

        episodes = self.episode_store.retrieve(
            ctx.query,
            top_k=cfg["episodic_top_k"],
            min_score=cfg["episodic_min_score"],
        )
        episodic_context = self.episode_store.format_context(episodes)
        changes = self.episode_store.detect_changes([d["label"] for d in detections])

        if changes["appeared"] or changes["disappeared"]:
            episodic_context = (_changes_to_text(changes) + "\n" + episodic_context).strip()

        pkg = build_prompt(
            detections=detections,
            static_context=static_context,
            episodic_context=episodic_context,
        )

        response = self._stream_response(pkg.messages) if cfg["stream"] else self.llm.chat(pkg.messages)

        episode_id = self.episode_store.record(
            labels=[d["label"] for d in detections],
            scores={d["label"]: d["score"] for d in detections},
            scene_summary=response,
        )

        return PipelineResult(
            detections=detections,
            scene_query=ctx.query,
            static_context=static_context,
            episodic_context=episodic_context,
            llm_response=response,
            changes=changes,
            episode_id=episode_id,
        )

    def _stream_response(self, messages: list) -> str:
        print("[LLM] ", end="", flush=True)
        tokens = []
        for token in self.llm.chat_stream(messages):
            print(token, end="", flush=True)
            tokens.append(token)
        print()
        return "".join(tokens)

    def _annotate_frame(self, frame, result: PipelineResult):
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for d in result.detections:
            x1 = int(d["bbox"][0] * w)
            y1 = int(d["bbox"][1] * h)
            x2 = int(d["bbox"][2] * w)
            y2 = int(d["bbox"][3] * h)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 100), 2)
            cv2.putText(annotated, f"{d['label']} {d['score']:.0%}",
                        (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 1)

        lines = _wrap_text(result.llm_response, max_chars=90)
        box_h = len(lines) * 20 + 12
        cv2.rectangle(annotated, (0, h - box_h - 4), (w, h), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(annotated, line, (8, h - box_h + i * 20 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

        return annotated

    def _check_ready(self):
        if not self._ready:
            raise RuntimeError("Call pipeline.setup() before running.")


def _changes_to_text(changes: dict) -> str:
    parts = []
    if changes["appeared"]:
        parts.append(f"Newly appeared: {', '.join(changes['appeared'])}")
    if changes["disappeared"]:
        parts.append(f"No longer visible: {', '.join(changes['disappeared'])}")
    return ". ".join(parts) + "."

def _wrap_text(text: str, max_chars: int) -> list:
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines