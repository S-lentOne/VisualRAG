import time
import cv2
from dataclasses import dataclass, field
from typing import Optional, Callable
from ultralytics import YOLO

from rag import KnowledgeBase, Retriever, EpisodeStore, build_query_from_detections
from rag.embedder import Embedder
from llm import OllamaClient, build_prompt
from llm.prompt_builder import build_followup_prompt

# ── config defaults (override by passing a config dict to Pipeline.__init__) ──
DEFAULT_CONFIG = {
    "model_path":        "models/tr-9.pt",
    "confidence":        0.4,
    "db_path":           "data/chroma_db",
    "static_top_k":      4,    # chunks from knowledge base
    "episodic_top_k":    3,    # past episodes to retrieve
    "episodic_min_score": 0.3,
    "max_episodes":      500,
    "ollama_model":      "llama3.2:3b",
    "ollama_host":       "http://localhost:11434",
    "stream":            False,  # set True to print LLM tokens as they arrive
    "frame_interval":    30,     # analyze every Nth frame in video/live modes
    "show_cv_window":    True,
}


@dataclass
class PipelineResult:
    # everything produced in one inference cycle
    detections:       list[dict]       # [{"label": ..., "score": ..., "bbox": ...}]
    scene_query:      str
    static_context:   str
    episodic_context: str
    llm_response:     str
    changes:          dict             # appeared / disappeared / stable vs recent memory
    timestamp:        float = field(default_factory=time.time)
    episode_id:       str = ""        # ID of the episode recorded for this result


class Pipeline:
    """
    Connects all three phases:
      vision (YOLO) → RAG (static KB + episodic memory) → LLM (Ollama)

    Typical use:
        pipe = Pipeline()
        pipe.setup()
        result = pipe.run_image("photo.jpg")
        print(result.llm_response)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model        = None
        self.retriever    = None
        self.episode_store = None
        self.llm          = None
        self._ready       = False

    def setup(self):
        # load everything — call once before any run_* method
        cfg = self.config

        print("[Pipeline] Loading YOLO model...")
        self.model = YOLO(cfg["model_path"])

        print("[Pipeline] Initializing RAG...")
        embedder = Embedder()
        kb = KnowledgeBase(db_path=cfg["db_path"], embedder=embedder)
        kb.build()
        self.retriever = Retriever(kb, embedder=embedder)
        self.episode_store = EpisodeStore(
            db_path=cfg["db_path"],
            embedder=embedder,
            max_episodes=cfg["max_episodes"],
        )

        print("[Pipeline] Connecting to Ollama...")
        self.llm = OllamaClient(
            model=cfg["ollama_model"],
            host=cfg["ollama_host"],
        )

        self._ready = True
        print("[Pipeline] Ready.\n")

    # ── public run methods ────────────────────────────────────────────────────

    def run_image(self, path: str, on_result: Optional[Callable] = None) -> PipelineResult:
        self._check_ready()
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        detections = self._detect(frame)
        result = self._analyze(detections)
        if self.config["show_cv_window"]:
            self._show_frame(frame, result, window="Image")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if on_result:
            on_result(result)
        return result

    def run_video(self, path: str, on_result: Optional[Callable] = None):
        # on_result is called with a PipelineResult each time a frame is analyzed
        self._check_ready()
        cap = cv2.VideoCapture(path)
        self._run_capture_loop(cap, source_label="Video", on_result=on_result)
        cap.release()

    def run_live(self, camera_index: int = 0, on_result: Optional[Callable] = None):
        self._check_ready()
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._run_capture_loop(cap, source_label="Live", on_result=on_result)
        cap.release()

    def followup(self, result: PipelineResult, question: str) -> str:
        # ask the LLM a follow-up question about the last analyzed scene
        self._check_ready()
        pkg = build_prompt(
            detections=result.detections,
            static_context=result.static_context,
            episodic_context=result.episodic_context,
        )
        messages = build_followup_prompt(pkg.messages, question)
        return self.llm.chat(messages)

    # ── internals ─────────────────────────────────────────────────────────────

    def _run_capture_loop(
        self,
        cap: cv2.VideoCapture,
        source_label: str,
        on_result: Optional[Callable],
    ):
        cfg = self.config
        frame_count = 0
        last_result = None

        print(f"[Pipeline] {source_label} feed running. Press Q to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # run full analysis every Nth frame; show annotated frame every frame
            if frame_count % cfg["frame_interval"] == 0:
                detections = self._detect(frame)
                last_result = self._analyze(detections)
                if on_result:
                    on_result(last_result)

            if cfg["show_cv_window"] and last_result is not None:
                display = self._annotate_frame(frame, last_result)
                cv2.imshow(f"Pipeline — {source_label}", display)
            elif cfg["show_cv_window"]:
                cv2.imshow(f"Pipeline — {source_label}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def _detect(self, frame) -> list[dict]:
        # run YOLO and return a clean list of dicts
        results = self.model(frame, conf=self.config["confidence"])[0]
        detections = []
        for box in results.boxes:
            label = self.model.names[int(box.cls)]
            score = float(box.conf)
            bbox  = [round(float(x), 3) for x in box.xyxyn[0].tolist()]  # normalized
            detections.append({"label": label, "score": score, "bbox": bbox})
        detections.sort(key=lambda d: -d["score"])
        return detections

    def _analyze(self, detections: list[dict]) -> PipelineResult:
        cfg = self.config

        # build semantic query from detections
        ctx = build_query_from_detections(detections)

        # static KB retrieval
        static_chunks = self.retriever.retrieve(ctx.query, top_k=cfg["static_top_k"])
        static_context = self.retriever.format_context(static_chunks)

        # episodic memory retrieval + change detection
        episodes = self.episode_store.retrieve(
            ctx.query,
            top_k=cfg["episodic_top_k"],
            min_score=cfg["episodic_min_score"],
        )
        episodic_context = self.episode_store.format_context(episodes)
        changes = self.episode_store.detect_changes([d["label"] for d in detections])

        # optionally inject change summary into episodic context
        if changes["appeared"] or changes["disappeared"]:
            change_note = _changes_to_text(changes)
            episodic_context = (change_note + "\n" + episodic_context).strip()

        # LLM
        pkg = build_prompt(
            detections=detections,
            static_context=static_context,
            episodic_context=episodic_context,
        )

        if cfg["stream"]:
            response = self._stream_response(pkg.messages)
        else:
            response = self.llm.chat(pkg.messages)

        # record this scene into episodic memory with the LLM response as summary
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

    def _stream_response(self, messages: list[dict]) -> str:
        # collect streamed tokens and print them live
        print("[LLM] ", end="", flush=True)
        tokens = []
        for token in self.llm.chat_stream(messages):
            print(token, end="", flush=True)
            tokens.append(token)
        print()
        return "".join(tokens)

    def _annotate_frame(self, frame, result: PipelineResult):
        # draw detected labels and the LLM response onto the frame
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # draw detection boxes
        for d in result.detections:
            x1, y1, x2, y2 = (
                int(d["bbox"][0] * w), int(d["bbox"][1] * h),
                int(d["bbox"][2] * w), int(d["bbox"][3] * h),
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 100), 2)
            label_text = f"{d['label']} {d['score']:.0%}"
            cv2.putText(annotated, label_text, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 1)

        # draw LLM response as wrapped text at the bottom
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


# ── helpers ────────────────────────────────────────────────────────────────────

def _changes_to_text(changes: dict) -> str:
    parts = []
    if changes["appeared"]:
        parts.append(f"Newly appeared: {', '.join(changes['appeared'])}")
    if changes["disappeared"]:
        parts.append(f"No longer visible: {', '.join(changes['disappeared'])}")
    return ". ".join(parts) + "."


def _wrap_text(text: str, max_chars: int) -> list[str]:
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