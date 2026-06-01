import base64
import json
import queue
import sys
import threading
import time

import cv2
from llm.prompt_builder import build_followup_prompt, build_prompt
from pipeline import Pipeline
from rag.user_store import UserStore

CHAT_QUEUE = queue.Queue()
COMMAND_QUEUE = queue.Queue()


def emit(event: str, data=None):
    try:
        print(
            json.dumps(
                {
                    "event": event,
                    "data": data,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    except (BrokenPipeError, OSError):
        pass


def emit_error(error):
    emit("error", str(error))


def submit_chat(question: str) -> bool:
    question = question.strip()

    if not question:
        return False

    CHAT_QUEUE.put(question)

    return True


def submit_command(command):
    COMMAND_QUEUE.put(command)


def command_worker():
    SETTINGS = {
        "camera_index": 0,
        "detection_interval": 1.0,
        "model": "gemma3:4b",
        "debug": True,
    }
    while True:
        try:
            line = sys.stdin.readline()

            if not line:
                continue

            payload = json.loads(line)
            print(payload)

            if "command" in payload:
                submit_command(payload)

            elif "message" in payload:
                submit_chat(payload["message"])

        except Exception as e:
            emit_error(e)


def _run_live(pipe, camera_index, user_store, profile):

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2

    cap = cv2.VideoCapture(
        camera_index,
        backend,
    )

    if not cap.isOpened():
        emit(
            "error",
            f"Could not open camera {camera_index}",
        )
        return

    stop_event = threading.Event()

    shared = {
        "latest_frame": None,
        "last_result": None,
        "chat_history": [],
        "lock": threading.Lock(),
    }

    def camera_worker():
        last_frame_emit = 0

        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            with shared["lock"]:
                shared["latest_frame"] = frame.copy()

            now = time.time()

            if now - last_frame_emit > 0.1:
                success, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 70],
                )

                if success:
                    emit(
                        "frame",
                        base64.b64encode(buffer).decode("utf-8"),
                    )

                    last_frame_emit = now

    def analysis_worker():
        interval = 1.0

        while not stop_event.is_set():
            with shared["lock"]:
                frame = (
                    None
                    if shared["latest_frame"] is None
                    else shared["latest_frame"].copy()
                )

            if frame is None:
                time.sleep(0.1)
                continue

            try:
                detections = pipe._detect(frame)

                result = pipe._analyze(detections)

                user_store.record_detections(
                    profile,
                    [d["label"] for d in result.detections],
                )

                user_ctx = user_store.get_context_for_llm(profile)

                pkg = build_prompt(
                    detections=result.detections,
                    static_context=result.static_context,
                    episodic_context=result.episodic_context,
                    user_context=user_ctx,
                )

                with shared["lock"]:
                    shared["last_result"] = result
                    shared["chat_history"] = pkg.messages + [
                        {
                            "role": "assistant",
                            "content": result.llm_response,
                        }
                    ]

                emit(
                    "detections",
                    [d["label"] for d in result.detections],
                )

                emit(
                    "scene",
                    result.llm_response,
                )

            except Exception as e:
                emit_error(e)

            time.sleep(interval)

    def answer_worker():
        while not stop_event.is_set():
            try:
                q = CHAT_QUEUE.get(timeout=0.5)
            except queue.Empty:
                continue

            with shared["lock"]:
                history = list(shared["chat_history"])
                result = shared["last_result"]

            if result is None:
                emit(
                    "chat_response",
                    {
                        "question": q,
                        "answer": "Still initializing.",
                    },
                )
                continue

            factual = _try_answer_factually(
                q,
                result.detections,
            )

            if factual:
                answer = factual

            else:
                updated = build_followup_prompt(
                    history,
                    q,
                )

                answer = pipe.llm.chat(updated)

                with shared["lock"]:
                    shared["chat_history"] = updated + [
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    ]

            user_store.record_exchange(
                profile,
                q,
                answer,
            )

            emit(
                "chat_response",
                {
                    "question": q,
                    "answer": answer,
                },
            )

    def command_runtime_worker():
        while not stop_event.is_set():
            try:
                cmd = COMMAND_QUEUE.get(timeout=0.5)

            except queue.Empty:
                continue

            try:
                command = cmd.get("command")

                if command == "update_settings":
                    settings = cmd.get("settings", {})

                    pipe.config.update(settings)

                    emit(
                        "status",
                        {
                            "settings": "updated",
                        },
                    )

                elif command == "analyze_image":
                    path = cmd.get("path")

                    if not path:
                        continue

                    frame = cv2.imread(path)

                    if frame is None:
                        emit_error(f"Failed to load image: {path}")
                        continue

                    detections = pipe._detect(frame)

                    result = pipe._analyze(detections)

                    annotated = frame.copy()

                    for d in detections:
                        if "bbox" not in d:
                            continue

                        x1, y1, x2, y2 = d["bbox"]

                        cv2.rectangle(
                            annotated,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                        )

                        cv2.putText(
                            annotated,
                            d["label"],
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                    emit(
                        "detections",
                        [d["label"] for d in detections],
                    )

                    emit(
                        "scene",
                        result.llm_response,
                    )

                    success, buffer = cv2.imencode(
                        ".jpg",
                        annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 85],
                    )

                    if success:
                        emit(
                            "frame",
                            base64.b64encode(buffer).decode("utf-8"),
                        )

                elif command == "analyze_video":
                    path = cmd.get("path")

                    if not path:
                        continue

                    cap = cv2.VideoCapture(path)

                    if not cap.isOpened():
                        emit_error(f"Failed to open video: {path}")
                        continue

                    while cap.isOpened() and not stop_event.is_set():
                        ret, frame = cap.read()

                        if not ret:
                            break

                        detections = pipe._detect(frame)

                        annotated = frame.copy()

                        for d in detections:
                            if "bbox" not in d:
                                continue

                            x1, y1, x2, y2 = d["bbox"]

                            cv2.rectangle(
                                annotated,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 255, 0),
                                2,
                            )

                            cv2.putText(
                                annotated,
                                d["label"],
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                        success, buffer = cv2.imencode(
                            ".jpg",
                            annotated,
                            [cv2.IMWRITE_JPEG_QUALITY, 80],
                        )

                        if success:
                            emit(
                                "frame",
                                base64.b64encode(buffer).decode("utf-8"),
                            )

                        time.sleep(0.03)

                    cap.release()

                else:
                    emit_error(f"Unknown command: {command}")

            except Exception as e:
                emit_error(e)

    t_camera = threading.Thread(
        target=camera_worker,
        daemon=True,
    )

    t_analysis = threading.Thread(
        target=analysis_worker,
        daemon=True,
    )

    t_answer = threading.Thread(
        target=answer_worker,
        daemon=True,
    )
    t_command = threading.Thread(
        target=command_runtime_worker,
        daemon=True,
    )
    t_command.start()
    t_camera.start()
    t_analysis.start()
    t_answer.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.2)

    except KeyboardInterrupt:
        stop_event.set()

    t_camera.join(timeout=2)
    t_analysis.join(timeout=2)
    t_answer.join(timeout=2)

    cap.release()

    with shared["lock"]:
        result = shared["last_result"]

    if result:
        user_store.update_scene_summary(
            profile,
            result.llm_response,
        )

    user_store.save(profile)


def _try_answer_factually(question, detections):
    q = question.lower().strip().rstrip("?")

    labels = [d["label"] for d in detections]
    labels_lower = [label.lower() for label in labels]

    if any(
        trigger in q
        for trigger in (
            "what do you see",
            "what did you see",
            "what objects",
            "list objects",
            "list detections",
        )
    ):
        if not labels:
            return "No objects detected."

        return "Detected objects:\n" + "\n".join(
            f"{i + 1}. {label}" for i, label in enumerate(labels)
        )

    if any(
        trigger in q
        for trigger in (
            "did you see",
            "did you detect",
            "is there",
            "was there",
            "were there",
        )
    ):
        for label in labels_lower:
            if label in q:
                return f"Yes, {label} was detected."

        return None

    return None


def start_live_session(camera_index=0):
    pipe = Pipeline(config={})
    pipe.setup()
    emit(
        "status",
        {
            "ollama": "starting",
            "camera": "inactive",
            "yolo": "idle",
        },
    )

    user_store = UserStore()

    profile = user_store.load("default")
    profile = user_store.record_session_start(profile)

    # emit("status", "ready")

    threading.Thread(
        target=command_worker,
        daemon=True,
    ).start()

    _run_live(
        pipe,
        camera_index,
        user_store,
        profile,
    )
