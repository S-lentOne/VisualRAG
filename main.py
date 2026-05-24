import os
import sys
import cv2
import threading
import queue
from pipeline import Pipeline, PipelineResult
from llm.prompt_builder import build_prompt, build_followup_prompt
from rag.user_store import UserStore


def _get_camera_name(index):
    path = f"/sys/class/video4linux/video{index}/name"
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return f"Camera {index}"

def _list_cameras(max_tested=10):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    cameras = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            cameras.append({"index": i, "name": _get_camera_name(i)})
            cap.release()
    return cameras

def _print_result(result):
    print("\n── Scene Analysis ───────────────────────────────────")
    labels = [f"{d['label']} ({d['score']:.0%})" for d in result.detections]
    print(f"Detected : {', '.join(labels) if labels else 'nothing'}")
    if result.changes["appeared"]:
        print(f"Appeared : {', '.join(result.changes['appeared'])}")
    if result.changes["disappeared"]:
        print(f"Gone     : {', '.join(result.changes['disappeared'])}")
    print(f"LLM      : {result.llm_response}")
    print("─────────────────────────────────────────────────────")

def _chat_loop(pipe, result, user_store, profile, is_video_session: bool = False):
    user_ctx = user_store.get_context_for_llm(profile)
    pkg = build_prompt(
        detections=result.detections,
        static_context=result.static_context,
        episodic_context=result.episodic_context,
        user_context=user_ctx,
        is_video_session=is_video_session,
    )
    history = pkg.messages + [{"role": "assistant", "content": result.llm_response}]

    labels = [d["label"] for d in result.detections]
    profile = user_store.record_detections(profile, labels)
    profile = user_store.update_scene_summary(profile, result.llm_response)

    print("\n── Chat (press Enter on empty line to quit) ─────────")
    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            print("Ending chat.")
            break
        history = build_followup_prompt(history, q)
        answer = pipe.llm.chat(history)
        print(f"LLM: {answer}\n")
        history.append({"role": "assistant", "content": answer})
        profile = user_store.record_exchange(profile, q, answer)

    user_store.save(profile)
    print(f"[Profile] Session saved for '{profile.user_id}'.")


def _video_summary(results: list):
    print("\n── Video Summary ────────────────────────────────────")
    all_labels = {}
    for r in results:
        for d in r.detections:
            all_labels[d["label"]] = all_labels.get(d["label"], 0) + 1
    if all_labels:
        ranked = sorted(all_labels.items(), key=lambda x: -x[1])
        print(f"Objects seen : {', '.join(f'{l}({c}x)' for l, c in ranked)}")
    else:
        print("Objects seen : nothing detected")
    print(f"Scenes analyzed : {len(results)}")
    if results:
        print(f"\nFinal scene description:")
        print(f"  {results[-1].llm_response}")
    print("─────────────────────────────────────────────────────")


def _build_video_session_result(results: list) -> PipelineResult:
    """
    Aggregate all per-frame results into one unified context for post-video chat.
    The LLM will know about every object seen across the entire video, not just the last frame.
    """
    # tally all labels, keep highest confidence score seen for each
    label_counts = {}
    label_best_score = {}
    for r in results:
        for d in r.detections:
            lbl = d["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            label_best_score[lbl] = max(label_best_score.get(lbl, 0.0), d["score"])

    # rank by frequency then score
    ranked = sorted(label_counts.keys(), key=lambda l: (-label_counts[l], -label_best_score[l]))
    aggregated_detections = [
        {"label": l, "score": label_best_score[l], "bbox": [0, 0, 0, 0]}
        for l in ranked
    ]

    # combine static context blocks, deduplicated by class label header
    seen_headers = set()
    static_blocks = []
    for r in results:
        for block in r.static_context.split("\n\n"):
            header = block.strip().splitlines()[0] if block.strip() else ""
            if header.startswith("[") and header not in seen_headers:
                seen_headers.add(header)
                static_blocks.append(block.strip())
    combined_static = "\n\n".join(static_blocks)

    # frequency note + per-scene narrative timeline for episodic context
    freq_note = "Object frequency across video: " + ", ".join(
        f"{l} ({label_counts[l]}x)" for l in ranked
    )
    timeline = "\n\n".join(
        f"Scene {i+1}: {r.llm_response}" for i, r in enumerate(results) if r.llm_response
    )
    combined_episodic = freq_note + "\n\n" + timeline

    # union of all appeared/disappeared across session
    all_appeared = set()
    all_disappeared = set()
    for r in results:
        all_appeared.update(r.changes.get("appeared", []))
        all_disappeared.update(r.changes.get("disappeared", []))

    return PipelineResult(
        detections=aggregated_detections,
        scene_query=f"video session — {len(results)} scenes analyzed",
        static_context=combined_static,
        episodic_context=combined_episodic,
        llm_response=results[-1].llm_response if results else "",
        changes={
            "appeared":    sorted(all_appeared),
            "disappeared": sorted(all_disappeared),
            "stable":      [],
        },
    )


def _run_live_interactive(pipe, camera_index: int, user_store, profile):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"Could not open camera {camera_index}.")
        return

    state = {"last_result": None, "chat_history": [], "running": True}
    lock = threading.Lock()
    question_queue = queue.Queue()

    def input_thread():
        print("\n── Live Chat (type questions anytime, 'quit' to stop) ──")
        while True:
            try:
                q = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if q.lower() in ("quit", "exit", "q"):
                with lock:
                    state["running"] = False
                break
            if q:
                question_queue.put(q)

    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    frame_count = 0
    interval = pipe.config["frame_interval"]
    print("[Live] Feed running. Press Q in video window or type 'quit' to stop.\n")

    while True:
        with lock:
            if not state["running"]:
                break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % interval == 0:
            detections = pipe._detect(frame)
            result = pipe._analyze(detections)

            labels = [d["label"] for d in result.detections]
            user_store.record_detections(profile, labels)

            user_ctx = user_store.get_context_for_llm(profile)
            pkg = build_prompt(
                detections=result.detections,
                static_context=result.static_context,
                episodic_context=result.episodic_context,
                user_context=user_ctx,
                is_video_session=False,
            )
            with lock:
                state["last_result"] = result
                state["chat_history"] = pkg.messages + [
                    {"role": "assistant", "content": result.llm_response}
                ]

            _print_result(result)

        while not question_queue.empty():
            q = question_queue.get()
            with lock:
                history = list(state["chat_history"])
                has_result = state["last_result"] is not None

            if not has_result:
                print("LLM: Still warming up, no scene analyzed yet.\n")
                continue

            updated = build_followup_prompt(history, q)
            answer = pipe.llm.chat(updated)
            print(f"LLM: {answer}\n")
            profile = user_store.record_exchange(profile, q, answer)
            with lock:
                state["chat_history"] = updated + [{"role": "assistant", "content": answer}]

        with lock:
            last = state["last_result"]
        display = pipe._annotate_frame(frame, last) if last else frame
        cv2.imshow("Live — VisualRAG", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            with lock:
                state["running"] = False
            break

    cap.release()
    cv2.destroyAllWindows()

    user_store.update_scene_summary(profile, state["last_result"].llm_response if state["last_result"] else "")
    user_store.save(profile)
    print(f"\n[Live] Session ended. Profile saved for '{profile.user_id}'.")
    print(f"[Live] Total episodes recorded: {pipe.episode_store.stats()['total_episodes']}")


def main():
    print("╔══════════════════════════════════════╗")
    print("║   Object Detection + RAG + Ollama    ║")
    print("╚══════════════════════════════════════╝\n")

    user_store = UserStore()
    user_id = input("Username (or Enter for 'guest'): ").strip() or "guest"
    profile = user_store.load(user_id)
    profile = user_store.record_session_start(profile)

    if profile.session_count > 1:
        print(f"Welcome back, {user_id}! Session #{profile.session_count}.")
        if profile.frequent_labels:
            top = sorted(profile.frequent_labels.items(), key=lambda x: -x[1])[:3]
            print(f"Your common objects: {', '.join(l for l, _ in top)}")
    else:
        print(f"Welcome, {user_id}! First session.")

    print("\nInput mode:")
    print("  L — live camera (real-time analysis + chat)")
    print("  V — video file")
    print("  I — image file")
    mode = input("\nChoice: ").strip().upper()

    config = {}
    if input("Stream LLM output? (y/N): ").strip().lower() == "y":
        config["stream"] = True

    pipe = Pipeline(config=config)
    pipe.setup()

    if mode == "I":
        path = input("Image path: ").strip()
        result = pipe.run_image(path)
        _print_result(result)
        _chat_loop(pipe, result, user_store, profile)

    elif mode == "V":
        path = input("Video path: ").strip()
        results = pipe.run_video(path)
        _video_summary(results)
        if results:
            print("\nYou can now chat about the video.")
            session_result = _build_video_session_result(results)
            _chat_loop(pipe, session_result, user_store, profile, is_video_session=True)

    elif mode == "L":
        cams = _list_cameras()
        if not cams:
            print("No cameras found.")
            sys.exit(1)
        print("\nAvailable cameras:")
        for cam in cams:
            print(f"  {cam['index']} — {cam['name']}")
        cam_idx = int(input("Camera index: ").strip())
        _run_live_interactive(pipe, cam_idx, user_store, profile)

    else:
        print("Invalid choice.")
        sys.exit(1)


main()