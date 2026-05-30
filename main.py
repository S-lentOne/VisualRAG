import os
import sys
import cv2
import threading
import queue
import time
from pipeline import Pipeline, PipelineResult
from llm.prompt_builder import build_prompt, build_followup_prompt
from rag.user_store import UserStore

# ── helpers ───────────────────────────────────────────────────────────────────

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

def _try_answer_factually(question, detections):
    q = question.lower().strip().rstrip("?")
    labels = [d["label"] for d in detections]
    labels_lower = [l.lower() for l in labels]

    list_triggers = ("list", "what did you see", "what was on", "what objects",
                     "what items", "what things", "show me everything", "all items", "all objects")
    if any(t in q for t in list_triggers):
        if not labels:
            return "No objects were detected."
        return "Detected objects:\n" + "\n".join(f"{i+1}. {l}" for i, l in enumerate(labels))

    yes_no_triggers = ("did you see", "were there", "was there", "did you detect", "is there", "any ")
    if any(t in q for t in yes_no_triggers):
        for label in labels_lower:
            if label in q or label.rstrip("s") in q:
                return f"Yes, {label} was detected on camera."
        known = ["laptop","notebook","mouse","pen","bottle","monitor","keyboard","headphones",
                 "chair","phone","cup","backpack","controller","camera","speaker","wallet",
                 "watch","glasses","earbuds","charger","wire","eraser","ruler","calculator",
                 "paper","plant","spoon","chopsticks","can","key"]
        for obj in known:
            if obj in q or obj.rstrip("s") in q:
                if obj not in labels_lower:
                    return f"No, {obj} was not detected on camera."
    return None

# ── image mode ────────────────────────────────────────────────────────────────

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

def _chat_loop(pipe, result, user_store, profile, is_video_session=False):
    user_ctx = user_store.get_context_for_llm(profile)
    pkg = build_prompt(
        detections=result.detections,
        static_context=result.static_context,
        episodic_context=result.episodic_context,
        user_context=user_ctx,
        is_video_session=is_video_session,
    )
    history = pkg.messages + [{"role": "assistant", "content": result.llm_response}]
    profile = user_store.record_detections(profile, [d["label"] for d in result.detections])
    profile = user_store.update_scene_summary(profile, result.llm_response)

    print("\n── Chat (empty Enter to quit) ───────────────────────")
    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            print("Ending chat.")
            break
        factual = _try_answer_factually(q, result.detections)
        if factual:
            print(f"LLM: {factual}\n")
            history += [{"role": "user", "content": q},
                        {"role": "assistant", "content": factual}]
            profile = user_store.record_exchange(profile, q, factual)
            continue
        history = build_followup_prompt(history, q)
        answer = pipe.llm.chat(history)
        print(f"LLM: {answer}\n")
        history.append({"role": "assistant", "content": answer})
        profile = user_store.record_exchange(profile, q, answer)

    user_store.save(profile)
    print(f"[Profile] Saved for '{profile.user_id}'.")

# ── video mode ────────────────────────────────────────────────────────────────

def _video_summary(results):
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
        print(f"\nFinal description:\n  {results[-1].llm_response}")
    print("─────────────────────────────────────────────────────")

def _build_video_session_result(results):
    label_counts, label_best = {}, {}
    for r in results:
        for d in r.detections:
            l = d["label"]
            label_counts[l] = label_counts.get(l, 0) + 1
            label_best[l] = max(label_best.get(l, 0.0), d["score"])

    ranked = sorted(label_counts, key=lambda l: (-label_counts[l], -label_best[l]))
    agg = [{"label": l, "score": label_best[l], "bbox": [0,0,0,0]} for l in ranked]

    seen, static_blocks = set(), []
    for r in results:
        for block in r.static_context.split("\n\n"):
            h = block.strip().splitlines()[0] if block.strip() else ""
            if h.startswith("[") and h not in seen:
                seen.add(h)
                static_blocks.append(block.strip())

    freq = "Object frequency: " + ", ".join(f"{l}({label_counts[l]}x)" for l in ranked)
    timeline = "\n\n".join(f"Scene {i+1}: {r.llm_response}" for i, r in enumerate(results) if r.llm_response)
    all_app, all_dis = set(), set()
    for r in results:
        all_app.update(r.changes.get("appeared", []))
        all_dis.update(r.changes.get("disappeared", []))

    return PipelineResult(
        detections=agg,
        scene_query=f"video session — {len(results)} scenes",
        static_context="\n\n".join(static_blocks),
        episodic_context=freq + "\n\n" + timeline,
        llm_response=results[-1].llm_response if results else "",
        changes={"appeared": sorted(all_app), "disappeared": sorted(all_dis), "stable": []},
    )

# ── live mode ─────────────────────────────────────────────────────────────────

def _run_live(pipe, camera_index, user_store, profile):
    """
    Three threads:
      display thread  — reads camera at full FPS, shows annotated window continuously
      analysis thread — pulls latest frame every N frames, runs YOLO + RAG + LLM
      input thread    — dedicated to stdin only, never touched by other threads

    No detection prints go to stdout. Status is shown only in the CV window title
    and in a print AFTER the user presses Enter, so it never breaks the input line.
    """
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Could not open camera {camera_index}.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Live] Camera opened at {actual_w}x{actual_h}")
    print("[Live] Video window open. Press Q in the window or type 'quit' to stop.\n")
    print("─" * 52)
    print("  Type your questions below at any time.")
    print("─" * 52 + "\n")

    stop_event     = threading.Event()
    question_queue = queue.Queue()
    answer_queue   = queue.Queue()

    # shared between display and analysis threads
    shared = {
        "latest_frame":   None,   # always the newest raw frame
        "overlay_result": None,
        "live_detections":  [],    # per-frame YOLO results for real-time boxes
        "chat_history":   [],
        "last_result":    None,
        "lock":           threading.Lock(),
    }

    # ── display thread: reads camera at full FPS, runs YOLO on every frame ──────
    def display_worker():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                stop_event.set()
                break

            frame = cv2.flip(frame, 1)

            # always run YOLO on every frame so boxes are frame-accurate
            live_detections = pipe._detect(frame)

            with shared["lock"]:
                shared["latest_frame"]      = frame.copy()
                shared["live_detections"]   = live_detections

            # draw boxes from live per-frame detection
            display = frame.copy()
            h, w = display.shape[:2]
            for d in live_detections:
                x1 = int(d["bbox"][0] * w)
                y1 = int(d["bbox"][1] * h)
                x2 = int(d["bbox"][2] * w)
                y2 = int(d["bbox"][3] * h)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 100), 2)
                cv2.putText(display, f"{d['label']} {d['score']:.0%}",
                            (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2)

            labels = ", ".join(d["label"] for d in live_detections) if live_detections else "no detections"
            cv2.setWindowTitle("VisualRAG — Live", f"VisualRAG  |  {labels}")
            cv2.imshow("VisualRAG — Live", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

        cv2.destroyAllWindows()

    # ── analysis thread: RAG + LLM on sampled detections (no YOLO here) ─────────
    def analysis_worker():
        interval = pipe.config["frame_interval"] * 0.033  # seconds between analyses
        while not stop_event.is_set():
            time.sleep(interval)

            with shared["lock"]:
                detections = list(shared["live_detections"])
            if not detections and shared["last_result"] is not None:
                # nothing detected — still update episodic memory with empty scene
                pass

            result = pipe._analyze(detections)

            user_store.record_detections(profile, [d["label"] for d in result.detections])
            user_ctx = user_store.get_context_for_llm(profile)
            pkg = build_prompt(
                detections=result.detections,
                static_context=result.static_context,
                episodic_context=result.episodic_context,
                user_context=user_ctx,
            )

            with shared["lock"]:
                shared["overlay_result"] = result
                shared["last_result"]    = result
                shared["chat_history"]   = pkg.messages + [
                    {"role": "assistant", "content": result.llm_response}
                ]

    # ── input thread: owns stdin completely ───────────────────────────────────
    def input_worker():
        while not stop_event.is_set():
            try:
                sys.stdout.write("You: ")
                sys.stdout.flush()
                line = sys.stdin.readline()
                if not line:
                    stop_event.set()
                    break

                q = line.strip()
                if not q:
                    continue
                if q.lower() in ("quit", "exit", "q"):
                    stop_event.set()
                    break

                question_queue.put(q)

                # block here until answer arrives — no other thread prints during this
                try:
                    answer = answer_queue.get(timeout=60)
                    # print detection status first so user sees what the answer is based on
                    with shared["lock"]:
                        cur = shared["last_result"]
                    if cur and cur.detections:
                        labels = ", ".join(d["label"] for d in cur.detections)
                        print(f"[Current detections: {labels}]")
                    print(f"LLM: {answer}\n")
                except queue.Empty:
                    print("[No response — Ollama may be slow. Try again.]\n")

            except (KeyboardInterrupt, EOFError):
                stop_event.set()
                break

    # ── answer worker: processes questions from input thread ──────────────────
    def answer_worker():
        while not stop_event.is_set():
            try:
                q = question_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            with shared["lock"]:
                history = list(shared["chat_history"])
                cur     = shared["last_result"]

            if cur is None:
                answer_queue.put("Still initializing — no scene analyzed yet.")
                continue

            factual = _try_answer_factually(q, cur.detections)
            if factual:
                answer = factual
            else:
                updated = build_followup_prompt(history, q)
                answer  = pipe.llm.chat(updated)
                with shared["lock"]:
                    shared["chat_history"] = updated + [
                        {"role": "assistant", "content": answer}
                    ]

            profile_ref = user_store.record_exchange(profile, q, answer)
            answer_queue.put(answer)

    # start all threads
    t_display  = threading.Thread(target=display_worker,  daemon=True)
    t_analysis = threading.Thread(target=analysis_worker, daemon=True)
    t_answer   = threading.Thread(target=answer_worker,   daemon=True)
    t_input    = threading.Thread(target=input_worker,    daemon=True)

    t_display.start()
    t_analysis.start()
    t_answer.start()
    t_input.start()

    # main thread just waits for stop
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()

    # cleanup
    t_display.join(timeout=2)
    cap.release()

    with shared["lock"]:
        last = shared["last_result"]
    if last:
        user_store.update_scene_summary(profile, last.llm_response)
    user_store.save(profile)
    print(f"\n[Live] Ended. Episodes: {pipe.episode_store.stats()['total_episodes']}")
    print(f"[Profile] Saved for '{profile.user_id}'.")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════╗")
    print("║   Object Detection + RAG + Ollama    ║")
    print("╚══════════════════════════════════════╝\n")

    user_store = UserStore()
    user_id    = input("Username (or Enter for 'guest'): ").strip() or "guest"
    profile    = user_store.load(user_id)
    profile    = user_store.record_session_start(profile)

    if profile.session_count > 1:
        print(f"Welcome back, {user_id}! Session #{profile.session_count}.")
        if profile.frequent_labels:
            top = sorted(profile.frequent_labels.items(), key=lambda x: -x[1])[:3]
            print(f"Common objects: {', '.join(l for l, _ in top)}")
    else:
        print(f"Welcome, {user_id}!")

    print("\n  L — live camera")
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
        path    = input("Video path: ").strip()
        results = pipe.run_video(path)
        _video_summary(results)
        if results:
            print("\nChat about the video below.")
            _chat_loop(pipe, _build_video_session_result(results), user_store, profile, is_video_session=True)

    elif mode == "L":
        cams = _list_cameras()
        if not cams:
            print("No cameras found.")
            sys.exit(1)
        print("\nAvailable cameras:")
        for cam in cams:
            print(f"  {cam['index']} — {cam['name']}")
        cam_idx = int(input("Camera index: ").strip())
        _run_live(pipe, cam_idx, user_store, profile)

    else:
        print("Invalid choice.")
        sys.exit(1)

main()