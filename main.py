import os
import sys
import cv2
from pipeline import Pipeline

# ── camera helpers (same logic as original vision.py) ─────────────────────────

def _get_camera_name(index):
    path = f"/sys/class/video4linux/video{index}/name"
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return f"Camera {index}"

def _list_cameras(max_tested=10):
    cameras = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            cameras.append({"index": i, "name": _get_camera_name(i)})
            cap.release()
    return cameras

# ── result printer (callback used by video/live modes) ────────────────────────

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

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════╗")
    print("║   Object Detection + RAG + Ollama    ║")
    print("╚══════════════════════════════════════╝\n")

    print("Input mode:")
    print("  L — live camera feed")
    print("  V — video file")
    print("  I — image file")
    mode = input("\nChoice: ").strip().upper()

    # optional config overrides at startup
    config = {}
    stream_choice = input("Stream LLM output token by token? (y/N): ").strip().lower()
    if stream_choice == "y":
        config["stream"] = True

    pipe = Pipeline(config=config)
    pipe.setup()

    if mode == "I":
        path = input("Image path: ").strip()
        result = pipe.run_image(path)
        _print_result(result)

        # allow follow-up questions after image analysis
        while True:
            q = input("\nFollow-up question (or Enter to quit): ").strip()
            if not q:
                break
            answer = pipe.followup(result, q)
            print(f"LLM: {answer}")

    elif mode == "V":
        path = input("Video path: ").strip()
        pipe.run_video(path, on_result=_print_result)

    elif mode == "L":
        cams = _list_cameras()
        if not cams:
            print("No cameras found.")
            sys.exit(1)

        print("\nAvailable cameras:")
        for cam in cams:
            print(f"  {cam['index']} — {cam['name']}")
        cam_idx = int(input("Camera index: ").strip())
        pipe.run_live(camera_index=cam_idx, on_result=_print_result)

    else:
        print("Invalid choice.")
        sys.exit(1)


main()