import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "models/tr-9.pt"
# swap to a real image/video path to test on file input
TEST_IMAGE = None
TEST_VIDEO = None
CONFIDENCE = 0.4

def test_model_load():
    print(f"Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print(f"Model loaded. Classes ({len(model.names)}): {model.names}")
    return model

def test_on_image(model, path):
    results = model(path, conf=CONFIDENCE)[0]
    img = results.plot()
    cv2.imshow("YOLO - Image Test", img)
    print(f"Detections: {len(results.boxes)}")
    for box in results.boxes:
        cls = int(box.cls)
        print(f"  {model.names[cls]} — conf={float(box.conf):.2f}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_on_video(model, path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=CONFIDENCE)[0]
        annotated = results.plot()
        cv2.imshow("YOLO - Video Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

def test_on_webcam(model):
    # lists available cameras same way as vision.py
    print("Scanning for cameras...")
    cams = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            cams.append(i)
            cap.release()
    
    if not cams:
        print("No cameras found.")
        return

    print(f"Found cameras at indices: {cams}")
    cam_idx = int(input(f"Pick camera index {cams}: "))
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Running — press Q to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = model(frame, conf=CONFIDENCE)[0]
        annotated = results.plot()
        cv2.imshow("YOLO - Webcam Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    if not Path(MODEL_PATH).exists():
        print(f"Model not found at {MODEL_PATH}. Make sure tr-9.pt is under /models.")
        return

    model = test_model_load()

    print("\nTest modes:")
    print("  1 - webcam live feed")
    print("  2 - image file")
    print("  3 - video file")
    mode = input("Pick mode: ").strip()

    if mode == "1":
        test_on_webcam(model)
    elif mode == "2":
        path = TEST_IMAGE or input("Image path: ").strip()
        test_on_image(model, path)
    elif mode == "3":
        path = TEST_VIDEO or input("Video path: ").strip()
        test_on_video(model, path)
    else:
        print("Invalid choice.")

main()