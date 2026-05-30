import cv2

working = []

for i in range(20):

    cap = cv2.VideoCapture(i)

    if cap.isOpened():

        ret, frame = cap.read()

        if ret:
            print(f"Camera {i} works: {frame.shape}")
            working.append(i)

        cap.release()

print("Working cameras:", working)