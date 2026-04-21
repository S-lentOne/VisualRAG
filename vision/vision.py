import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def get_camera_name(index):
    # Will be adding detection and use of a different logic per OS, starting with windows, however for now
    # this is Linux-specific: tries to read camera name
    path = f"/sys/class/video4linux/video{index}/name"
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return f"Camera {index}"


def list_cameras(max_tested=10):
    cameras = []

    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)

        if cap.isOpened():
            name = get_camera_name(i)

            cameras.append({
                "index": i,
                "name": name
            })

            cap.release()

    return cameras



def LiveVideo():
    
    # detect cameras
    cams = list_cameras()

    # show menu
    print("\nAvailable cameras:\n")

    for i, cam in enumerate(cams, start=1):
        print(f"{i}. {cam['name']} (Enter {cam['index']} to use)")

    # user choice
    choice = int(input("\nWhich one would you like to use: ")) - 1

    ch = cams[choice]["index"]

    print(f"\nUsing: {cams[choice]['name']}\n")

    camera = cv2.VideoCapture(ch, cv2.CAP_V4L2)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while camera.isOpened():
        ret, frame = camera.read(1)
        # These few modifications to frame are for mirroring and rotating the camera input
        frame = cv2.flip(frame,1)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        if not ret:
            break

        cv2.imshow('V 0.01', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    camera.release()
    cv2.destroyAllWindows()
    
    return 0

def Image():
    img = cv2.imread('image.jpg')
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

def Video():
    cap = cv2.VideoCapture('video.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0



def main():
     mode = input("L for live video, V for video, I for image: \n")

     if mode == "L" or mode == "l":
         LiveVideo()
     elif mode == "I" or mode == "i":
         Image()
     elif mode == "V" or mode == "v":
         Video()
     else:
         print("Invalid Input \nRestarting \n. \n. \n.")
         main()
     return 0
    
    # LiveVideo()
    # Uncomment this line only if the logic above fails, and you need to hard reset to a live video input
    
    
main()
