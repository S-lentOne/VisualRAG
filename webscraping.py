# Classes I want to detect

# laptop | keyboard | mouse | monitor | phone | earbuds | headphones | charger | usb cable | speaker | camera | controller
# 12
# wallet | key | backpack | watch | glasses | notebook | pen | eraser | ruler | calculator | paper | chair
# 12
# cup | bottle | can | plant | spoon | chopsticks
# 6
import os
import time
import random
import requests
from tqdm import tqdm
from ddgs import DDGS

CLASSES = {
    "laptop": "laptop",
    "keyboard": "computer keyboard",
    "mouse": "computer mouse",
    "monitor": "computer monitor",
    "phone": "smartphone",
    "earbuds": "wireless earbuds",
    "headphones": "over ear headphones",
    "charger": "phone charger",
    "wire": "electrical wire cable",
    "speaker": "computer speaker",
    "camera": "digital camera",
    "controller": "game controller",
    "wallet": "wallet",
    "key": "house key",
    "backpack": "backpack",
    "watch": "wrist watch",
    "glasses": "eyeglasses",
    "notebook": "paper notebook",
    "pen": "pen",
    "eraser": "eraser",
    "ruler": "ruler",
    "calculator": "calculator",
    "paper": "sheet of paper",
    "chair": "office chair",
    "cup": "cup",
    "bottle": "water bottle",
    "can": "soda can",
    "plant": "potted plant",
    "spoon": "spoon",
    "chopsticks": "chopsticks"
}

OUTPUT_DIR = "dataset/images"
IMAGES_PER_CLASS = 120


def download_images(query, folder, max_images):
    os.makedirs(folder, exist_ok=True)

    downloaded = 0
    attempts = 0

    with DDGS() as ddgs:
        while downloaded < max_images and attempts < 5:
            try:
                results = ddgs.images(query, max_results=max_images)

                for r in tqdm(results, desc=query):

                    if downloaded >= max_images:
                        break

                    try:
                        url = r["image"]

                        response = requests.get(url, timeout=8)

                        if response.status_code != 200:
                            continue

                        file_path = os.path.join(folder, f"{downloaded}.jpg")

                        with open(file_path, "wb") as f:
                            f.write(response.content)

                        downloaded += 1

                        time.sleep(random.uniform(0.2, 0.6))

                    except Exception:
                        continue

            except Exception:
                attempts += 1
                print(f"retrying {query}...")
                time.sleep(random.uniform(3, 6))


def main():
    for label, search_term in CLASSES.items():

        folder = os.path.join(OUTPUT_DIR, label)

        print(f"\nDownloading {label}...")
        download_images(search_term, folder, IMAGES_PER_CLASS)

        time.sleep(random.uniform(2, 5))


if __name__ == "__main__":
    main()