import cv2
import json
# import numpy as np
import os
from tqdm import tqdm

hands_dir = "../data/hand_labels/"

files = []
for file in os.listdir(hands_dir):
    if os.path.isdir(hands_dir + file) and "output" not in file:
        for sub_file in os.listdir(hands_dir + file):
            if "jpg" not in sub_file:
                files.append(hands_dir + file + "/" + sub_file)

count = 0
for file in tqdm(files):
    with open(file, "r") as f:
        data = json.load(f)

    # Draw the rectangle and square on the image
    image = cv2.imread(file[0:-5] + ".jpg")
    try:
        if data["is_left"]:
            cv2.imwrite(f"../data/hand_images_real/full/left/{count:08d}.jpg", image)
            image = cv2.flip(image, 1)
            cv2.imwrite(f"../data/hand_images_real/full/right/{count:08d}.jpg", image)
        else:
            cv2.imwrite(f"../data/hand_images_real/full/right/{count:08d}.jpg", image)
            image = cv2.flip(image, 1)
            cv2.imwrite(f"../data/hand_images_real/full/left/{count:08d}.jpg", image)
    except Exception:
        pass
    count += 1
