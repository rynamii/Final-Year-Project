import cv2
import json
import numpy as np
import os
from tqdm import tqdm

hands_dir = "../data/hand_labels_synth/"

files = []
for file in os.listdir(hands_dir):
    if os.path.isdir(hands_dir+file) and "output" not in file:
        for sub_file in os.listdir(hands_dir+file):
            if "jpg" not in sub_file:
                files.append(hands_dir+file+"/"+sub_file)

count = 0
for file in tqdm(files):
    with open(file, "r") as f:
        data = json.load(f)

    array = np.array(data["hand_pts"])

    # array = np.delete(array, 2, 1)
    array = array[array[:, 2] != 0]
    max_values = np.amax(array, axis=0)
    min_values = np.amin(array, axis=0)

    n = 30  # You can adjust this value as needed
    min_values -= n
    max_values += n

    # Calculate the center of the rectangle
    center = ((min_values[0] + max_values[0]) // 2, (min_values[1] + max_values[1]) // 2)

    # Determine the size of the square based on the larger dimension of the rectangle
    width = max(max_values[0] - min_values[0], max_values[1] - min_values[1])

    # Calculate the coordinates of the top-left and bottom-right corners of the square
    square_top_left = (center[0] - width // 2, center[1] - width // 2)
    square_bottom_right = (center[0] + width // 2, center[1] + width // 2)

    # Draw the rectangle and square on the image
    image = cv2.imread(file[0:-5]+".jpg")
    cropped_image = image[
        int(square_top_left[1]):int(square_bottom_right[1]), int(square_top_left[0]):int(square_bottom_right[0])
        ]
    try:
        if (data["is_left"]):
            cv2.imwrite(f"../data/hand_images_synth/left/{count:08d}.jpg", cropped_image)
            cropped_image = cv2.flip(cropped_image, 1)
            cv2.imwrite(f"../data/hand_images_synth/right/{count:08d}.jpg", cropped_image)
        else:
            cv2.imwrite(f"../data/hand_images_synth/right/{count:08d}.jpg", cropped_image)
            cropped_image = cv2.flip(cropped_image, 1)
            cv2.imwrite(f"../data/hand_images_synth/left/{count:08d}.jpg", cropped_image)
    except Exception:
        pass
    count += 1
