import cv2
from typing import List
import math
import numpy as np

HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))

HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))

HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))

HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))

HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))

HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

HAND_CONNECTIONS = frozenset().union(
    *[
        HAND_PALM_CONNECTIONS,
        HAND_THUMB_CONNECTIONS,
        HAND_INDEX_FINGER_CONNECTIONS,
        HAND_MIDDLE_FINGER_CONNECTIONS,
        HAND_RING_FINGER_CONNECTIONS,
        HAND_PINKY_FINGER_CONNECTIONS,
    ]
)


def _normalized_to_pixel_coordinates(
    norm_x: float, norm_y: float, image_width: int, image_height: int
):
    """
    Converts normalised coordinates to pixel coordinates.

    Args:
        `norm_x`: Normalised x
        `norm_y`: Normalised y
        `image_width`: Image width
        `image_height`: Image height
    Returns:
        Pixel coordinate pair
    """
    x_px = min(math.floor(norm_x * image_width), image_width - 1)
    y_px = min(math.floor(norm_y * image_height), image_height - 1)
    return x_px, y_px


def draw_landmarks(image: np.ndarray, landmarks: List):
    """
    Draws landmarks onto an image

    Args:
        `image`: Image to draw on
        `landmarks`: Landmarks to be drawn
    """
    image_rows, image_cols, _ = image.shape

    # Convert landmarks
    coords = []
    for keypoint in landmarks:
        keypoint_px = _normalized_to_pixel_coordinates(
            keypoint[0], keypoint[1], image_cols, image_rows
        )
        coords.append(keypoint_px)

    # Draw lines connecting landmarks
    for connection in HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        cv2.line(image, coords[start_idx], coords[end_idx], (224, 224, 224), 2)

    # Draw points of landmarks
    for landmark_px in coords:
        cv2.circle(image, landmark_px, 2, (224, 224, 224), 2)
        cv2.circle(image, landmark_px, 2, (0, 128, 0), 2)
