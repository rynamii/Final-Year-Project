import json

# import numpy as np
import cv2
import mediapipe as mp
import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
from google.protobuf import text_format

with open("../data/freihand/right/training_2d_points.json") as f:
    data = json.load(f)

print(len(data))

image = cv2.imread("../data/freihand/left/training/rgb/00111111.jpg")
joints = data[111111 % 32560]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

landmarks = ""
for X, Y, Z in joints:
    formatted_string = f"landmark {{x: {1-X} y: {Y} z: {Z}}} "
    landmarks = landmarks + formatted_string

hand_landmarks = text_format.Parse(landmarks, landmark_pb2.NormalizedLandmarkList())

mp_drawing.draw_landmarks(
    image,
    hand_landmarks,
    mp_hands.HAND_CONNECTIONS,
    mp_drawing_styles.get_default_hand_landmarks_style(),
    mp_drawing_styles.get_default_hand_connections_style(),
)

cv2.imshow("fjsfdbnvjks", image)
cv2.waitKey(0)
