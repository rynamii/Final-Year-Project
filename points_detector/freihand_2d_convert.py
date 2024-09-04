import cv2
import json
import numpy as np


def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


def renderPose(img, uv):
    connections = [[0, 1], [1, 2], [2, 3], [3, 4],
                   [0, 5], [5, 6], [6, 7], [7, 8],
                   [0, 9], [9, 10], [10, 11], [11, 12],
                   [0, 13], [13, 14], [14, 15], [15, 16],
                   [0, 17], [17, 18], [18, 19], [19, 20]]
    for c in connections:
        img = cv2.line(img, uv[c[0]], uv[c[1]], (255, 0, 0), 2)

    for point in uv:
        img = cv2.circle(img, point, 2, (0, 0, 255), -1)

    return img


with open("../data/freihand/right/training_xyz.json") as f:
    train_xyz = json.load(f)

with open("../data/freihand/right/training_K.json") as f:
    train_k = json.load(f)

new_xyz = []
for idx in range(len(train_k)):
    print(f"{idx}/{len(train_k)}")

    xyz_array = train_xyz[idx]
    k_array = train_k[idx]
    uv = projectPoints(xyz_array, k_array).astype(np.int32)

    img = cv2.imread(f"../data/freihand/right/training/rgb/{idx:08d}.jpg")
    image_width, image_height, _ = img.shape

    for idx, uv_coord in enumerate(uv):
        x_uv, y_uv = uv_coord
        xyz_array[idx][0] = x_uv/image_width
        xyz_array[idx][1] = y_uv/image_height

    new_xyz.append(xyz_array)

with open("../data/freihand/right/training_2d_points.json", "w") as outfile:
    outfile.write(str(new_xyz))
