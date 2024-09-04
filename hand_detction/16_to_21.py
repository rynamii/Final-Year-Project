import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math


def normalised_to_pixels(normalised_x, normalised_y, width, height):
    x_px = min(math.floor(normalised_x * width), width - 1)
    y_px = min(math.floor(normalised_y * height), height - 1)
    print(normalised_y, y_px)
    # print(norm)
    return [x_px, y_px]


image_path = "../data/freihand/right/training/rgb/00000000.jpg"
json_path = "../data/freihand/right/training_2d_points.json"

image = cv2.imread(image_path)

with open(json_path, "r") as f:
    data = json.load(f)

print(len(data[0]))


height = image.shape[1]
width = image.shape[0]

normalised_coords = data[0]

coords = []

for x, y, _ in normalised_coords:
    coords.append(
        normalised_to_pixels(x, y, width, height)
        )


edges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20]
    ]


pts = np.array(coords)
# invalid = pts[:,2]!=1

# Left hands are marked, but otherwise follow the same point order
# is_left = data['is_left']

# data[0] = pts.tolist()

# Plot annotations
plt.clf()

im = plt.imread(image_path)
plt.imshow(im)
count = 1
print(pts.shape)

print(pts)


for p in range(pts.shape[0]):
    # if pts[p,2]!=0:
    # print(count)
    count += 1
    # print(pts[p,0], pts[0,1])
    plt.plot(pts[p, 0], pts[p, 1])
    plt.text(pts[p, 0], pts[p, 1], '{0}'.format(p))
    print(p)
for ie, e in enumerate(edges):
    # if np.all(pts[e,2]!=0):
    rgb = matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0])
    plt.plot(pts[e, 0], pts[e, 1], color=rgb)

plt.show()
