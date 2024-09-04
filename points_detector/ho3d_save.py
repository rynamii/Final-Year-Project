import cv2
import numpy as np
import pickle
import random


def _project_points(pts3D, cam_mat):
    coord_change_mat = np.array(
        [[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
    )

    pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack(
        [
            proj_pts[:, 0] / proj_pts[:, 2],
            proj_pts[:, 1] / proj_pts[:, 2],
            proj_pts[:, 2],
        ],
        axis=1,
    )

    return proj_pts


def _order_points(proj_pts):
    reorder = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    proj_pts = proj_pts[reorder]
    return proj_pts


def _normalise_points(proj_pts, left_boundary, shape):
    proj_pts[:, 0] = (proj_pts[:, 0] - left_boundary) / shape[1]
    proj_pts[:, 1] = proj_pts[:, 1] / shape[0]
    return proj_pts


def _image_crop(image, xyz):
    min_x = max(0, xyz[:, 0].min())
    max_x = xyz[:, 0].max()

    valid_range = (min_x, max_x - 480)

    left_boundary = max(0, int(random.uniform(valid_range[0], valid_range[1])))
    left_boundary = min(left_boundary, (640 - 480))

    right_boundary = left_boundary + 480

    cropped_image = image[:, left_boundary:right_boundary, :]
    return cropped_image, left_boundary


total = 0

right_xyz = []
left_xyz = []


with open("../data/HOnnotate/train.txt", "r") as f:
    paths = f.readlines()

for index, path in enumerate(paths):
    sequence, idx = path[:-1].split("/")
    print(f"{index+1}/{len(paths)}")

    image = cv2.imread(f"../data/HOnnotate/train/{sequence}/rgb/{idx}.jpg")

    with open(f"../data/HOnnotate/train/{sequence}/meta/{idx}.pkl", "rb") as f:
        data = pickle.load(f)
    total += 1

    right = _project_points(data["rightHandJoints3D"], data["camMat"])
    left = _project_points(data["leftHandJoints3D"], data["camMat"])

    if np.sum(right[:, 2] > left[:, 2]) < 21 / 2:
        if np.all(data["jointValidRight"]) == 1:
            cropped_image, left_boundary = _image_crop(image, right.copy())
            norm_pts = _normalise_points(
                right.copy(), left_boundary, cropped_image.shape
            )
            ordered_points = _order_points(norm_pts)
            cv2.imwrite(
                f"../data/HO3D_Cropped/right/{len(right_xyz):05d}.jpg", cropped_image
            )
            right_xyz.append(ordered_points.tolist())

    else:
        if np.all(data["jointValidLeft"]) == 1:
            cropped_image, left_boundary = _image_crop(image, left.copy())
            norm_pts = _normalise_points(
                left.copy(), left_boundary, cropped_image.shape
            )
            ordered_points = _order_points(norm_pts)
            cv2.imwrite(
                f"../data/HO3D_Cropped/left/{len(left_xyz):05d}.jpg", cropped_image
            )
            left_xyz.append(ordered_points.tolist())

with open("../data/HO3D_Cropped/right/anno.json", "w") as f:
    f.write(str(right_xyz))

with open("../data/HO3D_Cropped/left/anno.json", "w") as f:
    f.write(str(left_xyz))

print(f"Right: {len(right_xyz)}, Left: {len(left_xyz)}")
