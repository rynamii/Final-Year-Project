import numpy as np

MAPPING = [
    0,
    13,
    14,
    15,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
]

PARENTS = [
    # Root
    None,
    # Thumb
    0,
    1,
    2,
    3,
    # Index
    0,
    5,
    6,
    7,
    # Middle
    0,
    9,
    10,
    11,
    # Ring
    0,
    13,
    14,
    15,
    # Pinky
    0,
    17,
    18,
    19,
]


def _alpha(v1: np.ndarray, v2: np.ndarray) -> float:
    numerator = np.dot(v1.T, v2)
    denominator = np.sqrt(np.sum(np.square(v1))) * np.sqrt(np.sum(np.square(v2)))
    return np.arccos(numerator / denominator)


def _norm(x: np.ndarray):
    norm_2 = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm_2 + 1e-8)


def _interval_loss(x: float, a: float, b: float) -> float:
    return max(a - x, 0) + max(b - x, 0)


def _calculate_curvature(bones: list) -> list:
    """
    Calculates the curvature of a finger
    c_i = (((e_(i+1) - e_i)^T )(b_(i+1) - b_i)) / (||b_(i+1) - b_i||^2)
    For simplicity:
    c_i = ((q^T)(r)) / (||r||^2)
    Where:
    q = ((e_(i+1) - e_i)^T
    r = b_(i+1) - b_i
    """

    return


def _calculate_angular_distance(bones: dict, i: int) -> float:
    return _alpha(bones[i], bones[i + 1])


def _angle_between(v1, v2):
    v1_u = _norm(v1)
    v2_u = _norm(v2)

    inner = np.sum(v1_u * v2_u, axis=-1)
    tmp = np.clip(inner, -1, 1)
    tmp = np.arccos(tmp)
    return tmp


def axangle2mat(axis, angle):

    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    Q = np.array(
        [
            [x * xC + c, xyC - zs, zxC + ys],
            [xyC + zs, y * yC + c, yzC - xs],
            [zxC - ys, yzC + xs, z * zC + c],
        ]
    )
    Q = Q.transpose(2, 0, 1)

    return Q


def _calculate_joint_angles(
    mcp_bones, pip_bones, dip_bones, tip_bones, joint, all_bones
):
    normals = _norm(np.cross(mcp_bones[1:5], mcp_bones[0:4]))

    rot_mats = np.zeros((15, 3, 3))
    all_z_axis = _norm(all_bones)

    # ROOT Bones
    # pip_z_axis = _norm(mcp_bones)
    pip_z_axis = all_z_axis[[0, 4, 8, 12, 16]]
    pip_x_axis = np.zeros([5, 3])
    pip_x_axis[[0, 1, 4], :] = -normals[[0, 1, 3], :]
    pip_x_axis[2:4] = -_norm(normals[2:4] + normals[1:3])
    pip_y_axis = _norm(np.cross(pip_z_axis, pip_x_axis))

    pip_xz = pip_bones - np.sum(pip_bones * pip_y_axis, axis=-1, keepdims=True)
    pip_theta_f = _angle_between(pip_xz, pip_z_axis)
    pip_theta_a = _angle_between(pip_xz, pip_bones)

    # x-component of bone vector
    x_comp = np.sum((pip_bones * pip_x_axis), axis=-1)
    index = np.where(x_comp < 0)
    pip_theta_f[index] = -pip_theta_f[index]

    # y-component of bone vector
    y_comp = np.sum((pip_bones * pip_y_axis), axis=-1)
    index = np.where(y_comp < 0)
    pip_theta_a[index] = -pip_theta_a[index]

    axis = _norm(np.cross(pip_z_axis, pip_bones))
    alpha = _angle_between(pip_z_axis, pip_bones)
    rotation_mat = axangle2mat(axis, alpha)
    rot_mats[0:5] = rotation_mat

    # DIP bones
    # dip_z_axis = _norm(pip_bones)
    dip_z_axis = all_z_axis[[1, 5, 9, 13, 17]]
    dip_x_axis = np.matmul(rotation_mat, pip_x_axis[:, :, np.newaxis])
    dip_y_axis = np.matmul(rotation_mat, pip_y_axis[:, :, np.newaxis])
    dip_x_axis = np.squeeze(dip_x_axis)
    dip_y_axis = np.squeeze(dip_y_axis)

    dip_xz = dip_bones - np.sum(dip_bones * dip_y_axis, axis=-1, keepdims=True)
    dip_theta_f = _angle_between(dip_xz, dip_z_axis)
    dip_theta_a = _angle_between(dip_xz, dip_bones)

    # x-component of bone vector
    x_comp = np.sum((dip_bones * dip_x_axis), axis=-1)
    index = np.where(x_comp < 0)
    dip_theta_f[index] = -dip_theta_f[index]

    # y-component of bone vector
    y_comp = np.sum((dip_bones * dip_y_axis), axis=-1)
    index = np.where(y_comp < 0)
    dip_theta_a[index] = -dip_theta_a[index]

    axis = _norm(np.cross(dip_z_axis, dip_bones))
    alpha = _angle_between(dip_z_axis, dip_bones)
    rotation_mat = axangle2mat(axis, alpha)
    rot_mats[5:10] = rotation_mat
    # exit()

    # TIP bones
    # tip_z_axis = _norm(dip_bones)
    tip_z_axis = all_z_axis[[2, 6, 10, 14, 18]]
    tip_x_axis = np.matmul(rotation_mat, dip_x_axis[:, :, np.newaxis])
    tip_y_axis = np.matmul(rotation_mat, dip_y_axis[:, :, np.newaxis])
    tip_x_axis = np.squeeze(tip_x_axis)
    tip_y_axis = np.squeeze(tip_y_axis)

    tip_xz = tip_bones - np.sum(tip_bones * tip_y_axis, axis=-1, keepdims=True)
    tip_theta_f = _angle_between(tip_xz, tip_z_axis)
    tip_theta_a = _angle_between(tip_xz, tip_bones)

    # x-component of bone vector
    x_comp = np.sum((tip_bones * tip_x_axis), axis=-1)
    index = np.where(x_comp < 0)
    tip_theta_f[index] = -tip_theta_f[index]

    # y-component of bone vector
    y_comp = np.sum((tip_bones * tip_y_axis), axis=-1)
    index = np.where(y_comp < 0)
    tip_theta_a[index] = -tip_theta_a[index]

    axis = _norm(np.cross(tip_z_axis, tip_bones))
    alpha = _angle_between(tip_z_axis, tip_bones)
    rotation_mat = axangle2mat(axis, alpha)
    rot_mats[10:15] = rotation_mat

    angles = np.zeros((16, 3))
    _add_angles(angles, pip_theta_f, pip_theta_a, 1)
    _add_angles(angles, dip_theta_f, dip_theta_a, 2)
    _add_angles(angles, tip_theta_f, tip_theta_a, 3)
    return angles


def _root_bone_loss(root_bones: dict):
    curves = _calculate_curvature(root_bones)
    print(curves)
    angular_distances = []
    for i in range(0, 4):
        theta = _alpha(root_bones[i], root_bones[i + 1])
        angular_distances.append(theta)
    print(angular_distances)
    return


def _get_fingers(joints: np.ndarray) -> dict:
    fingers = {}
    for i in range(1, 6):
        fingers[i] = []
        for j in range(4 * (i - 1), 4 * i):
            fingers[i].append(joints[j + 1] - joints[PARENTS[j + 1]])
    return fingers


def _add_angles(angles_array, flex_array, abduct_array, index):
    order = [1, 2, 4, 3, 0]
    flex_array = np.array(flex_array)[order]
    abduct_array = np.array(abduct_array)[order]

    for i in range(0, 5):
        angles_array[3 * i + index] = [0, abduct_array[i], flex_array[i]]


def get_pose(xyz: np.ndarray):
    xyz = xyz * 10
    fingers = _get_fingers(xyz)
    all_bones = np.concatenate(list(fingers.values()))

    mcp_bones = [finger[0] for finger in fingers.values()]  # Root bones
    pip_bones = [finger[1] for finger in fingers.values()]
    dip_bones = [finger[2] for finger in fingers.values()]
    tip_bones = [finger[3] for finger in fingers.values()]

    angles = _calculate_joint_angles(
        mcp_bones, pip_bones, dip_bones, tip_bones, xyz, all_bones
    )
    return angles
