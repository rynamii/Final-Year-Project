import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, coords, poses) -> None:
        self.coords = coords
        self.labels = poses

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return (self.coords[idx], self.labels[idx])


def frei_coord_from_index(data, index, is_left):
    index = index
    joints = data[index % 32560]

    if is_left:
        for idx, joint in enumerate(joints):
            joints[idx][0] = 1 - joint[0]
    return torch.tensor(joints)


def frei_pose_from_index(data, index):
    index = index
    pose = np.array(data[index % 32560], dtype=float)[:, :48].reshape(16, 3)

    return torch.tensor(pose)


def load_data(
    max_files: int = -1,
    freihand_path: str = "../data/freihand/",
):

    frei_length = 32560

    if max_files == -1:
        max_files = frei_length
    dataset_split = int(max_files)
    # frei_length = get_num_files(freihand_path, "right/training/rgb/")
    frei_right_indices = random.sample(range(frei_length), int(dataset_split))

    with open(f"{freihand_path}right/training_2d_points.json") as f:
        data = json.load(f)

    frei_right_coords = [
        frei_coord_from_index(data, idx, False) for idx in frei_right_indices
    ]

    with open(f"{freihand_path}right/training_mano.json", "r") as fi:
        data = json.load(fi)

    frei_right_poses = [frei_pose_from_index(data, idx) for idx in frei_right_indices]

    dataset = CustomDataset(coords=frei_right_coords, poses=frei_right_poses)

    return dataset
