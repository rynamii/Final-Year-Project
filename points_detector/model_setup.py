from PIL import Image
import os
import random
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch


class CustomDataset(Dataset):
    def __init__(self, paths, labels) -> None:
        """
        Initializes CustomDatset.

        Args:
            `paths`: Full set of paths.
            `labels`: Full set of labels.
        """
        self.paths = paths
        self.labels = labels
        self.data = self.preprocess_data()

    def __len__(self):
        return len(self.data)

    def preprocess_data(self):
        """
        Load images from paths and pair with labels
        """
        processed_data = []
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # Wrap the loop with tqdm to add a progress bar
        for image_path in tqdm(self.paths, desc="Processing Data", unit="image"):
            image = Image.open(image_path)
            image = transform(image) / 255
            processed_data.append((image, self.labels[len(processed_data)]))

        return processed_data

    def __getitem__(self, idx):
        return self.data[idx]


def get_full_paths(base_path, subdirectory):
    """
    Get the full paths of every file in `base_path/subdirectory/`.

    Args:
        `base_path`: The path to the base folder
        `subdirectory`: The subdirectory to get the paths from

    Returns:
        The full sorted list of paths
    """
    directory_path = os.path.join(base_path, subdirectory)
    paths = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
    ]
    return sorted(paths)


def frei_label_from_index(data, index, is_left):
    """
    Load the data.

    Args:
        `data`: The full set of labels
        `index`: The index of data wanted
        `is_left`: If the hand is left or not

    Returns:
        The label for the given index
    """
    index = index
    joints = data[index % 32560]

    if is_left:
        for idx, joint in enumerate(joints):
            joints[idx][0] = 1 - joint[0]
    return torch.tensor(joints)


def load_data(
    type: str = "all",
    max_files: int = 10000,
    freihand_path: str = "../data/freihand/",
    ho3d_path: str = "../data/HO3D_Cropped/",
) -> CustomDataset:
    """
    Load the data.

    Args:
        `type`: Whether to load, left, right, or both hands.
        `max_files`: The maximum number of files to load
        `freihand_path`: The path to the freihand dataset
        `ho3d_path`: The path to the HOnnotate dataset

    Returns:
        The dataset to be used
    """
    frei_left_list = get_full_paths(freihand_path, "left/training/rgb/")
    frei_right_list = get_full_paths(freihand_path, "right/training/rgb/")

    ho3d_left_list = get_full_paths(ho3d_path, "left/")
    ho3d_right_list = get_full_paths(ho3d_path, "right/")

    if type == "all":
        left_split = 4
        right_split = 4
    elif type == "left":
        left_split = 2
        right_split = 0
    elif type == "right":
        left_split = 0
        right_split = 2
    else:
        raise ValueError(f"Invalid type: '{type}', expected 'all', 'left', or 'right'")
    dataset_split = int(max_files)

    all_paths = []
    all_labels = []

    if left_split != 0:
        frei_left_indices = random.sample(
            range(len(frei_left_list)), int(dataset_split / left_split)
        )
        ho3d_left_indices = random.sample(
            range(len(ho3d_left_list) - 2), int(dataset_split / left_split)
        )

        frei_left_paths = [frei_left_list[idx] for idx in frei_left_indices]
        ho3d_left_paths = [ho3d_left_list[idx] for idx in ho3d_left_indices]

        with open("../data/freihand/right/training_2d_points.json") as f:
            data = json.load(f)

        frei_left_labels = [
            frei_label_from_index(data, idx, True) for idx in frei_left_indices
        ]

        with open(f"{ho3d_path}left/anno.json") as f:
            data = json.load(f)
            ho3d_left_labels = [torch.tensor(data[idx]) for idx in ho3d_left_indices]

        all_paths += frei_left_paths + ho3d_left_paths
        all_labels += frei_left_labels + ho3d_left_labels

    if right_split != 0:
        frei_right_indices = random.sample(
            range(len(frei_right_list)), int(dataset_split / right_split)
        )
        ho3d_right_indices = random.sample(
            range(len(ho3d_right_list) - 2), int(dataset_split / right_split)
        )

        frei_right_paths = [frei_right_list[idx] for idx in frei_right_indices]
        ho3d_right_paths = [ho3d_right_list[idx] for idx in ho3d_right_indices]

        with open("../data/freihand/right/training_2d_points.json") as f:
            data = json.load(f)

        frei_right_labels = [
            frei_label_from_index(data, idx, False) for idx in frei_right_indices
        ]

        with open(f"{ho3d_path}right/anno.json") as f:
            data = json.load(f)
            ho3d_right_labels = [torch.tensor(data[idx]) for idx in ho3d_right_indices]

        all_paths += frei_right_paths + ho3d_right_paths
        all_labels += frei_right_labels + ho3d_right_labels

    dataset = CustomDataset(paths=all_paths, labels=all_labels)

    return dataset
