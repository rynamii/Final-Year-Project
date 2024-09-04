from PIL import Image
import os
from os import listdir
import random
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm


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
    Load the data.

    Args:
        `base_path`: The path to the base folder
        `subdirectory`: The subdirectory to get the paths from

    Returns:
        The full list of paths
    """
    directory_path = os.path.join(base_path, subdirectory)
    return [
        os.path.join(directory_path, filename) for filename in listdir(directory_path)
    ]


def load_data(
    max_files: int = 10000,
    freihand_path: str = "../data/freihand/",
    real_path: str = "../data/hand_images_real/",
    synth_path: str = "../data/hand_images_synth/",
    pass_path: str = "../data/pass/images/",
) -> CustomDataset:
    """
    Load the data.

    Args:
        `max_files`: The maximum number of files to load
        `freihand_path`: The path to the freihand dataset
        `real_path`: The path to real hands dataset
        `synth_path`: The path to synthetic hands dataset
        `pass_path`: The path to PASS dataset

    Returns:
        The dataset to be used
    """
    class_split = int(max_files / 3)

    pass_list = listdir(pass_path)

    left_list = (
        get_full_paths(freihand_path, "left/training/rgb/")
        + get_full_paths(real_path, "cropped/left")
        + get_full_paths(real_path, "full/left")
        + get_full_paths(synth_path, "cropped/left")
        + get_full_paths(synth_path, "full/left")
    )

    right_list = (
        get_full_paths(freihand_path, "right/training/rgb/")
        + get_full_paths(real_path, "cropped/right")
        + get_full_paths(real_path, "full/right")
        + get_full_paths(synth_path, "cropped/right")
        + get_full_paths(synth_path, "full/right")
    )

    pass_indices = random.sample(range(len(pass_list)), class_split)
    left_indices = random.sample(range(len(left_list)), class_split)
    right_indices = random.sample(range(len(right_list)), class_split)

    pass_paths = [pass_path + pass_list[idx] for idx in pass_indices]
    left_paths = [left_list[idx] for idx in left_indices]
    right_paths = [right_list[idx] for idx in right_indices]

    all_paths = pass_paths + left_paths + right_paths
    all_labels = [0] * class_split + [1] * class_split + [2] * class_split

    dataset = CustomDataset(paths=all_paths, labels=all_labels)

    return dataset
