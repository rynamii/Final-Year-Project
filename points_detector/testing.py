import models
import cv2
import points_displayer
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Button
import os

# Load ml models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

l1_model = models.OccModel().to(device)
l1_model.load_state_dict(torch.load("models/points_20240314-012023.pt"))
l1_model.eval()

angle_model = models.OccModel().to(device)
angle_model.load_state_dict(torch.load("models/points_20240314-034337.pt"))
angle_model.eval()

bmc_model = models.OccModel().to(device)
bmc_model.load_state_dict(torch.load("models/points_20240314-060658.pt"))
bmc_model.eval()

# with open("val_paths.txt", "r") as f:
#     paths = [line.strip() for line in f.readlines()]

folder_path = "../data/HOnnotate/evaluation/MBC0/rgb/"
paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
paths = sorted(paths)

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Create a figure with two subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

# Counter for current image index
current_index = 0


def next_image(_):
    """
    Move to next image
    """
    global current_index
    current_index = (current_index + 1) % len(paths)
    display_image()


def previous_image(_):
    """
    Move to previous image
    """
    global current_index
    current_index = (current_index - 1) % len(paths)
    display_image()


def display_image():
    """
    Display image
    """
    # Load image
    global current_index
    path = paths[current_index]
    input_image = Image.open(path)
    input_image = transform(input_image) / 255

    # Predict joints
    l1_joints = l1_model(torch.stack([input_image]).to(device))
    angle_joints = angle_model(torch.stack([input_image]).to(device))
    bmc_joints = bmc_model(torch.stack([input_image]).to(device))

    # Load images
    l1_img = cv2.imread(path)
    angle_img = cv2.imread(path)
    bmc_img = cv2.imread(path)

    # Draw landmarks on images
    points_displayer.draw_landmarks(
        l1_img, l1_joints.cpu().detach().numpy().tolist()[0]
    )
    points_displayer.draw_landmarks(
        angle_img, angle_joints.cpu().detach().numpy().tolist()[0]
    )
    points_displayer.draw_landmarks(
        bmc_img, bmc_joints.cpu().detach().numpy().tolist()[0]
    )

    # Update the images on the subplots
    axes[0].imshow(cv2.cvtColor(l1_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("L1 Model")

    axes[1].imshow(cv2.cvtColor(angle_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Angle Model")

    axes[2].imshow(cv2.cvtColor(bmc_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title("BMC Model")

    # Hide the axes ticks
    for ax in axes:
        ax.axis("off")

    # Display the updated figure
    plt.draw()


# Create buttons for next and previous images
ax_next_button = plt.axes([0.55, 0.01, 0.1, 0.05])
button = Button(ax_next_button, "Next")
button.on_clicked(next_image)

ax_prev_button = plt.axes([0.35, 0.01, 0.1, 0.05])
button_prev = Button(ax_prev_button, "Previous")
button_prev.on_clicked(previous_image)


# Initial display
display_image()

# Show the figure
plt.show()
