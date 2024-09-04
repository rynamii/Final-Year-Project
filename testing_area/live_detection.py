import cv2
from points_detector.models import OccModel
from points_detector.points_displayer import draw_landmarks
from pose_finder.models import kp2pose
from hand_detction.models import HandDetectionModel
import open3d as o3d
from transforms3d.axangles import axangle2mat
from pose_finder.utils.model import HandModel
import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F
import re
import subprocess
import argparse
import os
from collections import deque, Counter

classes = ["Null", "Left", "Right"]

# How many frames to make hand prediction from
window_size = 10

# Minimum confidence for a label
threshold = 0.4

# Initialise a buffer for hand detection
# conf_buffers = {class_name: deque(maxlen=window_size) for class_name in classes}
conf_buffers = deque(maxlen=window_size)


def get_hand_label(model_result):
    """
    Get the hand detected based on the latest items in window.

    Args:
        `model_result`: Result given by model.

    Returns:
        Predicted label.
    """

    # # Add latest results to buffers
    # for class_name, buffer in conf_buffers.items():
    #     buffer.append(model_result[classes.index(class_name)])

    # # Average the confidences over the window
    # avg_conf = {
    #     class_name: sum(buffer) / len(buffer)
    #     for class_name, buffer in conf_buffers.items()
    # }

    # # Give the final label
    # final_label = (
    #     classes.index(max(avg_conf, key=avg_conf.get))
    #     # Only return if label is greater than threshold, otherwise return null
    #     if avg_conf[max(avg_conf, key=avg_conf.get)] >= threshold
    #     else 0
    # )

    # if avg_conf["Null"] < 1 / 3:
    #     final_label = avg_conf[max(avg_conf, key=avg_conf.get)]
    # else:
    #     final_label = avg_conf[max(avg_conf, key=avg_conf.get)]

    if model_result[0] > threshold:
        label = 0
    else:
        label = model_result.index(max(model_result[1:]))

    conf_buffers.append(label)

    final_label = Counter(conf_buffers).most_common(1)[0][0]

    return final_label


def app(flip, folder, labels):
    if folder:
        folder_path = f"data/HOnnotate/train/{folder}/rgb/"
        paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        paths = sorted(paths)
        print(f"Detecting on premade frames from: {folder_path}")
    else:
        print("Detecting on live data")

    labels = sorted(labels)
    print("Detecting hands on labels:", ", ".join([classes[val] for val in labels]))

    # Load ml models
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pose_model = kp2pose().to(device)
    pose_model.load_state_dict(torch.load("testing_area/models/kp2pose.pt"))
    pose_model.eval()

    points_model = OccModel().to(device)
    points_model.load_state_dict(torch.load("testing_area/models/l1_model.pt"))
    points_model.eval()

    detection_model = HandDetectionModel().to(device)
    detection_model.load_state_dict(
        torch.load("testing_area/models/detection_model.pt")
    )
    detection_model.eval()

    # Set up for pose window

    HAND_COLOR = [228 / 255, 178 / 255, 148 / 255]

    view_mat = axangle2mat([1, 0, 0], np.pi)
    window_size = 1080

    hand_mesh = HandModel(False, False)
    mesh = o3d.geometry.TriangleMesh()
    verts, faces = hand_mesh._get_verts_faces()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, verts.T).T * 1000)
    mesh.compute_vertex_normals()

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
        width=window_size + 1,
        height=window_size + 1,
        window_name="Mesh",
    )
    viewer.add_geometry(mesh)
    view_control = viewer.get_view_control()
    view_control.set_constant_z_far(2000)

    # Translate such that hand is center of frame
    view_control.camera_local_translate(-110, 80, 0)

    render_option = viewer.get_render_option()
    render_option.load_from_json("testing_area/RenderOption.json")
    viewer.update_renderer()

    # Using a virtual camera due to running in wsl
    # Extract IPv4 address using regex
    if not folder:
        result = subprocess.run(["ipconfig.exe"], capture_output=True, text=True)
        ipv4_pattern = re.compile(r"IPv4 Address[^\d]+(\d+\.\d+\.\d+\.\d+)")
        match = ipv4_pattern.search(result.stdout)
        ip = match.group(1)
        cap = cv2.VideoCapture(f"http://{ip}:8000")

    # Convert image to PIL, resize, and then convert to tensor
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    counter = 0
    # Initialise joints if hand not detected on first frame
    joints = torch.zeros((1, 21, 3)).to(device)

    while True:
        if folder:
            if counter == len(paths):
                counter = 0
            image = cv2.imread(paths[counter])
        else:
            _, image = cap.read()

        # Handle skipped frames
        if image is None:
            continue

        # Limit the number of frames calculated to prevent jitter from high frame rate
        counter += 1
        if counter % 3 != 0:
            continue

        # Mirrors image for selfie view
        if flip:
            image = cv2.flip(image, 1)

        if not folder:
            # Limit to center square as model performs better on squares
            image = image[290:790, 1210:1710]

        # Applies transformation
        transformed_image = transform(image) / 255

        # Check for hand
        output = detection_model(torch.stack([transformed_image]).to(device))
        softmax = F.softmax(output, dim=1).tolist()[0]
        label = get_hand_label(softmax)

        # Display label
        cv2.putText(
            image,
            f"Final Label: {classes[label]}",
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Only find points if within labels
        if label in labels:

            # Predict joints
            joints = points_model(torch.stack([transformed_image]).to(device))

            # Draw keypoints
            draw_landmarks(image, joints.cpu().detach().numpy().tolist()[0])
        cv2.imshow("Landmark Detection", image)

        # Predict pose
        pose = pose_model(joints)
        pose = pose.cpu().detach().numpy().reshape((48,))

        # Update mesh
        hand_mesh.pose_by_root([0, 0, 0], pose, [])
        verts, _ = hand_mesh._get_verts_faces()
        mesh.vertices = o3d.utility.Vector3dVector(
            np.matmul(view_mat, verts.T).T * 1000
        )
        mesh.paint_uniform_color(HAND_COLOR)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        viewer.update_geometry(mesh)
        viewer.poll_events()

        if folder:
            cv2.waitKey(33)

        # Exit on escape or window close
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Cleanup
    if not folder:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--flip", action="store_false", help="turn off image flip"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Folder for preloaded frames, otherwise does live",
        default=None,
    )
    parser.add_argument(
        "--labels",
        metavar="N",
        type=int,
        nargs="*",
        choices=[0, 1, 2],
        help="Labels to detect hands for\n\
            0 - Null\n\
            1 - Right\n\
            2 - Left",
        default=[0, 1, 2],
    )
    args = parser.parse_args()

    if len(args.labels) < 1:
        raise argparse.ArgumentTypeError("At least one label is required")

    app(args.flip, args.folder, args.labels)
