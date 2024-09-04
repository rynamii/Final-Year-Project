import json
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import cv2

import torch
from utils.model import HandModel
import numpy as np
from transforms3d.axangles import axangle2mat

import models

HAND_COLOR = [228 / 255, 178 / 255, 148 / 255]
view_mat = axangle2mat([1, 0, 0], np.pi)

# Load Coordinates
with open(
    # "../data/freihand/right/training_2d_points.json"
    "../data/HO3D_Cropped/right/anno.json",
    "r"
) as f:
    p2d = json.load(f)

# Load mano pose
with open(
    # "../data/freihand/right/training_mano.json",
    "../data/HO3D_Cropped/right/mano.json",
    "r",
) as fi:
    mano = json.load(fi)

# Load ml model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.kp2pose().to(device)
model.load_state_dict(torch.load("models/kp2pose_20240303-150617.pt"))
model.eval()


class MeshViewer:
    """
    Visualise hand mesh for testing purposes
    """

    def __init__(self):

        # Testing window: holds test mesh from prediction
        self.hand_meshT = HandModel(False, True)
        self.meshT = o3d.geometry.TriangleMesh()
        vertsT, facesT = self.hand_meshT._get_verts_faces()
        self.meshT.triangles = o3d.utility.Vector3iVector(facesT)
        self.meshT.vertices = o3d.utility.Vector3dVector(
            np.matmul(view_mat, vertsT.T).T * 1000
        )
        self.meshT.compute_vertex_normals()

        self.test_window = gui.Application.instance.create_window(
            "Test Hand", 1080, 1080
        )
        self.sceneT = gui.SceneWidget()
        self.sceneT.scene = rendering.Open3DScene(self.test_window.renderer)
        self.material = rendering.MaterialRecord()
        self.material.base_color = HAND_COLOR + [1]
        self.material.shader = "defaultLit"
        self.sceneT.scene.add_geometry("test mesh", self.meshT, self.material)

        self.test_window.add_child(self.sceneT)

        # Static Window: Holds ground truth mesh
        self.hand_meshS = HandModel(False, True)
        self.meshS = o3d.geometry.TriangleMesh()
        vertsS, facesS = self.hand_meshS._get_verts_faces()
        self.meshS.triangles = o3d.utility.Vector3iVector(facesS)
        self.meshS.vertices = o3d.utility.Vector3dVector(
            np.matmul(view_mat, vertsS.T).T * 1000
        )
        self.meshS.compute_vertex_normals()
        self.static_window = gui.Application.instance.create_window(
            "Static Hand", 1080, 1080
        )
        self.sceneS = gui.SceneWidget()
        self.sceneS.scene = rendering.Open3DScene(self.static_window.renderer)
        self.material = rendering.MaterialRecord()
        self.material.base_color = HAND_COLOR + [1]
        self.material.shader = "defaultLit"
        self.sceneS.scene.add_geometry("static mesh", self.meshS, self.material)
        self.static_window.add_child(self.sceneS)

        # Create Editing Window
        self.edit_window = gui.Application.instance.create_window("Edit Hand", 300, 300)

        layout = gui.Vert()

        button = gui.Button("Next Hand")
        button.set_on_clicked(self.next_hand)
        layout.add_child(button)
        self.hand_count = -1

        self.image = gui.ImageWidget()

        layout.add_child(self.image)
        self.edit_window.add_child(layout)

        self.angles = np.zeros((16, 3))

    def next_hand(self):
        """
        Moves to the next hand
        """
        self.hand_count += 1

        m = np.array(mano.pop(0))
        pose = m.reshape((16, 3))

        pred_pose = model(torch.tensor([p2d.pop(0)]).to(device))
        pred_pose = pred_pose.cpu().detach().numpy().reshape((48,))
        self.angles = pred_pose.reshape((16, 3))

        angles_copy = self.angles.copy()
        self.hand_meshT.pose_by_root([0, 0, 0], angles_copy.reshape((48,)), [])
        self.hand_meshS.pose_by_root([0, 0, 0], pose.reshape((48,)), [])

        new_img = o3d.geometry.Image(
            # cv2.imread(f"../data/freihand/right/training/rgb/{self.hand_count:08d}.jpg")
            cv2.imread(f"../data/HO3D_Cropped/right/{self.hand_count:05d}.jpg")
            # self.hand_meshS.render()
        )
        self.image.update_image(new_img)
        self.update_scenes()

    def update_scenes(self):
        """
        Updates the scenes of both viewing windows
        """
        verts, _ = self.hand_meshT._get_verts_faces()
        self.meshT.vertices = o3d.utility.Vector3dVector(
            np.matmul(view_mat, verts.T).T * 1000
        )
        self.meshT.paint_uniform_color(HAND_COLOR)
        self.meshT.compute_triangle_normals()
        self.meshT.compute_vertex_normals()
        self.sceneT.scene.clear_geometry()
        self.sceneT.scene.add_geometry("test mesh", self.meshT, self.material)

        verts, _ = self.hand_meshS._get_verts_faces()
        self.meshS.vertices = o3d.utility.Vector3dVector(
            np.matmul(view_mat, verts.T).T * 1000
        )
        self.meshS.paint_uniform_color(HAND_COLOR)
        self.meshS.compute_triangle_normals()
        self.meshS.compute_vertex_normals()
        self.sceneS.scene.clear_geometry()
        self.sceneS.scene.add_geometry("static mesh", self.meshS, self.material)


def main():
    gui.Application.instance.initialize()
    MeshViewer()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
