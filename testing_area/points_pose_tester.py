# import json
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import cv2
from torchvision import transforms
import torch
from pose_finder.utils.model import HandModel
import numpy as np
from PIL import Image
from transforms3d.axangles import axangle2mat
from points_detector.models import OccModel
from pose_finder.models import kp2pose

HAND_COLOR = [120 / 255, 53 / 255, 32 / 255]
view_mat = axangle2mat([1, 0, 0], np.pi)

# # Load val paths
# with open("testing_area/val_paths.txt", "r") as f:
#     paths = [line.strip() for line in f.readlines()]

# Load ml models
device = "cuda" if torch.cuda.is_available() else "cpu"

pose_model = kp2pose().to(device)
pose_model.load_state_dict(torch.load("testing_area/models/kp2pose.pt"))
pose_model.eval()

l1_model = OccModel().to(device)
l1_model.load_state_dict(torch.load("testing_area/models/l1_model.pt"))
l1_model.eval()

angle_model = OccModel().to(device)
angle_model.load_state_dict(torch.load("testing_area/models/angle_model.pt"))
angle_model.eval()

bmc_model = OccModel().to(device)
bmc_model.load_state_dict(torch.load("testing_area/models/bmc_model.pt"))
bmc_model.eval()

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


class MeshViewer:
    """
    Visualise hand mesh for testing purposes
    """

    def __init__(self):

        # L1 window
        self.hand_meshL1 = HandModel(False, True)
        self.meshL1 = o3d.geometry.TriangleMesh()
        vertsL1, facesL1 = self.hand_meshL1._get_verts_faces()
        self.meshL1.triangles = o3d.utility.Vector3iVector(facesL1)
        self.meshL1.vertices = o3d.utility.Vector3dVector(
            np.matmul(view_mat, vertsL1.T).T * 1000
        )
        self.meshL1.compute_vertex_normals()

        self.l1_window = gui.Application.instance.create_window("L1 Hand", 600, 600)
        self.sceneL1 = gui.SceneWidget()
        self.sceneL1.scene = rendering.Open3DScene(self.l1_window.renderer)
        self.material = rendering.MaterialRecord()
        self.material.base_color = HAND_COLOR + [1]
        self.material.shader = "defaultLit"
        self.sceneL1.scene.add_geometry("l1 mesh", self.meshL1, self.material)

        self.l1_window.add_child(self.sceneL1)

        # Angles Window
        self.hand_meshA = HandModel(False, True)
        self.meshA = o3d.geometry.TriangleMesh()
        vertsA, facesA = self.hand_meshA._get_verts_faces()
        self.meshA.triangles = o3d.utility.Vector3iVector(facesA)
        self.meshA.vertices = o3d.utility.Vector3dVector(
            np.matmul(view_mat, vertsA.T).T * 1000
        )
        self.meshA.compute_vertex_normals()
        self.angle_window = gui.Application.instance.create_window(
            "Angle Hand", 600, 600
        )
        self.sceneA = gui.SceneWidget()
        self.sceneA.scene = rendering.Open3DScene(self.angle_window.renderer)
        self.material = rendering.MaterialRecord()
        self.material.base_color = HAND_COLOR + [1]
        self.material.shader = "defaultLit"
        self.sceneA.scene.add_geometry("angle mesh", self.meshA, self.material)
        self.angle_window.add_child(self.sceneA)

        # Angles Window
        self.hand_meshBMC = HandModel(False, True)
        self.meshBMC = o3d.geometry.TriangleMesh()
        vertsBMC, facesBMC = self.hand_meshBMC._get_verts_faces()
        self.meshBMC.triangles = o3d.utility.Vector3iVector(facesBMC)
        self.meshBMC.vertices = o3d.utility.Vector3dVector(
            np.matmul(view_mat, vertsBMC.T).T * 1000
        )
        self.meshBMC.compute_vertex_normals()
        self.bmc_window = gui.Application.instance.create_window(
            "BMC Hand", 600, 600
        )
        self.sceneBMC = gui.SceneWidget()
        self.sceneBMC.scene = rendering.Open3DScene(self.bmc_window.renderer)
        self.material = rendering.MaterialRecord()
        self.material.base_color = HAND_COLOR + [1]
        self.material.shader = "defaultLit"
        self.sceneBMC.scene.add_geometry("angle mesh", self.meshBMC, self.material)
        self.bmc_window.add_child(self.sceneBMC)

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

    def predict_pose(self, path, model):
        input_image = Image.open(path)
        input_image = transform(input_image) / 255
        joints = model(torch.stack([input_image]).to(device))
        pose = pose_model(joints)
        return pose.cpu().detach().numpy().reshape((48,))

    def next_hand(self):
        """
        Moves to the next hand
        """
        self.hand_count += 1

        # path = paths.pop(0)[3:]
        # while "right" not in path:
        #     path = paths.pop(0)[3:]

        path = f"data/hand_images_real/cropped/right/{self.hand_count:08d}.jpg"

        l1_pose = self.predict_pose(path, l1_model)
        angle_pose = self.predict_pose(path, angle_model)
        bmc_pose = self.predict_pose(path, bmc_model)

        self.hand_meshL1.pose_by_root([0, 0, 0], l1_pose, [])
        self.hand_meshA.pose_by_root([0, 0, 0], angle_pose, [])
        self.hand_meshBMC.pose_by_root([0, 0, 0], bmc_pose, [])

        img = cv2.imread(path)
        new_img = o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.image.update_image(new_img)
        self.update_scenes()

    def update_scenes(self):
        """
        Updates the scenes of both viewing windows
        """
        # Update L1
        verts, _ = self.hand_meshL1._get_verts_faces()
        self.meshL1.vertices = o3d.utility.Vector3dVector(
            np.matmul(view_mat, verts.T).T * 1000
        )
        self.meshL1.paint_uniform_color(HAND_COLOR)
        self.meshL1.compute_triangle_normals()
        self.meshL1.compute_vertex_normals()
        self.sceneL1.scene.clear_geometry()
        self.sceneL1.scene.add_geometry("l1 mesh", self.meshL1, self.material)

        # Update Angle
        verts, _ = self.hand_meshA._get_verts_faces()
        self.meshA.vertices = o3d.utility.Vector3dVector(
            np.matmul(view_mat, verts.T).T * 1000
        )
        self.meshA.paint_uniform_color(HAND_COLOR)
        self.meshA.compute_triangle_normals()
        self.meshA.compute_vertex_normals()
        self.sceneA.scene.clear_geometry()
        self.sceneA.scene.add_geometry("angle mesh", self.meshA, self.material)

        # Update BMC
        verts, _ = self.hand_meshBMC._get_verts_faces()
        self.meshBMC.vertices = o3d.utility.Vector3dVector(
            np.matmul(view_mat, verts.T).T * 1000
        )
        self.meshBMC.paint_uniform_color(HAND_COLOR)
        self.meshBMC.compute_triangle_normals()
        self.meshBMC.compute_vertex_normals()
        self.sceneBMC.scene.clear_geometry()
        self.sceneBMC.scene.add_geometry("BMC mesh", self.meshBMC, self.material)


def main():
    gui.Application.instance.initialize()
    MeshViewer()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
