import transforms3d as t3d
import chumpy as ch
import numpy as np
from opendr.lighting import LambertianPointLight

try:
    from utils.mano_core.mano_loader import load_model
    from utils.mano_utils import get_keypoints_from_mesh_ch
except ImportError as e:
    raise ImportError("%s \nDid you set up the repository for advanced use?" % e)


class HandModel(object):
    def __init__(self, use_mean_pca=False, use_mean_pose=False):
        if use_mean_pca:
            self.model = load_model(
                "./data/MANO_RIGHT.pkl",
                ncomps=6,
                flat_hand_mean=not use_mean_pose,
                use_pca=True,
            )
        else:
            self.model = load_model(
                "./data/MANO_RIGHT.pkl",
                ncomps=48,
                flat_hand_mean=not use_mean_pose,
                use_pca=False,
            )

        self.global_trans = ch.array([0.0, 0.0, 0.3])

    def _get_verts_faces(self):
        V = self.model + self.global_trans
        F = self.model.f
        return V, F

    def _calc_coords(self):
        # calculate joint location and rotation
        V, _ = self._get_verts_faces()
        J_regressor = self.model.dd["J_regressor"]
        Jtr_x = ch.MatVecMult(J_regressor, V[:, 0])
        Jtr_y = ch.MatVecMult(J_regressor, V[:, 1])
        Jtr_z = ch.MatVecMult(J_regressor, V[:, 2])
        Jtr = ch.vstack([Jtr_x, Jtr_y, Jtr_z]).T
        coords_kp_xyz = get_keypoints_from_mesh_ch(V, Jtr)
        return coords_kp_xyz

    def pose_by_root(self, xyz_root, poses, shapes, root_id=9):
        """Poses the MANO model according to the root keypoint given."""
        self.model.pose[:] = poses  # set estimated articulation
        try:
            self.model.betas[:] = shapes
        except ValueError:
            self.model.betas[:] = np.zeros((10,))
        # self.global_trans[:] = 0.0

        # how to chose translation
        # xyz = np.array(self._calc_coords())
        # global_t = xyz_root - xyz[root_id]  # new - old root keypoint

        # self.global_trans[:] = global_t

    def render(self, dist=None, M=None, img_shape=None, render_mask=False):
        from opendr.camera import ProjectPoints
        from utils.renderer import ColoredRenderer

        if dist is None:
            dist = np.zeros(5)
        dist = dist.flatten()
        if M is None:
            M = np.eye(4)

        # get R, t from M (has to be world2cam)
        R = M[:3, :3]
        ax, angle = t3d.axangles.mat2axangle(R)
        rt = ax * angle
        rt = rt.flatten()
        t = M[:3, 3]

        w, h = (320, 320)
        if img_shape is not None:
            w, h = img_shape[1], img_shape[0]

        # pp = np.array([cam_intrinsics[0, 2], cam_intrinsics[1, 2]])
        # f = np.array([cam_intrinsics[0, 0], cam_intrinsics[1, 1]])

        # Create OpenDR renderer
        rn = ColoredRenderer()

        # Assign attributes to renderer
        rn.camera = ProjectPoints(
            rt=rt,
            t=t,  # camera translation
            f=[300, 300],  # focal lengths
            c=[112, 112],  # camera center (principal point)
            k=dist,
        )  # OpenCv distortion params
        rn.frustum = {"near": 0.1, "far": 5.0, "width": w, "height": h}

        V, F = self._get_verts_faces()
        rn.set(v=V, f=F, bgcolor=np.zeros(3))

        if render_mask:
            rn.vc = np.ones_like(V)  # for segmentation mask like rendering
        else:
            colors = np.ones_like(V)

            # Construct point light sources
            rn.vc = LambertianPointLight(
                f=F,
                v=V,
                num_verts=V.shape[0],
                light_pos=np.array([-1000, -1000, -2000]),
                vc=0.8 * colors,
                light_color=np.array([1.0, 1.0, 1.0]),
            )

            rn.vc += LambertianPointLight(
                f=F,
                v=V,
                num_verts=V.shape[0],
                light_pos=np.array([1000, 1000, -2000]),
                vc=0.25 * colors,
                light_color=np.array([1.0, 1.0, 1.0]),
            )

            rn.vc += LambertianPointLight(
                f=F,
                v=V,
                num_verts=V.shape[0],
                light_pos=np.array([2000, 2000, 2000]),
                vc=0.1 * colors,
                light_color=np.array([1.0, 1.0, 1.0]),
            )

            rn.vc += LambertianPointLight(
                f=F,
                v=V,
                num_verts=V.shape[0],
                light_pos=np.array([-2000, -2000, 2000]),
                vc=0.1 * colors,
                light_color=np.array([1.0, 1.0, 1.0]),
            )

        # render
        img = (np.array(rn.r) * 255).astype(np.uint8)
        return img
