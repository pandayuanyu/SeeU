import math

import numpy as np

import open3d as o3d
import trimesh

import torch
import torch.nn.functional as F
from torch import Tensor

from tqdm import tqdm

def focus_point_fn(
    poses: np.ndarray,
) -> np.ndarray:
    """
    Calculate nearest point to all focal axes in poses.
    """
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def transform_poses_pca(
    poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms poses so principal components lie on XYZ axes.

    Args:
        poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
        A tuple (poses, transform), with the transformed poses and the applied
        camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot
    
    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform

def to_cam_open3d(viewpoint_stack, Ks, W, H):
    camera_traj = []
    for i, (extrinsic, intrins) in enumerate(zip(viewpoint_stack, Ks)):

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=H,
            height=W,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic = extrinsic.cpu().numpy()
        
        extrinsic = np.linalg.inv(extrinsic)

        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj

class MeshExtractor(object):

    def __init__(
        self, 
        #TODO (WZ): parse Gaussian model in gsplat 
        # voxel_size: float,
        # depth_trunc: float,
        # sdf_trunc: float,
        # num_cluster: float,
        # mesh_res: int,   
        bg_color: Tensor=None,
    ):
        """
        Mesh extraction class for gsplat Gaussians model

        TODO (WZ): docstring...
        """
        if bg_color is None:
            bg_color = [0., 0., 0.]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.clean()

    @torch.no_grad()
    def set_viewpoint_stack(
        self,
        viewpoint_stack: torch.Tensor,
    ) -> None:
        self.viewpoint_stack = viewpoint_stack

    @torch.no_grad()
    def set_Ks(
        self,
        Ks: torch.Tensor,
    ) -> None:
        self.Ks = Ks

    @torch.no_grad()
    def set_rgb_maps(
        self,
        rgb_maps: torch.Tensor,
    ) -> None:
        self.rgbmaps = rgb_maps

    @torch.no_grad()
    def set_depth_maps(
        self,
        depth_maps: torch.Tensor,
    ) -> None:
        self.depthmaps = depth_maps

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.rgbmaps = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(
        self,
        viewpoint_stack,
    ):
        """
        Render Gaussian Splatting given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg["render"]
            alpha = render_pkg["rend_alpha"]
            normal = torch.nn.functional.normalize(render_pkg["rend_normal"], dim=0)
            depth = render_pkg["surf_depth"]
            depth_normal = render_pkg["surf_normal"]
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())

        self.estimate_bounding_sphere()

    @torch.no_grad()
    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        torch.cuda.empty_cache()

        c2ws = np.array([np.asarray((camtoworld).cpu().numpy()) for camtoworld in self.viewpoint_stack])
        poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1]) # opengl to opencv?
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()

        print(f"The estimated bounding radius is: {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    
    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_background=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_background: whether to mask background, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f"voxel_size: {voxel_size}")
        print(f"sdf_trunc: {sdf_trunc}")
        print(f"depth_trunc: {depth_trunc}")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        W, H = self.rgbmaps.shape[1:3]


        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack, self.Ks, W, H)), desc="TSDF integration progress"):
    
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            import imageio

            surf_norm_save = rgb.detach().cpu()
            surf_norm_save = (surf_norm_save * 0.5 + 0.5)
            surf_norm_save = (surf_norm_save - torch.min(surf_norm_save)) / (torch.max(surf_norm_save) - torch.min(surf_norm_save))
            imageio.imwrite(f"./tmp.png", (surf_norm_save * 255).numpy().astype(np.uint8))

            surf_norm_save = depth.detach().cpu().repeat(1, 1, 3)
            surf_norm_save = (surf_norm_save * 0.5 + 0.5)
            surf_norm_save = (surf_norm_save - torch.min(surf_norm_save)) / (torch.max(surf_norm_save) - torch.min(surf_norm_save))
            imageio.imwrite(f"./tmp_depth.png", (surf_norm_save * 255).numpy().astype(np.uint8))
        
            
            # make open3d rgbd

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.cpu().numpy(), order="C")),
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0
            )
            
            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)
        
        mesh = volume.extract_triangle_mesh()
        return mesh