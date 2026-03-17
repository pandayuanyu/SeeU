import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flow3d.transforms import cont_6d_to_rmat

###### Deprecated ######
class CameraScales(nn.Module):
    """Align the monst3r camera pose scale with the scene"""

    def __init__(
        self,
        camera_scales: torch.Tensor,
    ):
        super().__init__()
        self.params = nn.ParameterDict(
            {"camera_scales": camera_scales}
        )    

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        param_keys = ["camera_scales"]
        assert all(f"{prefix}{k}" in state_dict for k in param_keys)
        args = {k: state_dict[f"{prefix}{k}"] for k in param_keys}
        return CameraScales(**args)
    
    def get_camera_scales(self) -> torch.Tensor:
        return self.params["camera_scales"]


class CameraPoses(nn.Module):
    def __init__(
        self,
        Rs: Tensor,
        ts: Tensor,
    ):
        super().__init__()
        self.params = nn.ParameterDict(
            {
                "Rs": nn.Parameter(Rs),
                "ts": nn.Parameter(ts),
            }
        )

    
    @staticmethod
    def init_from_state_dict(
        state_dict: dict[str, Tensor],
        prefix: str,
    ):
        param_keys = ["Rs", "ts"]
        # import pdb
        # pdb.set_trace()
        assert all(f"{prefix}{k}" in state_dict for k in param_keys)
        args = {k: state_dict[f"{prefix}{k}"] for k in param_keys}
        return CameraPoses(**args)

    def get_rot_matrix(self):
        """convert a six number rotation representation to a SO(3) matrix"""
        Rs = self.params["Rs"]
        r1 = Rs[:, :, 0]
        r2 = Rs[:, :, 1]
        r1 = r1 / torch.norm(r1, dim=-1)[:, None]
        r2 = r2 - torch.sum(r1 * r2, dim=-1)[:, None] * r1
        r2 = r2 / torch.norm(r1, dim=-1)[:, None]

        r2 = r2 / torch.norm(r2, dim=-1)[:, None]
        r3 = torch.cross(r1, r2)

        return torch.stack([r1, r2, r3], dim=-1)

    def get_camera_matrix(self):
        """get the 3x4 camera pose"""
        rot_mats = self.get_rot_matrix()
        ts = self.params["ts"]
        pose = torch.cat([rot_mats, ts], dim=-1) # [..., 3, 4]
        homo_pad = torch.tensor([0., 0., 0., 1.]).repeat(pose.shape[0], 1, 1).to(pose.device)
        pose = torch.cat([
            pose,
            homo_pad], dim=-2) # (..., 4, 4)
        assert pose.shape[-2:] == (4, 4)
        return pose

    def invert(self, use_inverse: bool=False):
        """invert the camera pose"""
        Rs = self.params["Rs"]
        ts = self.params["ts"]
        Rs_inv = Rs.inverse() if use_inverse else R.transpose(-1, -2)
        ts_inv = (-R_inv @ t)[..., 0]
        return self(Rs_inv, ts_inv)

    def compose(self, pose):
        """Compose self pose with another pose"""
        Rs_, ts_ = pose.params["Rs"], pose.params["ts"]
        Rs, ts = self.params["Rs"], self.params["ts"]
        R_new = Rs_ @ Rs
        t_new = (Rs_ @ ts + ts_)[..., 0]
        return self(R_new, t_new)

class GaussianParams(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        motion_coefs: torch.Tensor | None = None,
        scene_center: torch.Tensor | None = None,
        scene_scale: torch.Tensor | float = 1.0,
    ):
        super().__init__()
        if not check_gaussian_sizes(
            means, quats, scales, colors, opacities, motion_coefs
        ):
            import ipdb

            ipdb.set_trace()
        params_dict = {
            "means": nn.Parameter(means),
            "quats": nn.Parameter(quats),
            "scales": nn.Parameter(scales),
            "colors": nn.Parameter(colors),
            "opacities": nn.Parameter(opacities),
        }
        if motion_coefs is not None:
            params_dict["motion_coefs"] = nn.Parameter(motion_coefs)
        self.params = nn.ParameterDict(params_dict)
        self.quat_activation = lambda x: F.normalize(x, dim=-1, p=2)
        self.color_activation = torch.sigmoid
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.motion_coef_activation = lambda x: F.softmax(x, dim=-1)

        if scene_center is None:
            scene_center = torch.zeros(3, device=means.device)
        self.register_buffer("scene_center", scene_center)
        self.register_buffer("scene_scale", torch.as_tensor(scene_scale))

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        req_keys = ["means", "quats", "scales", "colors", "opacities"]
        assert all(f"{prefix}{k}" in state_dict for k in req_keys)
        args = {
            "motion_coefs": None,
            "scene_center": torch.zeros(3),
            "scene_scale": torch.tensor(1.0),
        }
        for k in req_keys + list(args.keys()):
            if f"{prefix}{k}" in state_dict:
                args[k] = state_dict[f"{prefix}{k}"]
        return GaussianParams(**args)

    @property
    def num_gaussians(self) -> int:
        return self.params["means"].shape[0]

    def get_colors(self) -> torch.Tensor:
        return self.color_activation(self.params["colors"])

    def get_scales(self) -> torch.Tensor:
        return self.scale_activation(self.params["scales"])

    def get_opacities(self) -> torch.Tensor:
        return self.opacity_activation(self.params["opacities"])

    def get_quats(self) -> torch.Tensor:
        return self.quat_activation(self.params["quats"])

    def get_coefs(self) -> torch.Tensor:
        assert "motion_coefs" in self.params
        return self.motion_coef_activation(self.params["motion_coefs"])

    def densify_params(self, should_split, should_dup):
        """
        densify gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_dup = x[should_dup]
            x_split = x[should_split].repeat([2] + [1] * (x.ndim - 1))
            if name == "scales":
                x_split -= math.log(1.6)
            x_new = nn.Parameter(torch.cat([x[~should_split], x_dup, x_split], dim=0))
            updated_params[name] = x_new
            self.params[name] = x_new
        return updated_params

    def cull_params(self, should_cull):
        """
        cull gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_new = nn.Parameter(x[~should_cull])
            updated_params[name] = x_new
            self.params[name] = x_new
        return updated_params

    def reset_opacities(self, new_val):
        """
        reset all opacities to new_val
        """
        self.params["opacities"].data.fill_(new_val)
        updated_params = {"opacities": self.params["opacities"]}
        return updated_params


class MotionBases(nn.Module):
    def __init__(self, rots, transls):
        super().__init__()
        self.num_frames = rots.shape[1]
        self.num_bases = rots.shape[0]
        assert check_bases_sizes(rots, transls)
        self.params = nn.ParameterDict(
            {
                "rots": nn.Parameter(rots),
                "transls": nn.Parameter(transls),
            }
        )

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        param_keys = ["rots", "transls"]
        assert all(f"{prefix}{k}" in state_dict for k in param_keys)
        args = {k: state_dict[f"{prefix}{k}"] for k in param_keys}
        return MotionBases(**args)

    def compute_transforms(self, ts: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        """
        :param ts (B)
        :param coefs (G, K)
        returns transforms (G, B, 3, 4)
        """
        transls = self.params["transls"][:, ts]  # (K, B, 3)
        rots = self.params["rots"][:, ts]  # (K, B, 6)
        transls = torch.einsum("pk,kni->pni", coefs, transls)
        rots = torch.einsum("pk,kni->pni", coefs, rots)  # (G, B, 6)
        rotmats = cont_6d_to_rmat(rots)  # (K, B, 3, 3)
        return torch.cat([rotmats, transls[..., None]], dim=-1)


def check_gaussian_sizes(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    motion_coefs: torch.Tensor | None = None,
) -> bool:
    dims = means.shape[:-1]
    leading_dims_match = (
        quats.shape[:-1] == dims
        and scales.shape[:-1] == dims
        and colors.shape[:-1] == dims
        and opacities.shape == dims
    )
    if motion_coefs is not None and motion_coefs.numel() > 0:
        leading_dims_match &= motion_coefs.shape[:-1] == dims
    dims_correct = (
        means.shape[-1] == 3
        and (quats.shape[-1] == 4)
        and (scales.shape[-1] == 3)
        and (colors.shape[-1] == 3)
    )
    return leading_dims_match and dims_correct


def check_bases_sizes(motion_rots: torch.Tensor, motion_transls: torch.Tensor) -> bool:
    return (
        motion_rots.shape[-1] == 6
        and motion_transls.shape[-1] == 3
        and motion_rots.shape[:-2] == motion_transls.shape[:-2]
    )
