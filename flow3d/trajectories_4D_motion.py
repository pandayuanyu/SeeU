import numpy as np
import roma
import torch
import torch.nn.functional as F

from .transforms import rt_to_mat4 


def get_avg_w2c(w2cs: torch.Tensor):
    c2ws = torch.linalg.inv(w2cs)
    # 1. Compute the center
    center = c2ws[:, :3, -1].mean(0)
    # 2. Compute the z axis
    z = F.normalize(c2ws[:, :3, 2].mean(0), dim=-1)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = c2ws[:, :3, 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = F.normalize(torch.cross(y_, z, dim=-1), dim=-1)  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = torch.cross(z, x, dim=-1)  # (3)
    avg_c2w = rt_to_mat4(torch.stack([x, y, z], 1), center)
    avg_w2c = torch.linalg.inv(avg_c2w)
    return avg_w2c



def get_lookat(origins: torch.Tensor, viewdirs: torch.Tensor) -> torch.Tensor:
    """Triangulate a set of rays to find a single lookat point."""
    viewdirs = torch.nn.functional.normalize(viewdirs, dim=-1)
    eye = torch.eye(3, device=origins.device, dtype=origins.dtype)[None]
    I_min_cov = eye - (viewdirs[..., None] * viewdirs[..., None, :])
    sum_proj = I_min_cov.matmul(origins[..., None]).sum(dim=-3)
    lookat = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]
    assert not torch.any(torch.isnan(lookat))
    return lookat


@torch.no_grad()
def get_ref_fixed_w2cs(ref_w2c: torch.Tensor, num_frames: int, **_) -> torch.Tensor:
    """Camera pose is fixed at the reference frame."""
    return ref_w2c.unsqueeze(0).repeat(num_frames, 1, 1)

@torch.no_grad()
def get_ref_tilt_up_w2cs(ref_w2c: torch.Tensor, num_frames: int, degree: float, **_) -> torch.Tensor:
    """Camera tilts up/down (rotates around its X-axis) from the reference pose."""
    # Negative degree tilts up (convention from your original script)
    angles = torch.linspace(0, -torch.deg2rad(torch.tensor(degree)), num_frames, device=ref_w2c.device)
    rots_x = roma.rotvec_to_rotmat(torch.stack([angles, torch.zeros_like(angles), torch.zeros_like(angles)], dim=-1))
    ref_c2w = torch.linalg.inv(ref_w2c)
    new_c2ws = ref_c2w.clone().unsqueeze(0).repeat(num_frames, 1, 1)
    # Apply tilt rotation relative to the camera's local frame: R_new_world = R_ref_world @ R_tilt_local
    new_c2ws[:, :3, :3] = ref_c2w[:3, :3] @ rots_x
    return torch.linalg.inv(new_c2ws)

@torch.no_grad()
def get_ref_pan_right_w2cs(ref_w2c: torch.Tensor, num_frames: int, degree: float, **_) -> torch.Tensor:
    """Camera pans right/left (rotates around its Y-axis) from the reference pose."""
    angles = torch.linspace(0, torch.deg2rad(torch.tensor(degree)), num_frames, device=ref_w2c.device)
    rots_y = roma.rotvec_to_rotmat(torch.stack([torch.zeros_like(angles), angles, torch.zeros_like(angles)], dim=-1))
    ref_c2w = torch.linalg.inv(ref_w2c)
    new_c2ws = ref_c2w.clone().unsqueeze(0).repeat(num_frames, 1, 1)
    new_c2ws[:, :3, :3] = ref_c2w[:3, :3] @ rots_y
    return torch.linalg.inv(new_c2ws)

@torch.no_grad()
def get_ref_dolly_up_w2cs(ref_w2c: torch.Tensor, num_frames: int, distance: float, **_) -> torch.Tensor:
    translations = torch.linspace(0, distance, num_frames, device=ref_w2c.device)
    y_axis_world = ref_w2c[:3, 1]
    t_offsets = translations.unsqueeze(-1) * y_axis_world.unsqueeze(0)
    new_w2cs = ref_w2c.clone().unsqueeze(0).repeat(num_frames, 1, 1)
    new_w2cs[:, :3, 3] = ref_w2c[:3, 3] + t_offsets
    return new_w2cs

@torch.no_grad()
def get_ref_dolly_right_w2cs(ref_w2c: torch.Tensor, num_frames: int, distance: float, **_) -> torch.Tensor:
    """Camera moves right/left (translates along *camera's* X-axis) from the reference pose."""
    translations = torch.linspace(0, distance, num_frames, device=ref_w2c.device)
    
    ref_c2w = torch.linalg.inv(ref_w2c)
    
    x_axis_world = ref_c2w[:3, 0] 
    
    t_offsets = translations.unsqueeze(-1) * x_axis_world.unsqueeze(0)

    new_c2ws = ref_c2w.clone().unsqueeze(0).repeat(num_frames, 1, 1)
    new_c2ws[:, :3, 3] = ref_c2w[:3, 3] + t_offsets 
    
    return torch.linalg.inv(new_c2ws)

@torch.no_grad()
def get_ref_dolly_up_w2cs(ref_w2c: torch.Tensor, num_frames: int, distance: float, **_) -> torch.Tensor:
    """Camera moves up/down (translates along *camera's* Y-axis) from the reference pose."""
    translations = torch.linspace(0, distance, num_frames, device=ref_w2c.device)
    ref_c2w = torch.linalg.inv(ref_w2c)
    
    y_axis_world = ref_c2w[:3, 1]
    
    t_offsets = translations.unsqueeze(-1) * y_axis_world.unsqueeze(0)
    new_c2ws = ref_c2w.clone().unsqueeze(0).repeat(num_frames, 1, 1)
    new_c2ws[:, :3, 3] = ref_c2w[:3, 3] + t_offsets
    return torch.linalg.inv(new_c2ws)@torch.no_grad()

def get_ref_dolly_up_w2cs(ref_w2c: torch.Tensor, num_frames: int, distance: float, **_) -> torch.Tensor:
    """Camera moves up/down (translates along *camera's* Y-axis) from the reference pose."""
    translations = torch.linspace(0, distance, num_frames, device=ref_w2c.device)
    ref_c2w = torch.linalg.inv(ref_w2c)
    
    y_axis_world = ref_c2w[:3, 1]
    
    t_offsets = translations.unsqueeze(-1) * y_axis_world.unsqueeze(0)
    new_c2ws = ref_c2w.clone().unsqueeze(0).repeat(num_frames, 1, 1)
    new_c2ws[:, :3, 3] = ref_c2w[:3, 3] + t_offsets
    return torch.linalg.inv(new_c2ws)

@torch.no_grad()
def get_ref_dolly_out_w2cs(ref_w2c: torch.Tensor, num_frames: int, distance: float, **_) -> torch.Tensor:
    """Camera moves out/in (translates along *camera's* Z-axis) from the reference pose."""
    translations = torch.linspace(0, distance, num_frames, device=ref_w2c.device)
    ref_c2w = torch.linalg.inv(ref_w2c)

    z_axis_world = ref_c2w[:3, 2] 

    t_offsets = translations.unsqueeze(-1) * z_axis_world.unsqueeze(0)
    new_c2ws = ref_c2w.clone().unsqueeze(0).repeat(num_frames, 1, 1)
    new_c2ws[:, :3, 3] = ref_c2w[:3, 3] + t_offsets
    return torch.linalg.inv(new_c2ws)
