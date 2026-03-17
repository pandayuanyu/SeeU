import torch
import torch.nn.functional as F
from typing import Literal
import roma



def rt_to_mat4(
    R: torch.Tensor, t: torch.Tensor, s: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Args:
        R (torch.Tensor): (..., 3, 3).
        t (torch.Tensor): (..., 3).
        s (torch.Tensor): (...,).

    Returns:
        torch.Tensor: (..., 4, 4)
    """
    mat34 = torch.cat([R, t[..., None]], dim=-1)
    if s is None:
        bottom = (
            mat34.new_tensor([[0.0, 0.0, 0.0, 1.0]])
            .reshape((1,) * (mat34.dim() - 2) + (1, 4))
            .expand(mat34.shape[:-2] + (1, 4))
        )
    else:
        bottom = F.pad(1.0 / s[..., None, None], (3, 0), value=0.0)
    mat4 = torch.cat([mat34, bottom], dim=-2)
    return mat4



def rmat_to_cont_6d_fallback(matrix: torch.Tensor) -> torch.Tensor:
    """
    :param matrix (*, 3, 3)
    :returns 6d vector (*, 6)
    """

    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def cont_6d_to_rmat_safe(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [..., 6] -> R: [..., 3, 3]
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    
    b1 = F.normalize(a1, dim=-1, eps=eps)
    a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)

    R = torch.stack([b1, b2, b3], dim=-1)
    return R


def rot_loss_6d_cosine(pred6d: torch.Tensor, tgt6d: torch.Tensor) -> torch.Tensor:
    pred_n = F.normalize(pred6d, dim=-1, eps=1e-6)
    tgt_n  = F.normalize(tgt6d,  dim=-1, eps=1e-6)
    loss = 1.0 - (pred_n * tgt_n).sum(dim=-1)
    return torch.nan_to_num(loss, nan=0.0, posinf=2.0, neginf=0.0).mean()


def rot_loss_geodesic_safe(R_pred: torch.Tensor, R_tgt: torch.Tensor) -> torch.Tensor:
    R_rel = R_pred @ R_tgt.transpose(-1, -2)
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos_theta).mean()


def vee(S: torch.Tensor) -> torch.Tensor:
    return torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], dim=-1)


def so3_log_safe(R: torch.Tensor) -> torch.Tensor:
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    S = 0.5 * (R - R.transpose(-1, -2))
    v = vee(S)
    sin_theta = torch.sqrt((1.0 - cos_theta * cos_theta).clamp_min(1e-12))
    coeff = theta / (sin_theta + 1e-9)
    small = (theta < 1e-3)
    coeff = torch.where(small, 1.0 + (theta * theta) / 6.0, coeff)
    return coeff.unsqueeze(-1) * v


def solve_procrustes(
    src: torch.Tensor,
    dst: torch.Tensor,
    weights: torch.Tensor | None = None,
    enforce_se3: bool = False,
    rot_type: Literal["quat", "mat", "6d"] = "quat",
):
    """
    Solve the Procrustes problem to align two point clouds... (docstring omitted)
    """
    # Compute weights.
    if weights is None:
        weights = src.new_ones(src.shape[0])
    weights = weights[:, None] / weights.sum()
    # Normalize point positions.
    src_mean = (src * weights).sum(dim=0)
    dst_mean = (dst * weights).sum(dim=0)
    src_cent = src - src_mean
    dst_cent = dst - dst_mean
    # Normalize point scales.
    if not enforce_se3:
        src_scale = (src_cent**2 * weights).sum(dim=-1).mean().sqrt()
        dst_scale = (dst_cent**2 * weights).sum(dim=-1).mean().sqrt()
    else:
        src_scale = dst_scale = src.new_tensor(1.0)
    src_scaled = src_cent / src_scale
    dst_scaled = dst_cent / dst_scale
    # Compute the matrix for the singular value decomposition (SVD).
    matrix = (weights * dst_scaled).T @ src_scaled
    U, _, Vh = torch.linalg.svd(matrix)
    # Special reflection case.
    S = torch.eye(3, device=src.device)
    if torch.det(U) * torch.det(Vh) < 0:
        S[2, 2] = -1
    R = U @ S @ Vh
    # Compute the transformation.
    if rot_type == "quat":
        rot = roma.rotmat_to_unitquat(R).roll(1, dims=-1)
    elif rot_type == "6d":
        rot = rmat_to_cont_6d_fallback(R)
    else:
        rot = R
    s = dst_scale / src_scale
    t = dst_mean / s - src_mean @ R.T
    sim3 = rot, t, s
    # Debug: error.
    procrustes_dst = torch.einsim(
        "ij,nj->ni", rt_to_mat4(R, t, s), F.pad(src, (0, 1), value=1.0)
    )
    procrustes_dst = procrustes_dst[:, :3] / procrustes_dst[:, 3:]
    error_before = (torch.linalg.norm(dst - src, dim=-1) * weights[:, 0]).sum()
    error = (torch.linalg.norm(dst - procrustes_dst, dim=-1) * weights[:, 0]).sum()
    
    return sim3, (error.item(), error_before.item())