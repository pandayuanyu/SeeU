import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal, Optional

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import yaml
from loguru import logger as guru
from tqdm import tqdm

from flow3d.data import get_train_val_datasets, CustomDataConfig
from flow3d.renderer import Renderer
from flow3d.trajectories_4D_motion import get_avg_w2c, get_lookat
from flow3d.vis.utils import draw_keypoints_cv2, make_video_divisble


try:
    from flow3d.continuous_4D_motion import ContinuousMotionBases, ContinuousCameraPose
except ImportError:
    guru.error("ContinuousMotionBases or ContinuousCameraPose not found. This script requires the continuous motion model.")
    ContinuousMotionBases = None
    ContinuousCameraPose = None  

try:
    from training_4D_motion import normalize_time_idx_to_norm
except ImportError:
    guru.warning("normalize_time_idx_to_norm not found; using fallback implementation.")
    def normalize_time_idx_to_norm(t: torch.Tensor, T: int) -> torch.Tensor:
        if T <= 1:
            return torch.zeros_like(t, dtype=torch.float32)
        return (t.float() / (T - 1)) * 2.0 - 1.0

from flow3d.transforms_4D_motion import cont_6d_to_rmat_safe as cont_6d_to_rmat
import roma
import matplotlib.pyplot as plt  
import cv2  

torch.set_float32_matmul_precision("high")


# --- Time handling logic ---
def _get_extrapolated_ts_uniform_step(
    num_replay_frames: int,
    num_past_extrap_frames: int,
    num_future_extrap_frames: int,
    num_frames: int,
    device: torch.device,
):
    if num_frames <= 0:
        guru.error("num_frames must be positive.")
        return torch.tensor([], device=device)
    last_frame_idx = num_frames - 1
    duration = float(last_frame_idx) if last_frame_idx > 0 else 0.0
    if num_replay_frames <= 0 and (num_past_extrap_frames > 0 or num_future_extrap_frames > 0):
        time_step = 1.0
        guru.warning("Extrapolation with num_replay_frames=0, using dt=1.0.")
    elif num_replay_frames == 1:
        time_step = 1.0
        guru.warning("num_replay_frames=1, using dt=1.0 for extrapolation.")
    else:
        time_step = duration / max(num_replay_frames - 1, 1)
    t_start_orig = 0.0
    t_end_orig = duration
    t_start_render = t_start_orig - num_past_extrap_frames * time_step
    t_end_render = t_end_orig + num_future_extrap_frames * time_step
    total_render_frames = num_past_extrap_frames + num_replay_frames + num_future_extrap_frames
    if total_render_frames <= 0:
        guru.warning("Total render frames is 0.")
        return torch.tensor([], device=device)
    if total_render_frames == 1:
        if num_past_extrap_frames > 0:
            return torch.tensor([t_start_render], device=device)
        if num_future_extrap_frames > 0:
            return torch.tensor([t_end_render], device=device)
        return torch.tensor([(t_start_orig + t_end_orig) / 2.0], device=device)
    return torch.linspace(t_start_render, t_end_render, total_render_frames, device=device)


# --- HSV to RGB conversion function ---
def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    i = (h * 6.0).floor()
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i.long() % 6
    rgb = torch.stack(
        (
            torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v))))),
            torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p))))),
            torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q))))),
        ),
        dim=-1,
    )
    return rgb


# --- Spherical-coordinate color mapping ---
def _calculate_motion_colors_spherical(
    velocities_3d: torch.Tensor,
    min_value: float = 0.5,
    max_value: float = 0.9,
    min_saturation_vertical: float = 0.6,
    max_saturation_horizontal: float = 0.9,
    speed_power_factor: float = 0.8,
    hue_shift: float = 0.3,
    eps: float = 1e-6,
) -> torch.Tensor:
    N, T, _ = velocities_3d.shape
    device = velocities_3d.device
    speed = torch.linalg.norm(velocities_3d, dim=-1)
    directions = velocities_3d / (speed.unsqueeze(-1) + eps)
    min_speed = torch.min(speed)
    max_speed = torch.max(speed)
    speed_range = max_speed - min_speed
    if speed_range < eps:
        normalized_speed = torch.full_like(speed, 0.5)
        guru.warning("Spherical: Speed range is nearly zero, using 0.5 for normalized speed.")
    else:
        normalized_speed = (speed - min_speed) / (speed_range + eps)
    normalized_speed_powered = torch.pow(normalized_speed, speed_power_factor)
    azimuth = torch.atan2(directions[..., 1], directions[..., 0])

    hue = (azimuth / (2 * torch.pi) + 0.5 + hue_shift) % 1.0

    elevation_cos = torch.clamp(directions[..., 2], -1.0, 1.0)
    elevation = torch.acos(elevation_cos)
    saturation_base = torch.sin(elevation)
    saturation = min_saturation_vertical + (max_saturation_horizontal - min_saturation_vertical) * saturation_base
    saturation = torch.clamp(saturation, 0.0, 1.0)
    value = min_value + (max_value - min_value) * normalized_speed_powered
    value = torch.clamp(value, 0.0, 1.0)
    hsv = torch.stack([hue, saturation, value], dim=-1)
    colors_rgb = hsv_to_rgb(hsv.view(-1, 3)).view(N, T, 3)
    very_low_speed_threshold = min_speed + speed_range * 0.005
    very_low_speed_mask = speed < very_low_speed_threshold
    grey_color = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 1, 3)
    colors_rgb = torch.where(very_low_speed_mask.unsqueeze(-1), grey_color, colors_rgb)
    return colors_rgb


# --- Direct XYZ-to-RGB mapping ---
def _calculate_motion_colors_xyz(
    velocities_3d: torch.Tensor,
    min_brightness: float = 0.3,
    max_brightness: float = 1.0,
    direction_power_factor: float = 1.5,
    speed_power_factor: float = 1.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    N, T, _ = velocities_3d.shape
    device = velocities_3d.device
    speed = torch.linalg.norm(velocities_3d, dim=-1)
    directions = velocities_3d / (speed.unsqueeze(-1) + eps)
    min_speed = torch.min(speed)
    max_speed = torch.max(speed)
    speed_range = max_speed - min_speed
    if speed_range < eps:
        normalized_speed = torch.full_like(speed, 0.5)
        guru.warning("XYZ: Speed range is nearly zero, using 0.5 for normalized speed.")
    else:
        normalized_speed = (speed - min_speed) / (speed_range + eps)
    normalized_speed_powered = torch.pow(normalized_speed, speed_power_factor)

    # Remap (X, Z, Y) -> (R, G, B)
    remapped_directions = torch.stack(
        [
            directions[..., 2],  # X -> Red
            directions[..., 1],  # Z -> Green
            directions[..., 0],  # Y -> Blue
        ],
        dim=-1,
    )
    abs_directions = torch.abs(remapped_directions)
    rgb_direction = torch.pow(abs_directions, direction_power_factor)
    rgb_direction = rgb_direction / (torch.max(rgb_direction, dim=-1, keepdim=True).values + eps)

    brightness = min_brightness + (max_brightness - min_brightness) * normalized_speed_powered
    brightness = torch.clamp(brightness, 0.0, 1.0).unsqueeze(-1)

    colors_rgb = rgb_direction * brightness
    colors_rgb = torch.clamp(colors_rgb, 0.0, 1.0)
    very_low_speed_threshold = min_speed + speed_range * 0.005
    very_low_speed_mask = speed < very_low_speed_threshold
    grey_color = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 1, 3)
    colors_rgb = torch.where(very_low_speed_mask.unsqueeze(-1), grey_color, colors_rgb)
    return colors_rgb


# --- Drawing function ---
def draw_tracks_2d_static(
    img: torch.Tensor,  # [Note] img is the background; trajectories will be drawn on top of it
    tracks_2d: torch.Tensor,
    track_colors_rgb: Optional[torch.Tensor] = None,
    is_extrapolated: Optional[torch.Tensor] = None,
    track_line_width: int = 1,
    cmap_name: str = "gist_rainbow",
    dash_modulus: int = 3,
) -> np.ndarray:
    # [Modified] We copy img_np instead of modifying it directly
    img_np_base = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    if not isinstance(img_np_base, np.ndarray):
        img_np_base = np.array(img_np_base)
    
    img_np = img_np_base.copy()  # Draw on the copy

    H, W, _ = img_np.shape
    tracks_2d_np = tracks_2d.detach().cpu().numpy()

    if track_colors_rgb is not None:
        colors_np = (track_colors_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    else:
        cmap = plt.get_cmap(cmap_name)
        num_tracks = tracks_2d.shape[0]
        base_colors = [cmap(i / num_tracks)[:3] for i in range(num_tracks)]
        colors_np = np.zeros((num_tracks, tracks_2d.shape[1], 3), dtype=np.uint8)
        for i in range(num_tracks):
            colors_np[i, :, :] = (np.array(base_colors[i]) * 255).astype(np.uint8)

    is_extrapolated_np = None
    if is_extrapolated is not None:
        is_extrapolated_np = is_extrapolated.cpu().numpy()

    shift = 3
    factor = 1 << shift  # = 8

    for i in range(tracks_2d_np.shape[0]):
        track_points = tracks_2d_np[i]

        valid_mask = (
            np.isfinite(track_points).all(axis=1)
            & (track_points[:, 0] >= 0)
            & (track_points[:, 0] < W)
            & (track_points[:, 1] >= 0)
            & (track_points[:, 1] < H)
        )
        valid_points = track_points[valid_mask]

        if len(valid_points) < 2:
            continue

        valid_points_shifted = (valid_points * factor).astype(np.int32)
        valid_indices = np.where(valid_mask)[0]

        for j in range(len(valid_points) - 1):
            pt1 = (valid_points_shifted[j, 0], valid_points_shifted[j, 1])
            pt2 = (valid_points_shifted[j + 1, 0], valid_points_shifted[j + 1, 1])

            color_idx = valid_indices[j]

            should_draw = True
            if is_extrapolated_np is not None and color_idx < len(is_extrapolated_np):
                if is_extrapolated_np[color_idx]:
                    if j % dash_modulus != 0:
                        should_draw = False

            if not should_draw:
                continue

            if color_idx < colors_np.shape[1]:
                color = colors_np[i, color_idx].tolist()
                color_bgr = color[::-1]

                cv2.line(
                    img_np,
                    pt1,
                    pt2,
                    color_bgr,
                    track_line_width,
                    lineType=cv2.LINE_AA,
                    shift=shift,
                )
    return img_np


# --- View 2: draw camera trajectory ---
def _draw_camera_path_cv2(
    img_np: np.ndarray,  # [Note] Draw directly on this image
    path_2d_np: np.ndarray,
    H: int,
    W: int,
    color_bgr: tuple,
    is_extrapolated_np: Optional[np.ndarray] = None,
    dash_modulus: int = 3,
    line_width: int = 1,
    shift: int = 3,
):
    """Draw a 2D path on the image (supports dashed lines)."""
    factor = 1 << shift

    valid_mask = np.isfinite(path_2d_np).all(axis=1)

    valid_points = path_2d_np[valid_mask]
    if len(valid_points) < 2:
        return  # Not enough points to draw a line

    valid_points_shifted = (valid_points * factor).astype(np.int32)
    valid_indices = np.where(valid_mask)[0]  # [0, 1, 2, ...]

    for j in range(len(valid_points) - 1):
        pt1 = (valid_points_shifted[j, 0], valid_points_shifted[j, 1])
        pt2 = (valid_points_shifted[j + 1, 0], valid_points_shifted[j + 1, 1])

        color_idx = valid_indices[j]
        should_draw = True
        if is_extrapolated_np is not None and color_idx < len(is_extrapolated_np):
            if is_extrapolated_np[color_idx]:
                if j % dash_modulus != 0:
                    should_draw = False

        if should_draw:
            cv2.line(img_np, pt1, pt2, color_bgr, line_width, lineType=cv2.LINE_AA, shift=shift)


# --- View 2: draw camera frustum ---
def _draw_camera_frustum_cv2(
    img_np: np.ndarray,  # [Note] Draw directly on this image
    frustum_pts_2d_np: np.ndarray,  # (5, 2): [origin, tl, tr, br, bl]
    H: int,
    W: int,
    color_bgr: tuple,
    line_width: int = 1,
    shift: int = 3,
):
    """Draw a 2D frustum (5 points) on the image."""
    factor = 1 << shift

    if not np.isfinite(frustum_pts_2d_np).all():
        return

    pts_shifted = (frustum_pts_2d_np * factor).astype(np.int32)

    p0 = (pts_shifted[0, 0], pts_shifted[0, 1])  # Camera center
    p1 = (pts_shifted[1, 0], pts_shifted[1, 1])  # Top-Left
    p2 = (pts_shifted[2, 0], pts_shifted[2, 1])  # Top-Right
    p3 = (pts_shifted[3, 0], pts_shifted[3, 1])  # Bottom-Right
    p4 = (pts_shifted[4, 0], pts_shifted[4, 1])  # Bottom-Left

    cv2.line(img_np, p0, p1, color_bgr, line_width, lineType=cv2.LINE_AA, shift=shift)
    cv2.line(img_np, p0, p2, color_bgr, line_width, lineType=cv2.LINE_AA, shift=shift)
    cv2.line(img_np, p0, p3, color_bgr, line_width, lineType=cv2.LINE_AA, shift=shift)
    cv2.line(img_np, p0, p4, color_bgr, line_width, lineType=cv2.LINE_AA, shift=shift)
    cv2.line(img_np, p1, p2, color_bgr, line_width, lineType=cv2.LINE_AA, shift=shift)
    cv2.line(img_np, p2, p3, color_bgr, line_width, lineType=cv2.LINE_AA, shift=shift)
    cv2.line(img_np, p3, p4, color_bgr, line_width, lineType=cv2.LINE_AA, shift=shift)
    cv2.line(img_np, p4, p1, color_bgr, line_width, lineType=cv2.LINE_AA, shift=shift)


# --- Configurable offset for View 2 ---
@dataclass
class CameraView2Config:
    """Relative pose of the second fixed camera (third-person observer), relative to View 1."""
    pan_up_deg: float = 1.0  
    pan_right_deg: float = -0.3
    dolly_out: float = -10.0  # REVERSE
    dolly_right: float = 0.1
    dolly_up: float = 0.1  # REVERSE


@dataclass
class VideoConfig:
    work_dir: str
    data: CustomDataConfig = field(default_factory=CustomDataConfig)
    num_past_extrap_frames: int = 5
    num_replay_frames: int = 71
    num_future_extrap_frames: int = 5

    fps: float = 15.0
    port: int = 8890
    num_control_points: int = 8
    spline_degree: int = 3
    plot_extrapolation_frames: int = 100
    debug_print_single_gaussian: bool = False

    # View 2 config
    view2: CameraView2Config = field(default_factory=CameraView2Config)

    # Spherical and XYZ parameters
    sph_min_value: float = 0.1
    sph_max_value: float = 1.0
    sph_min_saturation_vertical: float = 0.2
    sph_max_saturation_horizontal: float = 1.0
    sph_speed_power_factor: float = 1.5

    xyz_min_brightness: float = 0.1
    xyz_max_brightness: float = 1.0
    xyz_direction_power_factor: float = 2.0
    xyz_speed_power_factor: float = 0.8

    sph_hue_shift: float = 0.6

    # Plotting parameters
    track_line_width: int = 1
    point_radius: int = 2
    dash_modulus: int = 3

    # Tracking point selection
    grid_selection_step: int = 80
    opacity_selection_threshold: float = 0.8

    # Frustum depth & line width
    frustum_depth: float = 0.7
    actor_cam_line_width: int = 2

    actor_cam_traj_line_width: int = 2
    actor_cam_frustum_line_width: int = 2
    
    track_alpha: float = 0.7  # 


def main(cfg: VideoConfig):
    # Load data
    train_dataset = get_train_val_datasets(cfg.data, load_val=False)[0]
    num_frames_trained = train_dataset.num_frames
    guru.info(f"Training dataset has {num_frames_trained} frames")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    assert os.path.exists(ckpt_path)
    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        False, # use_DDP
        work_dir=cfg.work_dir,
        port=None,
    )
    assert train_dataset.num_frames == renderer.num_frames
    guru.info(f"Rendering video from {renderer.global_step=}")

    # --- Load continuous motion model ---
    continuous_motion_model = None
    if ContinuousMotionBases is None:
        raise ImportError("Could not import ContinuousMotionBases")
    ckpt_m = f"{cfg.work_dir}/checkpoints/continuous_motion_bases.ckpt"
    if os.path.exists(ckpt_m):
        try:
            num_bases = renderer.model.motion_bases.num_bases
            continuous_motion_model = ContinuousMotionBases(
                num_bases=num_bases,
                num_frames=num_frames_trained,
                num_control_points=cfg.num_control_points,
                degree=cfg.spline_degree,
            ).to(device)
            continuous_motion_model.load_state_dict(
                torch.load(ckpt_m, map_location=device, weights_only=True)
            )
            continuous_motion_model.eval()
            guru.info(f"Loaded continuous motion model: {ckpt_m}")
        except Exception as e:
            guru.error(f"Failed loading motion model: {e}")
            continuous_motion_model = None
    else:
        guru.error(f"Motion checkpoint not found: {ckpt_m}")
    assert continuous_motion_model is not None, "Continuous motion model is required for smooth rendering."

    # --- Load continuous camera model ---
    continuous_camera_model = None
    if ContinuousCameraPose is None:
        raise ImportError("Could not import ContinuousCameraPose")

    ckpt_cam = f"{cfg.work_dir}/checkpoints/continuous_camera_pose.ckpt"
    if os.path.exists(ckpt_cam):
        try:
            continuous_camera_model = ContinuousCameraPose(
                num_frames=num_frames_trained,
                num_control_points=cfg.num_control_points,
                degree=cfg.spline_degree,
            ).to(device)
            continuous_camera_model.load_state_dict(
                torch.load(ckpt_cam, map_location=device, weights_only=True)
            )
            continuous_camera_model.eval()
            guru.info(f"Loaded continuous camera model: {ckpt_cam}")
        except Exception as e:
            guru.error(f"Failed loading camera model: {e}")
            continuous_camera_model = None
    else:
        guru.error(f"Camera motion checkpoint not found: {ckpt_cam}")
    assert continuous_camera_model is not None, "Continuous camera model is required for drawing the actor camera."

    train_w2cs = train_dataset.get_w2cs().to(device)
    avg_w2c = get_avg_w2c(train_w2cs)
    K = train_dataset.get_Ks()[0].to(device)
    img_wh = train_dataset.get_img_wh()
    W, H = img_wh

    # --- Time & camera setup ---
    ts = _get_extrapolated_ts_uniform_step(
        num_replay_frames=cfg.num_replay_frames,
        num_past_extrap_frames=cfg.num_past_extrap_frames,
        num_future_extrap_frames=cfg.num_future_extrap_frames,
        num_frames=num_frames_trained,
        device=device,
    )
    num_frames_render = len(ts)
    guru.info(f"Generated {num_frames_render} time steps for rendering (extrapolated).")

    ts_norm = normalize_time_idx_to_norm(ts, num_frames_trained)

    # View1: original average camera
    w2cs_view1 = avg_w2c[None].repeat(num_frames_render, 1, 1)

    # View2: new camera
    guru.info(f"Calculating new camera view (View 2) based on config: {cfg.view2}")
    avg_c2w = torch.linalg.inv(avg_w2c)

    # Translation
    T_trans_cam = torch.eye(4, device=device)
    T_trans_cam[0, 3] = cfg.view2.dolly_right
    T_trans_cam[1, 3] = cfg.view2.dolly_up
    T_trans_cam[2, 3] = cfg.view2.dolly_out

    # Rotation
    pan_up_rad = torch.deg2rad(torch.tensor(cfg.view2.pan_up_deg, device=device))
    cos_u, sin_u = torch.cos(pan_up_rad), torch.sin(pan_up_rad)
    R_pan_up = torch.eye(4, device=device)
    R_pan_up[1, 1] = cos_u
    R_pan_up[1, 2] = -sin_u
    R_pan_up[2, 1] = sin_u
    R_pan_up[2, 2] = cos_u

    pan_right_rad = torch.deg2rad(torch.tensor(cfg.view2.pan_right_deg, device=device))
    cos_r, sin_r = torch.cos(pan_right_rad), torch.sin(pan_right_rad)
    R_pan_right = torch.eye(4, device=device)
    R_pan_right[0, 0] = cos_r
    R_pan_right[0, 2] = sin_r
    R_pan_right[2, 0] = -sin_r
    R_pan_right[2, 2] = cos_r
    T_rot_cam = R_pan_up @ R_pan_right

    # Compose
    c2w_new = avg_c2w @ T_trans_cam @ T_rot_cam
    w2c_new = torch.linalg.inv(c2w_new)
    w2cs_view2 = w2c_new[None].repeat(num_frames_render, 1, 1)
    w2c_v2_static = w2cs_view2[0]
    P_v2 = K @ w2c_v2_static[:3]

    # Extrapolation mask
    last_frame_idx_trained = num_frames_trained - 1
    is_extrapolated = (ts < 0) | (ts > last_frame_idx_trained)
    is_extrapolated_np = is_extrapolated.cpu().numpy()
    guru.info(f"Created extrapolation mask. Total extrapolated frames: {torch.sum(is_extrapolated)}")

    grid = cfg.grid_selection_step
    opacity_threshold = cfg.opacity_selection_threshold

    # Point selection & color
    guru.info("Calculating smooth 3D tracks using continuous spline model...")
    fg = renderer.model.fg
    opacities = fg.get_opacities()
    mask = (opacities > opacity_threshold)
    inds_to_track = torch.where(mask)[0][::grid]
    num_tracks = len(inds_to_track)
    tracks_3d = None
    tracks_colors_spherical_rgb = None
    tracks_colors_xyz_rgb = None

    tracks_2d_static_view1 = torch.empty(0, num_frames_render, 2, device=device)
    tracks_2d_static_view2 = torch.empty(0, num_frames_render, 2, device=device)

    if num_tracks == 0:
        guru.warning("No tracks selected. The video will not contain tracks.")
    else:
        guru.info(f"Selected {num_tracks} Gaussians to track.")
        coefs_track = fg.get_coefs()[inds_to_track]
        means_track = fg.params["means"][inds_to_track]

        with torch.no_grad():
            bases = continuous_motion_model.forward_extrap(ts_norm.unsqueeze(-1))
            rots_bases = bases[..., :6]
            trans_bases = bases[..., 6:]
            rots_track = torch.einsum("nk,ktd->ntd", coefs_track, rots_bases)
            trans_track = torch.einsum("nk,ktd->ntd", coefs_track, trans_bases)
            N, T_render = rots_track.shape[:2]
            rotmats_track = cont_6d_to_rmat(rots_track.view(-1, 6)).view(N, T_render, 3, 3)
            tracks_3d = torch.einsum("ntij,nj->nti", rotmats_track, means_track) + trans_track
        guru.info(f"Calculated smooth 3D tracks, shape: {tracks_3d.shape}")

        velocities_3d = torch.cat(
            [tracks_3d[:, 1:, :] - tracks_3d[:, :-1, :], torch.zeros_like(tracks_3d[:, -1:, :])],
            dim=1,
        )

        guru.info("Calculating spherical colors (with hue shift and brighter/lighter tones)...")
        tracks_colors_spherical_rgb = _calculate_motion_colors_spherical(
            velocities_3d,
            min_value=cfg.sph_min_value,
            max_value=cfg.sph_max_value,
            min_saturation_vertical=cfg.sph_min_saturation_vertical,
            max_saturation_horizontal=cfg.sph_max_saturation_horizontal,
            speed_power_factor=cfg.sph_speed_power_factor,
            hue_shift=cfg.sph_hue_shift,
        )

        guru.info("Calculating XYZ colors (with remapped channels and brighter/lighter tones)...")
        tracks_colors_xyz_rgb = _calculate_motion_colors_xyz(
            velocities_3d,
            min_brightness=cfg.xyz_min_brightness,
            max_brightness=cfg.xyz_max_brightness,
            direction_power_factor=cfg.xyz_direction_power_factor,
            speed_power_factor=cfg.xyz_speed_power_factor,
        )

        guru.info("Projecting tracks for View 1...")
        tracks_2d_view1 = torch.einsum(
            "ij,bjk,nbk->nbi", K, w2cs_view1[:, :3], F.pad(tracks_3d, (0, 1), value=1.0)
        )
        tracks_2d_view1 = tracks_2d_view1[..., :2] / (tracks_2d_view1[..., 2:] + 1e-6)
        tracks_2d_static_view1 = tracks_2d_view1
        guru.info(f"View 1 tracks shape: {tracks_2d_static_view1.shape=}")

        guru.info("Projecting tracks for View 2...")
        tracks_2d_view2 = torch.einsum(
            "ij,bjk,nbk->nbi", K, w2cs_view2[:, :3], F.pad(tracks_3d, (0, 1), value=1.0)
        )
        tracks_2d_view2 = tracks_2d_view2[..., :2] / (tracks_2d_view2[..., 2:] + 1e-6)
        tracks_2d_static_view2 = tracks_2d_view2
        guru.info(f"View 2 tracks shape: {tracks_2d_static_view2.shape=}")

    # --- Compute the "actor camera" trajectory & frustum ---
    guru.info("Calculating dynamic 'actor' camera trajectory using continuous_camera_model...")

    with torch.no_grad():
        bases_cam = continuous_camera_model.forward_extrap(ts_norm.unsqueeze(-1))  # [T, 1, 9]
        pose_9d_cam = bases_cam.squeeze(1)  # [T, 9]

        rots_6d_cam = pose_9d_cam[..., :6]
        trans_3d_cam = pose_9d_cam[..., 6:]
        rotmats_cam = cont_6d_to_rmat(rots_6d_cam.view(-1, 6)).view(num_frames_render, 3, 3)

        w2cs_actor_cam = torch.eye(4, device=device)[None].repeat(num_frames_render, 1, 1)
        w2cs_actor_cam[:, :3, :3] = rotmats_cam
        w2cs_actor_cam[:, :3, 3] = trans_3d_cam

    c2ws_actor_cam = torch.linalg.inv(w2cs_actor_cam)
    traj_3d_actor = c2ws_actor_cam[:, :3, 3]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = cfg.frustum_depth
    p0_cam = torch.tensor([0.0, 0.0, 0.0], device=device)
    p1_cam = torch.tensor([(0 - cx) / fx * d, (0 - cy) / fy * d, d], device=device)      # TL
    p2_cam = torch.tensor([(W - cx) / fx * d, (0 - cy) / fy * d, d], device=device)      # TR
    p3_cam = torch.tensor([(W - cx) / fx * d, (H - cy) / fy * d, d], device=device)      # BR
    p4_cam = torch.tensor([(0 - cx) / fx * d, (H - cy) / fy * d, d], device=device)      # BL
    frustum_pts_cam = torch.stack([p0_cam, p1_cam, p2_cam, p3_cam, p4_cam], dim=0)
    frustum_pts_cam_hom = F.pad(frustum_pts_cam, (0, 1), value=1.0)

    frustums_world_actor = torch.einsum("tij, kj -> tki", c2ws_actor_cam, frustum_pts_cam_hom)
    frustums_world_actor = frustums_world_actor[..., :3]

    guru.info("Projecting 'actor' camera trajectory and frustum into View 2...")

    traj_3d_actor_hom = F.pad(traj_3d_actor, (0, 1), value=1.0)
    traj_2d_v2_hom = (P_v2 @ traj_3d_actor_hom.T).T
    traj_2d_v2_actor = (traj_2d_v2_hom[..., :2] / (traj_2d_v2_hom[..., 2:] + 1e-6))
    traj_2d_v2_actor_np = traj_2d_v2_actor.cpu().numpy()

    T_frus, N_frus, _ = frustums_world_actor.shape
    frustums_world_actor_hom = F.pad(frustums_world_actor.reshape(-1, 3), (0, 1), value=1.0)
    frustums_2d_v2_hom = (P_v2 @ frustums_world_actor_hom.T).T
    frustums_2d_v2_actor = (frustums_2d_v2_hom[..., :2] / (frustums_2d_v2_hom[..., 2:] + 1e-6))
    frustums_2d_v2_actor = frustums_2d_v2_actor.view(T_frus, N_frus, 2)
    frustums_2d_v2_actor_np = frustums_2d_v2_actor.cpu().numpy()
    guru.info("Projection of 'actor' camera completed.")

    # --- Debug print ---
    print("\n" + "=" * 50)
    print("DEBUGGING INFORMATION (Using Spline Model)")
    print("=" * 50)
    print(f"\n## Camera Extrinsics (w2c) (View 1 - Original):")
    print(w2cs_view1[0].cpu().numpy())
    print(f"\n## Camera Extrinsics (w2c) (View 2 - New):")
    print(w2cs_view2[0].cpu().numpy())
    print(f"\n## Camera Extrinsics (w2c) (Actor Cam @ t=0 - From Model):")
    print(c2ws_actor_cam[0].cpu().numpy())  # Printing c2w here is also fine

    # --- Video buffers ---
    video_spherical_view1 = []
    video_xyz_view1 = []
    video_spherical_view2 = []
    video_xyz_view2 = []

    # **New**: clean video buffers without tracks
    video_clean_view1 = []
    video_clean_view2 = []

    fg = renderer.model.fg
    all_fg_coefs = fg.get_coefs()
    all_canonical_means = fg.params["means"]
    all_canonical_quats = fg.get_quats()

    if renderer.model.has_bg:
        all_means_bg = renderer.model.bg.params["means"]
        all_quats_bg = renderer.model.bg.get_quats()

    POINT_COLOR_RGB = (255, 255, 0)  # Bright yellow (for CV2, BGR format)
    POINT_COLOR_BGR = (0, 255, 255)  # Bright yellow (BGR)
    CAM_TRAJ_COLOR_BGR = (0, 0, 255)  # Red (BGR)
    CAM_FRUSTUM_COLOR_BGR = (0, 80, 160) # Dark orange (BGR)

    shift = 3
    factor = 1 << shift  # = 8
    point_radius_shifted = cfg.point_radius * factor

    for frame_idx, (w2c_v1, w2c_v2, t, t_norm_single) in enumerate(
        tqdm(zip(w2cs_view1, w2cs_view2, ts, ts_norm), total=num_frames_render, desc="Rendering videos (2 views) with tracks")
    ):
        # 1) Compute geometry for the current frame
        with torch.no_grad():
            bases_main = continuous_motion_model.forward_extrap(t_norm_single.view(1, 1))
            bases_main = bases_main[:, 0, :]
            rots_main = bases_main[:, :6]
            trans_main = bases_main[:, 6:]
            rots_final = torch.einsum("gk,kd->gd", all_fg_coefs, rots_main)
            trans_final = torch.einsum("gk,kd->gd", all_fg_coefs, trans_main)
            rotmats_final = cont_6d_to_rmat(rots_final)
            means_fg_main = (torch.bmm(rotmats_final, all_canonical_means.unsqueeze(-1)).squeeze(-1) + trans_final)
            quats_transform_xyzw = roma.rotmat_to_unitquat(rotmats_final)
            quats_final_xyzw = roma.quat_product(quats_transform_xyzw, roma.quat_wxyz_to_xyzw(all_canonical_quats))
            quats_fg_main = F.normalize(roma.quat_xyzw_to_wxyz(quats_final_xyzw), p=2, dim=-1)
            if renderer.model.has_bg:
                render_means_main = torch.cat([means_fg_main, all_means_bg], dim=0)
                render_quats_main = torch.cat([quats_fg_main, all_quats_bg], dim=0)
            else:
                render_means_main = means_fg_main
                render_quats_main = quats_fg_main

        # 2) Render the "clean images" for both views (before drawing any tracks/overlays)
        with torch.inference_mode():
            img_torch_v1 = renderer.model.render(
                t=None, w2cs=w2c_v1[None], Ks=K[None], img_wh=img_wh,
                means=render_means_main, quats=render_quats_main,
            )["img"][0]

            img_torch_v2 = renderer.model.render(
                t=None, w2cs=w2c_v2[None], Ks=K[None], img_wh=img_wh,
                means=render_means_main, quats=render_quats_main,
            )["img"][0]

        # Save clean frames before any drawing
        clean_v1 = (img_torch_v1.detach().cpu().numpy() * 255).astype(np.uint8)
        clean_v2 = (img_torch_v2.detach().cpu().numpy() * 255).astype(np.uint8)
        video_clean_view1.append(clean_v1.copy())
        video_clean_view2.append(clean_v2.copy())

        # Get the clean background for the current frame
        bg_v1 = clean_v1.copy() 
        bg_v2 = clean_v2.copy()
        
        # Define alpha blending weights
        alpha = cfg.track_alpha
        beta = 1.0 - cfg.track_alpha
    

        # If there are no tracks, directly draw the opaque camera (if any)
        if num_tracks == 0:
            frame_v1_final = bg_v1 # V1 stays clean
            frame_v2_final = bg_v2 # V2 will be drawn on

            H_img_v2, W_img_v2, _ = frame_v2_final.shape
            current_frustum_pts_np = frustums_2d_v2_actor_np[frame_idx]
            current_cam_pos_2d = traj_2d_v2_actor_np[frame_idx]

            # Draw directly on frame_v2_final (opaque)
            _draw_camera_path_cv2(
                frame_v2_final, traj_2d_v2_actor_np, H_img_v2, W_img_v2,
                CAM_TRAJ_COLOR_BGR, is_extrapolated_np, cfg.dash_modulus, cfg.actor_cam_traj_line_width, shift
            )
            _draw_camera_frustum_cv2(
                frame_v2_final, current_frustum_pts_np, H_img_v2, W_img_v2,
                CAM_FRUSTUM_COLOR_BGR, cfg.actor_cam_frustum_line_width, shift
            )
            if np.isfinite(current_cam_pos_2d).all():
                cam_pos_shifted = (current_cam_pos_2d * factor).astype(np.int32)
                cv2.circle(
                    frame_v2_final,
                    (cam_pos_shifted[0], cam_pos_shifted[1]),
                    point_radius_shifted,
                    POINT_COLOR_BGR, # Use BGR color
                    -1,
                    lineType=cv2.LINE_AA,
                    shift=shift,
                )

            video_spherical_view1.append(frame_v1_final)
            video_xyz_view1.append(frame_v1_final)
            video_spherical_view2.append(frame_v2_final)
            video_xyz_view2.append(frame_v2_final)
            continue

        # 3) With tracks: prepare 2D positions
        H_img, W_img = img_torch_v1.shape[:2]

        curr_pts_v1 = tracks_2d_static_view1[:, frame_idx]
        curr_pts_np_v1 = curr_pts_v1.detach().cpu().numpy()
        valid_mask_v1 = (
            np.isfinite(curr_pts_np_v1).all(axis=1)
            & (curr_pts_np_v1[:, 0] >= 0)
            & (curr_pts_np_v1[:, 0] < W_img)
            & (curr_pts_np_v1[:, 1] >= 0)
            & (curr_pts_np_v1[:, 1] < H_img)
        )
        curr_pts_np_valid_v1 = curr_pts_np_v1[valid_mask_v1]
        curr_pts_shifted_v1 = (curr_pts_np_valid_v1 * factor).astype(np.int32)

        curr_pts_v2 = tracks_2d_static_view2[:, frame_idx]
        curr_pts_np_v2 = curr_pts_v2.detach().cpu().numpy()
        valid_mask_v2 = (
            np.isfinite(curr_pts_np_v2).all(axis=1)
            & (curr_pts_np_v2[:, 0] >= 0)
            & (curr_pts_np_v2[:, 0] < W_img)
            & (curr_pts_np_v2[:, 1] >= 0)
            & (curr_pts_np_v2[:, 1] < H_img)
        )
        curr_pts_np_valid_v2 = curr_pts_np_v2[valid_mask_v2]
        curr_pts_shifted_v2 = (curr_pts_np_valid_v2 * factor).astype(np.int32)

        current_cam_pos_2d = traj_2d_v2_actor_np[frame_idx]
        cam_pos_shifted = None
        if np.isfinite(current_cam_pos_2d).all():
            cam_pos_shifted = (current_cam_pos_2d * factor).astype(np.int32)
        
        current_frustum_pts_np = frustums_2d_v2_actor_np[frame_idx]

        # 4) Spherical colors (both views)
        
        # --- V1 Spherical ---
        # 4a. Draw trajectories onto a *copy of the background*
        overlay_sph_v1_tracks = draw_tracks_2d_static(
            img_torch_v1, # Draw on the clean background (torch)
            tracks_2d_static_view1,
            track_colors_rgb=tracks_colors_spherical_rgb,
            is_extrapolated=is_extrapolated,
            track_line_width=cfg.track_line_width,
            dash_modulus=cfg.dash_modulus,
        )
        # 4b. Blend (background with tracks) and (clean background)
        frame_sph_v1_blended = cv2.addWeighted(overlay_sph_v1_tracks, alpha, bg_v1, beta, 0)
        
        # 4c. Draw opaque points
        for pt_shifted in curr_pts_shifted_v1:
            cv2.circle(
                frame_sph_v1_blended, # Draw on the blended image
                (pt_shifted[0], pt_shifted[1]),
                point_radius_shifted,
                POINT_COLOR_BGR, # Use BGR color
                -1,
                lineType=cv2.LINE_AA,
                shift=shift,
            )
        video_spherical_view1.append(frame_sph_v1_blended)

        # --- V2 Spherical ---
        # 4d. Draw trajectories onto a *copy of the background*
        overlay_sph_v2_tracks = draw_tracks_2d_static(
            img_torch_v2, # Draw on the clean background (torch)
            tracks_2d_static_view2,
            track_colors_rgb=tracks_colors_spherical_rgb,
            is_extrapolated=is_extrapolated,
            track_line_width=cfg.track_line_width,
            dash_modulus=cfg.dash_modulus,
        )
        # 4e. Blend (background with tracks) and (clean background)
        frame_sph_v2_blended = cv2.addWeighted(overlay_sph_v2_tracks, alpha, bg_v2, beta, 0)
        
        # 4f. Draw opaque points
        for pt_shifted in curr_pts_shifted_v2:
            cv2.circle(
                frame_sph_v2_blended, # Draw on the blended image
                (pt_shifted[0], pt_shifted[1]),
                point_radius_shifted,
                POINT_COLOR_BGR, # Use BGR color
                -1,
                lineType=cv2.LINE_AA,
                shift=shift,
            )
        
        # 4g. Draw opaque camera
        _draw_camera_path_cv2(
            frame_sph_v2_blended, # Draw on the blended image
            traj_2d_v2_actor_np,
            H_img,
            W_img,
            CAM_TRAJ_COLOR_BGR,
            is_extrapolated_np,
            cfg.dash_modulus,
            cfg.actor_cam_traj_line_width,
            shift,
        )
        _draw_camera_frustum_cv2(
            frame_sph_v2_blended, # Draw on the blended image
            current_frustum_pts_np,
            H_img,
            W_img,
            CAM_FRUSTUM_COLOR_BGR,
            cfg.actor_cam_frustum_line_width,
            shift,
        )
        if cam_pos_shifted is not None:
            cv2.circle(
                frame_sph_v2_blended, # Draw on the blended image
                (cam_pos_shifted[0], cam_pos_shifted[1]),
                point_radius_shifted,
                POINT_COLOR_BGR, # Use BGR color
                -1,
                lineType=cv2.LINE_AA,
                shift=shift,
            )
        video_spherical_view2.append(frame_sph_v2_blended)

        # 5) XYZ colors (both views)
        
        # --- V1 XYZ ---
        # 5a. Draw trajectories onto a *copy of the background*
        overlay_xyz_v1_tracks = draw_tracks_2d_static(
            img_torch_v1, # Draw on the clean background (torch)
            tracks_2d_static_view1,
            track_colors_rgb=tracks_colors_xyz_rgb,
            is_extrapolated=is_extrapolated,
            track_line_width=cfg.track_line_width,
            dash_modulus=cfg.dash_modulus,
        )
        # 5b. Blend (background with tracks) and (clean background)
        frame_xyz_v1_blended = cv2.addWeighted(overlay_xyz_v1_tracks, alpha, bg_v1, beta, 0)
        
        # 5c. Draw opaque points
        for pt_shifted in curr_pts_shifted_v1:
            cv2.circle(
                frame_xyz_v1_blended, # Draw on the blended image
                (pt_shifted[0], pt_shifted[1]),
                point_radius_shifted,
                POINT_COLOR_BGR, # Use BGR color
                -1,
                lineType=cv2.LINE_AA,
                shift=shift,
            )
        video_xyz_view1.append(frame_xyz_v1_blended)
        
        # --- V2 XYZ ---
        # 5d. Draw trajectories onto a *copy of the background*
        overlay_xyz_v2_tracks = draw_tracks_2d_static(
            img_torch_v2, # Draw on the clean background (torch)
            tracks_2d_static_view2,
            track_colors_rgb=tracks_colors_xyz_rgb,
            is_extrapolated=is_extrapolated,
            track_line_width=cfg.track_line_width,
            dash_modulus=cfg.dash_modulus,
        )
        # 5e. Blend (background with tracks) and (clean background)
        frame_xyz_v2_blended = cv2.addWeighted(overlay_xyz_v2_tracks, alpha, bg_v2, beta, 0)
        
        # 5f. Draw opaque points
        for pt_shifted in curr_pts_shifted_v2:
            cv2.circle(
                frame_xyz_v2_blended, # Draw on the blended image
                (pt_shifted[0], pt_shifted[1]),
                point_radius_shifted,
                POINT_COLOR_BGR, # Use BGR color
                -1,
                lineType=cv2.LINE_AA,
                shift=shift,
            )

        # 5g. Draw opaque camera
        _draw_camera_path_cv2(
            frame_xyz_v2_blended, # Draw on the blended image
            traj_2d_v2_actor_np,
            H_img,
            W_img,
            CAM_TRAJ_COLOR_BGR,
            is_extrapolated_np,
            cfg.dash_modulus,
            cfg.actor_cam_traj_line_width,
            shift,
        )
        _draw_camera_frustum_cv2(
            frame_xyz_v2_blended, # Draw on the blended image
            current_frustum_pts_np,
            H_img,
            W_img,
            CAM_FRUSTUM_COLOR_BGR,
            cfg.actor_cam_frustum_line_width,
            shift,
        )
        if cam_pos_shifted is not None:
            cv2.circle(
                frame_xyz_v2_blended, # Draw on the blended image
                (cam_pos_shifted[0], cam_pos_shifted[1]),
                point_radius_shifted,
                POINT_COLOR_BGR, # Use BGR color
                -1,
                lineType=cv2.LINE_AA,
                shift=shift,
            )
        video_xyz_view2.append(frame_xyz_v2_blended)


    # --- Save all 6 videos ---
    video_spherical_view1 = np.stack(video_spherical_view1, axis=0)
    video_xyz_view1 = np.stack(video_xyz_view1, axis=0)
    video_spherical_view2 = np.stack(video_spherical_view2, axis=0)
    video_xyz_view2 = np.stack(video_xyz_view2, axis=0)

    # Clean videos
    video_clean_view1 = np.stack(video_clean_view1, axis=0)
    video_clean_view2 = np.stack(video_clean_view2, axis=0)

    video_dir = f"{cfg.work_dir}/videos/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(video_dir, exist_ok=True)

    guru.info("Saving spherical color video (View 1)...")
    iio.imwrite(f"{video_dir}/video_track_spherical_view1.mp4", make_video_divisble(video_spherical_view1), fps=cfg.fps, quality=8)

    guru.info("Saving XYZ color video (View 1)...")
    iio.imwrite(f"{video_dir}/video_track_xyz_view1.mp4", make_video_divisble(video_xyz_view1), fps=cfg.fps, quality=8)

    guru.info("Saving spherical color video (View 2 - with Actor Cam)...")
    iio.imwrite(f"{video_dir}/video_track_spherical_view2_actor_cam.mp4", make_video_divisble(video_spherical_view2), fps=cfg.fps, quality=8)

    guru.info("Saving XYZ color video (View 2 - with Actor Cam)...")
    iio.imwrite(f"{video_dir}/video_track_xyz_view2_actor_cam.mp4", make_video_divisble(video_xyz_view2), fps=cfg.fps, quality=8)

    # **Save clean versions without tracks**
    guru.info("Saving clean (no tracks) video (View 1)...")
    iio.imwrite(f"{video_dir}/video_clean_view1.mp4", make_video_divisble(video_clean_view1), fps=cfg.fps, quality=8)

    guru.info("Saving clean (no tracks) video (View 2)...")
    iio.imwrite(f"{video_dir}/video_clean_view2.mp4", make_video_divisble(video_clean_view2), fps=cfg.fps, quality=8)

    guru.info(f"Saving render config to {video_dir}/render_cfg.yaml")
    try:
        from tyro.extras import dump_yaml
        with open(f"{video_dir}/render_cfg.yaml", "w") as f:
            dump_yaml(cfg, f)
    except ImportError:
        with open(f"{video_dir}/render_cfg.yaml", "w") as f:
            yaml.dump(asdict(cfg), f, default_flow_style=None, sort_keys=False)


if __name__ == "__main__":
    main(tyro.cli(VideoConfig))