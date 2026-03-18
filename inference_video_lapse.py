import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal
import re

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import yaml
from loguru import logger as guru
from tqdm import tqdm
from viser import ViserServer 

# --- Matplotlib (headless safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# project imports
from flow3d.data import get_train_val_datasets, CustomDataConfig
from flow3d.renderer import Renderer
from flow3d.trajectories_4D_motion import (
    get_avg_w2c,
    get_lookat,
    get_ref_fixed_w2cs,
    get_ref_tilt_up_w2cs,
    get_ref_pan_right_w2cs,
    get_ref_dolly_up_w2cs,
    get_ref_dolly_right_w2cs,
    get_ref_dolly_out_w2cs,
)
from flow3d.vis.utils import make_video_divisble, get_server
from flow3d.transforms_4D_motion import (
    cont_6d_to_rmat_safe as cont_6d_to_rmat,
    rt_to_mat4,
    rmat_to_cont_6d_fallback as rmat_to_cont_6d,
)

import roma
import viser.transforms as vt
from scipy.ndimage import sobel, binary_dilation

torch.set_float32_matmul_precision("high")


# =============================================================================
# Time handling
# =============================================================================

def _get_extrapolated_ts_uniform_step(
    num_replay_frames: int,
    num_past_extrap_frames: int,
    num_future_extrap_frames: int,
    num_frames: int,
    device: torch.device,
):
    """
    Build a continuous timeline `ts` in frame-index units, with optional
    extrapolation before frame 0 and after frame (num_frames-1).
    """
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


# Normalize time to [-1,1] for the spline models
try:
    from training_4D_motion import normalize_time_idx_to_norm
except ImportError:
    guru.warning("normalize_time_idx_to_norm not found; using fallback.")
    def normalize_time_idx_to_norm(t: torch.Tensor, T: int) -> torch.Tensor:
        if T <= 1:
            return torch.zeros_like(t, dtype=torch.float32)
        return (t.float() / (T - 1)) * 2.0 - 1.0


# =============================================================================
# Camera config / CLI interface
# =============================================================================

@dataclass
class CameraConfig:
    mode: Literal["continuous", "reference"] = "continuous"
    ref: int = 0
    traj: Literal["fixed", "tilt-up", "pan-right", "dolly-up", "dolly-right", "dolly-out"] = "fixed"
    degree: float = 15.0
    distance: float = -1.0


# =============================================================================
# Video config
# =============================================================================

@dataclass
class VideoConfig:
    work_dir: str
    data: CustomDataConfig = field(default_factory=CustomDataConfig)

    # Timeline / extrapolation
    num_past_extrap_frames: int = 5
    num_replay_frames: int = 71
    num_future_extrap_frames: int = 5

    fps: float = 15.0
    port: int = 8890

    num_control_points: int = 8
    spline_degree: int = 3

    plot_extrapolation_frames: int = 2
    debug_print_single_gaussian: bool = False

    camera: CameraConfig = field(default_factory=CameraConfig)
    gt: "GTConfig" = field(default_factory=lambda: GTConfig())


# =============================================================================
# GT Config
# =============================================================================
@dataclass
class GTConfig:
    gt_dir: str | None = None  # Path to GT folder


# =============================================================================
# GT Helper Functions
# =============================================================================

def parse_frame_index(filename: str) -> int | None:
    match = re.search(r'(\d+)\.\w+$', filename)
    if match:
        return int(match.group(1))
    return None


def load_gt_info(gt_dir: str):
    guru.info(f"Loading GT info from: {gt_dir}")
    txt_path = os.path.join(gt_dir, "sample_frame.txt")
    if not os.path.exists(txt_path):
        guru.error(f"GT Error: sample_frame.txt not found in {gt_dir}")
        return None

    gt_filename_map = {}
    available_gt_indices = []
    try:
        all_files = os.listdir(gt_dir)
        for f in all_files:
            idx = parse_frame_index(f)
            if idx is not None:
                gt_filename_map[idx] = f
        if not gt_filename_map:
            guru.error(f"GT Error: No image files found in {gt_dir}")
            return None
        available_gt_indices = np.array(sorted(gt_filename_map.keys()))
        guru.info(f"Found {len(gt_filename_map)} total GT frames.")
    except Exception as e:
        guru.error(f"GT Error loading image files: {e}")
        return None

    try:
        with open(txt_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        total_frames_line = [l for l in lines if l.startswith("total_frames=")]
        sampled_names_line_idx = lines.index("sampled_names:")
        T_gt = int(total_frames_line[0].split('=')[1])
        sampled_names = [l for l in lines[sampled_names_line_idx + 1:] if l]
        if not sampled_names:
            guru.error("GT Error: No sampled_names found in sample_frame.txt")
            return None
        first_gt_idx = parse_frame_index(sampled_names[0])
        last_gt_idx = parse_frame_index(sampled_names[-1])
        if first_gt_idx is None or last_gt_idx is None:
            guru.error("GT Error: Could not parse frame indices in sample_frame.txt")
            return None
        guru.info(f"GT Info: T_gt={T_gt}, first_idx={first_gt_idx}, last_idx={last_gt_idx}")
        return first_gt_idx, last_gt_idx, T_gt, gt_filename_map, available_gt_indices
    except Exception as e:
        guru.error(f"GT Error parsing sample_frame.txt: {e}")
        return None


# =============================================================================
# Debug helper (optional)
# =============================================================================

def print_single_gaussian_trajectory(renderer: "Renderer", num_frames_to_print: int = 30):
    actual_frames = min(num_frames_to_print, renderer.num_frames)
    if actual_frames <= 0:
        return

    guru.info("=" * 20 + f" Trajectory of Gaussian #0 (First {actual_frames} Frames) " + "=" * 20)

    fg_params = renderer.model.fg
    gaussian_index = 0
    if gaussian_index >= fg_params.num_gaussians:
        guru.warning(
            f"Gaussian index {gaussian_index} out of bounds ({fg_params.num_gaussians}). Skipping trajectory."
        )
        return

    canonical_mean = fg_params.params["means"][gaussian_index]
    canonical_quat = fg_params.get_quats()[gaussian_index]  # [4] WXYZ

    ts_to_compute = torch.arange(actual_frames, device=renderer.device)
    inds_to_compute = torch.tensor([gaussian_index], device=renderer.device)

    with torch.no_grad():
        transfms = renderer.model.compute_transforms(ts_to_compute, inds_to_compute)  # [1,T,4,4]
        padded_mean = F.pad(canonical_mean.unsqueeze(0), (0, 1), value=1.0)  # [1,4]

        for i in range(actual_frames):
            guru.info(f"--- Frame {i} ---")
            T_i = transfms[0, i]        # [4,4]
            mean_h = (T_i @ padded_mean.T).squeeze()  # [4]
            mean_3d = mean_h[:3] / mean_h[3].clamp(min=1e-6)

            R_i = T_i[:3, :3]           # [3,3]
            quat_i_xyzw = roma.rotmat_to_unitquat(R_i.unsqueeze(0))  # [1,4] XYZW
            dyn_quat_xyzw = roma.quat_product(
                roma.quat_wxyz_to_xyzw(quat_i_xyzw),
                roma.quat_wxyz_to_xyzw(canonical_quat.unsqueeze(0)),
            )
            dyn_quat_wxyz = roma.quat_xyzw_to_wxyz(dyn_quat_xyzw).squeeze(0)

            print(f"  Position: {mean_3d.detach().cpu().numpy()}")
            print(f"  Rotation (Quat WXYZ): {dyn_quat_wxyz.detach().cpu().numpy()}")

    guru.info("=" * 20 + " Trajectory Print End " + "=" * 20)


# =============================================================================
# Helper: build reference camera trajectory from config
# =============================================================================

def build_reference_w2cs(
    cam_cfg: CameraConfig,
    num_frames_render: int,
    ref_w2c: torch.Tensor,  #  Accept a single (4,4) pose instead of the full w2cs list
) -> torch.Tensor:
    """
    Return [num_frames_render,4,4] world<-cam for a chosen reference trajectory,
    starting from the provided ref_w2c pose.
    """

    if cam_cfg.traj == "fixed":
        return get_ref_fixed_w2cs(ref_w2c=ref_w2c, num_frames=num_frames_render)
    elif cam_cfg.traj == "tilt-up":
        return get_ref_tilt_up_w2cs(ref_w2c=ref_w2c, num_frames=num_frames_render, degree=8.0)
    elif cam_cfg.traj == "pan-right":
        return get_ref_pan_right_w2cs(ref_w2c=ref_w2c, num_frames=num_frames_render, degree=5.0)
    elif cam_cfg.traj == "dolly-up":
        return get_ref_dolly_up_w2cs(ref_w2c=ref_w2c, num_frames=num_frames_render, distance=-0.9)
    elif cam_cfg.traj == "dolly-right":
        return get_ref_dolly_right_w2cs(ref_w2c=ref_w2c, num_frames=num_frames_render, distance=0.9)
    elif cam_cfg.traj == "dolly-out":
        return get_ref_dolly_out_w2cs(ref_w2c=ref_w2c, num_frames=num_frames_render, distance=-8.0)
    else:
        raise ValueError(f"Unknown camera.traj '{cam_cfg.traj}'")


# =============================================================================
# Helper: interpolate training camera poses
# =============================================================================

def interpolate_camera_poses(
    ts_render: torch.Tensor,
    train_rots_6d: torch.Tensor,
    train_transls: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    num_frames_trained = len(train_rots_6d)
    if num_frames_trained == 0:
        guru.warning("No training frames to interpolate, returning identity poses.")
        return torch.eye(4, device=device).unsqueeze(0).repeat(len(ts_render), 1, 1)

    last_frame_idx_trained = num_frames_trained - 1
    t_floor_float = torch.floor(ts_render)
    t_floor = t_floor_float.long()
    t_ceil = torch.ceil(ts_render).long()
    alpha = (ts_render - t_floor_float).unsqueeze(-1)  # [N,1]

    t_floor = torch.clamp(t_floor, 0, last_frame_idx_trained)
    t_ceil = torch.clamp(t_ceil, 0, last_frame_idx_trained)

    rots_6d_floor = train_rots_6d[t_floor]      # [N,6]
    rots_6d_ceil = train_rots_6d[t_ceil]        # [N,6]
    transls_floor = train_transls[t_floor]      # [N,3]
    transls_ceil = train_transls[t_ceil]        # [N,3]

    rots_6d_interp = torch.lerp(rots_6d_floor, rots_6d_ceil, alpha)
    transls_interp = torch.lerp(transls_floor, transls_ceil, alpha)

    rots_mat_interp = cont_6d_to_rmat(rots_6d_interp)         # [N,3,3]
    w2cs_interp = rt_to_mat4(rots_mat_interp, transls_interp) # [N,4,4]
    return w2cs_interp


# =============================================================================
# Helpers: render FG / BG at a given (normalized) time and camera
# =============================================================================

def render_fg_bg_at_time(
    renderer: "Renderer",
    t_norm_single: torch.Tensor,   # shape [1,1] normalized time
    w2c: torch.Tensor,             # [4,4] world<-cam
    K: torch.Tensor,               # [3,3]
    img_wh: tuple,
    continuous_motion_model,     
):
    """
    Returns:
      fg_img_np:  HxWx3 uint8
      fg_acc_np:  HxW float32 in [0,1]
      bg_img_np:  HxWx3 uint8
    """
    device = w2c.device
    with torch.no_grad():
        # --- FG by continuous model ---
        bases = continuous_motion_model.forward_extrap(t_norm_single.view(1, 1))  # [K,1,9]
        bases = bases[:, 0, :]               # [K,9]
        rots = bases[:, :6]                  # [K,6]
        trans = bases[:, 6:]                 # [K,3]

        fg_coefs = renderer.model.fg.get_coefs()             # [G,K]
        canonical_means = renderer.model.fg.params["means"]  # [G,3]
        canonical_quats = renderer.model.fg.get_quats()      # [G,4] WXYZ

        rots_final = torch.einsum("gk,kd->gd", fg_coefs, rots)      # [G,6]
        trans_final = torch.einsum("gk,kd->gd", fg_coefs, trans)    # [G,3]

        rotmats_final = cont_6d_to_rmat(rots_final)                 # [G,3,3]
        means_fg = (
            torch.bmm(rotmats_final, canonical_means.unsqueeze(-1)).squeeze(-1)
            + trans_final
        )  # [G,3]

        quats_transform_xyzw = roma.rotmat_to_unitquat(rotmats_final)  # [G,4] XYZW
        quats_final_xyzw = roma.quat_product(
            quats_transform_xyzw,
            roma.quat_wxyz_to_xyzw(canonical_quats),
        )
        quats_fg = F.normalize(roma.quat_xyzw_to_wxyz(quats_final_xyzw), p=2, dim=-1)  # [G,4] WXYZ

        # --- render FG-only ---
        out_fg = renderer.model.render(
            t=None,
            w2cs=w2c[None, ...],   # [1,4,4]
            Ks=K[None, ...],       # [1,3,3]
            img_wh=img_wh,
            return_depth=False,
            fg_only=True,          # ★ only FG
            means=means_fg,        # [G,3]
            quats=quats_fg,        # [G,4]
        )

        fg_img = (out_fg["img"][0].clamp(0,1).cpu().numpy() * 255.0).astype(np.uint8)
        if "acc" in out_fg:
            fg_acc = out_fg["acc"][0].squeeze(-1).cpu().numpy().astype(np.float32).clip(0.0, 1.0)
        else:
            fg_acc = (np.mean(fg_img.astype(np.float32)/255.0, axis=-1)).astype(np.float32).clip(0.0,1.0)

        # --- render BG-only ---
        if renderer.model.has_bg:
            num_fg = renderer.model.fg.num_gaussians
            num_bg = renderer.model.bg.num_gaussians
            N = num_fg + num_bg
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            mask[num_fg:] = True  # keep BG only

            out_bg = renderer.model.render(
                t=None,                        # BG static
                w2cs=w2c[None, ...],
                Ks=K[None, ...],
                img_wh=img_wh,
                return_depth=False,
                fg_only=False,                 # allow BG
                filter_mask=mask,              # ★ BG-only
            )
            bg_img = (out_bg["img"][0].clamp(0,1).cpu().numpy() * 255.0).astype(np.uint8)
        else:
            H, W = img_wh[1], img_wh[0]
            bg_img = np.full((H, W, 3), 127, dtype=np.uint8)

    return fg_img, fg_acc, bg_img


# =============================================================================
# Main
# =============================================================================

def main(cfg: VideoConfig):
    # -------------------------------------------------------------------------
    # 0. Load dataset / renderer / checkpoints
    # -------------------------------------------------------------------------
    train_dataset = get_train_val_datasets(cfg.data, load_val=False)[0]
    num_frames_trained = train_dataset.num_frames
    last_frame_idx_trained = num_frames_trained - 1
    guru.info(f"Training dataset has {num_frames_trained} frames (0..{last_frame_idx_trained})")

    # Load GT images for raw reference
    guru.info("Loading all training images for raw reference video...")
    gt_images_np = []
    try:
        gt_img_dir = train_dataset.img_dir
        gt_img_ext = train_dataset.img_ext
        gt_img_files = sorted([f for f in os.listdir(gt_img_dir) if f.endswith(gt_img_ext)])
        gt_img_paths = [os.path.join(gt_img_dir, f) for f in gt_img_files]
        assert len(gt_img_paths) == num_frames_trained, "Image file count mismatch"
        for path in tqdm(gt_img_paths, desc="Loading GT Images"):
            gt_images_np.append(iio.imread(path))
        guru.info(f"Loaded {len(gt_images_np)} GT images.")
    except Exception as e:
        guru.error(f"Failed to load GT images: {e}. 'video_raw_reference.mp4' will be empty.")
        gt_images_np = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"

    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        False,
        work_dir=cfg.work_dir,
        port=None,
    )
    assert (num_frames_trained == renderer.num_frames), "Frame count mismatch vs renderer!"
    guru.info(f"Rendering video from global_step={renderer.global_step}")

    # -------------------------------------------------------------------------
    # 1. Load continuous models
    # -------------------------------------------------------------------------
    continuous_motion_model = None
    continuous_camera_model = None
    try:
        from flow3d.continuous_4D_motion import ContinuousMotionBases, ContinuousCameraPose
    except ImportError:
        guru.error("Could not import ContinuousMotionBases / ContinuousCameraPose.")
        ContinuousMotionBases = None
        ContinuousCameraPose = None

    if ContinuousMotionBases is not None:
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
            guru.error(f"Motion ckpt not found: {ckpt_m}")

    if ContinuousCameraPose is not None:
        ckpt_c = f"{cfg.work_dir}/checkpoints/continuous_camera_pose.ckpt"
        if os.path.exists(ckpt_c):
            try:
                continuous_camera_model = ContinuousCameraPose(
                    num_frames=num_frames_trained,
                    num_control_points=cfg.num_control_points,
                    degree=cfg.spline_degree,
                ).to(device)
                continuous_camera_model.load_state_dict(
                    torch.load(ckpt_c, map_location=device, weights_only=True)
                )
                continuous_camera_model.eval()
                guru.info(f"Loaded continuous camera model: {ckpt_c}")
            except Exception as e:
                guru.error(f"Failed loading camera model: {e}")
                continuous_camera_model = None
        else:
            guru.warning(f"Camera ckpt not found: {ckpt_c}")

    assert (continuous_motion_model is not None), "continuous_motion_model is required."

    if cfg.debug_print_single_gaussian:
        print_single_gaussian_trajectory(renderer, num_frames_to_print=5)

    # -------------------------------------------------------------------------
    # 2. Extract camera info / intrinsics
    # -------------------------------------------------------------------------
    train_w2cs = train_dataset.get_w2cs().to(device)      # [T,4,4] world<-cam
    avg_w2c = get_avg_w2c(train_w2cs)                     # [4,4]

    Ks = train_dataset.get_Ks().to(device)                # [T,3,3]
    K0 = Ks[0]
    img_wh = train_dataset.get_img_wh()

    train_c2ws = torch.linalg.inv(train_w2cs)
    lookat = get_lookat(train_c2ws[:, :3, -1], train_c2ws[:, :3, 2])
    up = torch.tensor([0.0, 0.0, 1.0], device=device)

    rc_train_c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(avg_w2c), train_c2ws)
    rc_pos = rc_train_c2ws[:, :3, -1]
    rads = (rc_pos.amax(0) - rc_pos.amin(0)) * 1.25

    train_rots_mat = train_w2cs[:, :3, :3]
    train_transls = train_w2cs[:, :3, 3]
    train_rots_6d = rmat_to_cont_6d(train_rots_mat)
    guru.info("Extracted 6D rotations and translations from training cameras for interpolation.")

    # -------------------------------------------------------------------------
    # 3. Build global time axis
    # -------------------------------------------------------------------------
    ts = _get_extrapolated_ts_uniform_step(
        num_replay_frames=cfg.num_replay_frames,
        num_past_extrap_frames=cfg.num_past_extrap_frames,
        num_future_extrap_frames=cfg.num_future_extrap_frames,
        num_frames=num_frames_trained,
        device=device,
    )
    num_frames_render = len(ts)
    guru.info(f"Generated {num_frames_render} time steps via extrapolated timeline.")
    if num_frames_render == 0:
        guru.error("No frames to render. Abort.")
        return

    t_min_render = ts.min().item()
    t_max_render = ts.max().item()
    guru.info(
        f"Rendering time range: [{t_min_render:.2f}, {t_max_render:.2f}] "
        f"(training range is [0,{last_frame_idx_trained}])"
    )
    ts_norm = normalize_time_idx_to_norm(ts, num_frames_trained).unsqueeze(-1)


    # -------------------------------------------------------------------------
    # 4. Camera trajectories
    # -------------------------------------------------------------------------
    cam_cfg = cfg.camera

    # Decide the "starting pose" used for the reference trajectory (ref_w2c_start_pose)
    ref_w2c_start_pose = None

    # User requirement: if the mode is 'reference' and we have a continuous camera model,
    # then use the continuous camera pose at t=0 as the starting pose.
    if cam_cfg.mode == "reference" and continuous_camera_model is not None:
        guru.info("Reference mode: Using continuous camera pose at t=0 as reference start.")
        # Compute normalized time for t=0
        t_zero = torch.tensor([0.0], device=device)
        t_zero_norm = normalize_time_idx_to_norm(t_zero, num_frames_trained).unsqueeze(-1)

        # Infer the pose at t=0 from the continuous model
        with torch.no_grad():
            poses_9d_render = continuous_camera_model.forward_extrap(t_zero_norm)  # [1, 9]
            rots_6d = poses_9d_render[..., :6]      # [1, 6]
            transls = poses_9d_render[..., 6:]      # [1, 3]
            rots_mat = cont_6d_to_rmat(rots_6d)      # [1, 3, 3]
            ref_w2c_start_pose = rt_to_mat4(rots_mat, transls).squeeze(0) # [4, 4]

    # If the above condition is not met (for example, in 'continuous' mode,
    # or in 'reference' mode but without a continuous camera model),
    # fall back to the original logic: use a discrete training-frame pose as the reference start.
    if ref_w2c_start_pose is None:
        if cam_cfg.mode == "reference":
            guru.warning("Could not use continuous cam for ref start (model missing?). "
                         "Falling back to discrete training camera.")

        # Original logic: select from train_w2cs
        ref_idx = int(max(0, min(cam_cfg.ref, train_w2cs.shape[0] - 1)))
        ref_w2c_start_pose = train_w2cs[ref_idx]

    # (a) reference trajectory (red reference camera)
    #     Now uses the computed ref_w2c_start_pose
    w2cs_ref_list = build_reference_w2cs(
        cam_cfg=cam_cfg,
        num_frames_render=num_frames_render,
        ref_w2c=ref_w2c_start_pose, 
    )

    # (b) main trajectory (green main camera)
    w2cs_main_list = w2cs_ref_list
    if cam_cfg.mode == "continuous":
        if continuous_camera_model is None:
            guru.warning("camera.mode='continuous' but no continuous_camera_model loaded. Fallback to reference.")
        else:
            guru.info("Generating main camera poses from continuous camera model.")
            with torch.no_grad():
                poses_9d_render = continuous_camera_model.forward_extrap(ts_norm)  # [N,9]
                rots_6d = poses_9d_render[..., :6]
                transls = poses_9d_render[..., 6:]
                rots_mat = cont_6d_to_rmat(rots_6d)
                w2cs_main_list = rt_to_mat4(rots_mat, transls)
            guru.info(f"Continuous camera produced {len(w2cs_main_list)} poses.")

    guru.info("Generating interpolated GT camera poses for baseline video...")
    w2cs_interp_cam_list = interpolate_camera_poses(ts, train_rots_6d, train_transls, device)

    # (Pad/Slice logic 
    def _pad_or_slice(seq: torch.Tensor, target_len: int) -> torch.Tensor:
        if len(seq) == target_len:
            return seq
        if len(seq) > target_len:
            return seq[:target_len]
        last_pose = seq[-1:] if len(seq) > 0 else torch.eye(4, device=device)[None]
        padding = last_pose.repeat(target_len - len(seq), 1, 1)
        return torch.cat([seq, padding], dim=0)

    w2cs_main_list = _pad_or_slice(w2cs_main_list, num_frames_render)
    w2cs_ref_list = _pad_or_slice(w2cs_ref_list, num_frames_render)
    w2cs_interp_cam_list = _pad_or_slice(w2cs_interp_cam_list, num_frames_render)
    guru.info(f"All camera trajectories padded to {num_frames_render} frames.")


    # -------------------------------------------------------------------------
    # 5. Plots (optional)
    # -------------------------------------------------------------------------
    plotting_data_available = False
    t_discrete_np = torch.arange(num_frames_trained).cpu().numpy()
    try:
        discrete_rots_gt_np = renderer.model.motion_bases.params["rots"].detach().cpu().numpy()      # [K,T,6]
        discrete_transls_gt_np = renderer.model.motion_bases.params["transls"].detach().cpu().numpy()# [K,T,3]
        discrete_cam_rots_mat = renderer.model.camera_poses.params["Rs"].detach().clone()            # [T,3,3]
        discrete_cam_transls_vec = renderer.model.camera_poses.params["ts"].detach().clone().squeeze(-1)  # [T,3]
        discrete_cam_rots_6d_gt_np = rmat_to_cont_6d(discrete_cam_rots_mat).cpu().numpy()            # [T,6]
        discrete_cam_transls_gt_np = discrete_cam_transls_vec.cpu().numpy()                          # [T,3]
        plotting_data_available = True
    except KeyError as e:
        guru.error(f"Cannot plot GT data: {e}")

    t_plot_min = t_min_render - cfg.plot_extrapolation_frames
    t_plot_max = t_max_render + cfg.plot_extrapolation_frames
    t_span_plot_frames = torch.linspace(t_plot_min, t_plot_max, 200, device=device)
    t_span_plot_norm = normalize_time_idx_to_norm(t_span_plot_frames, num_frames_trained).unsqueeze(-1)
    t_plot_np = t_span_plot_frames.cpu().numpy()

    video_dir_name = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    video_dir_name += f"_{cfg.num_past_extrap_frames}p_{cfg.num_replay_frames}r_{cfg.num_future_extrap_frames}f"
    if cfg.camera.mode == "continuous":
        video_dir_name += "_Cam_Continuous"
    elif cfg.camera.mode == "reference":
        video_dir_name += f"_Cam_Ref_{cfg.camera.traj}"

    video_dir = f"{cfg.work_dir}/videos/{video_dir_name}"
    os.makedirs(video_dir, exist_ok=True)

    TITLE_FONTSIZE = 22
    LABEL_FONTSIZE = 20
    TICK_FONTSIZE = 17
    LINE_WIDTH = 4.0
    SCATTER_SIZE = 130

    if continuous_camera_model is not None:
        guru.info("Plotting continuous camera trajectory...")
        with torch.no_grad():
            poses_9d_plot = continuous_camera_model.forward_extrap(t_span_plot_norm).cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(18, 8))
        lines_all = ax.plot(t_plot_np, poses_9d_plot, linewidth=LINE_WIDTH)
        if plotting_data_available:
            discrete_cam_9d_gt_np = np.concatenate((discrete_cam_rots_6d_gt_np, discrete_cam_transls_gt_np), axis=1)
            for i_dim in range(9):
                ax.scatter(
                    t_discrete_np, discrete_cam_9d_gt_np[:, i_dim],
                    marker="o", s=SCATTER_SIZE, color=lines_all[i_dim].get_color(), alpha=0.6,
                )
        ax.set_title("Continuous Camera Pose - 9D (6D Rot + 3D Trans)", fontsize=TITLE_FONTSIZE)
        ax.set_ylabel("Value", fontsize=LABEL_FONTSIZE)
        ax.set_xlabel("Time (frame idx)", fontsize=LABEL_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        fig.tight_layout()
        save_path = os.path.join(video_dir, "cont_camera_pose_9D.png")
        fig.savefig(save_path); plt.close(fig)

    if continuous_motion_model is not None:
        guru.info("Plotting continuous motion bases trajectories...")
        with torch.no_grad():
            pred_bases_vis = continuous_motion_model.forward_extrap(t_span_plot_norm)  # [K,S,9]
        pred_bases_np = pred_bases_vis.cpu().numpy()
        K_vis = getattr(continuous_motion_model, "K", pred_bases_np.shape[0])
        plot_limit = min(K_vis, 5)
        for k in range(plot_limit):
            all_cont = pred_bases_np[k, :, :]  # [S,9]
            fig, ax = plt.subplots(1, 1, figsize=(18, 8))
            lines_all = ax.plot(t_plot_np, all_cont, linewidth=LINE_WIDTH)
            if plotting_data_available:
                discrete_rots_k = discrete_rots_gt_np[k, :, :]
                discrete_transls_k = discrete_transls_gt_np[k, :, :]
                discrete_9d_k = np.concatenate((discrete_rots_k, discrete_transls_k), axis=1)
                for i_dim in range(9):
                    ax.scatter(
                        t_discrete_np, discrete_9d_k[:, i_dim],
                        marker="o", s=SCATTER_SIZE, color=lines_all[i_dim].get_color(), alpha=0.6,
                    )
            ax.set_title(f"Motion Basis {k} - 9D (6D Rot + 3D Trans)", fontsize=TITLE_FONTSIZE)
            ax.set_ylabel("Value", fontsize=LABEL_FONTSIZE)
            ax.set_xlabel("Time (frame idx)", fontsize=LABEL_FONTSIZE)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
            fig.tight_layout()
            save_path = os.path.join(video_dir, f"cont_motion_basis_{k:02d}_9D.png")
            fig.savefig(save_path); plt.close(fig)

    # -------------------------------------------------------------------------
    # 6. Viser debug visualization     # -------------------------------------------------------------------------
    try:
        server = get_server(port=cfg.port)
        viser_scale = 0.05
        for i, train_w2c in enumerate(train_w2cs):
            c2w = torch.linalg.inv(train_w2c).cpu().numpy()
            server.scene.add_camera_frustum(
                f"/train/{i:03d}", fov=np.pi / 3, aspect=1.0, scale=viser_scale * 0.5, color=(0, 0, 0),
                wxyz=vt.SO3.from_matrix(c2w[:3, :3]).wxyz, position=c2w[:3, -1],
            )
        for i, w2c_ref in enumerate(w2cs_ref_list):
            c2w_ref = torch.linalg.inv(w2c_ref).cpu().numpy()
            server.scene.add_camera_frustum(
                f"/ref_path/{i:03d}", fov=np.pi / 3, aspect=1.0, scale=viser_scale, color=(255, 0, 0),
                wxyz=vt.SO3.from_matrix(c2w_ref[:3, :3]).wxyz, position=c2w_ref[:3, -1],
            )
        for i, w2c_interp in enumerate(w2cs_interp_cam_list):
            c2w_interp = torch.linalg.inv(w2c_interp).cpu().numpy()
            server.scene.add_camera_frustum(
                f"/interp_gt_path/{i:03d}", fov=np.pi / 3, aspect=1.0, scale=viser_scale * 0.9, color=(255, 255, 0),
                wxyz=vt.SO3.from_matrix(c2w_interp[:3, :3]).wxyz, position=c2w_interp[:3, -1],
            )
        for i, w2c_main in enumerate(w2cs_main_list):
            c2w_main = torch.linalg.inv(w2c_main).cpu().numpy()
            server.scene.add_camera_frustum(
                f"/main_path/{i:03d}", fov=np.pi / 3, aspect=1.0, scale=viser_scale * 0.8, color=(0, 255, 0),
                wxyz=vt.SO3.from_matrix(c2w_main[:3, :3]).wxyz, position=c2w_main[:3, -1],
            )
        avg_c2w = torch.linalg.inv(avg_w2c).cpu().numpy()
        server.scene.add_camera_frustum(
            "/avg_camera", fov=np.pi / 3, aspect=1.0, scale=viser_scale * 1.2, color=(0, 0, 255),
            wxyz=vt.SO3.from_matrix(avg_c2w[:3, :3]).wxyz, position=avg_c2w[:3, -1],
        )
        guru.info(f"Viser visualization at http://localhost:{cfg.port}")
    except Exception as e:
        guru.error(f"Failed to start Viser server: {e}")

    # -------------------------------------------------------------------------
    # 7. Rendering loop
    # -------------------------------------------------------------------------
    BLANK_THRESHOLD = 0.8
    BLANK_DILATE_ITERATIONS = 1
    EDGE_THRESHOLD = 50.0
    DILATE_ITERATIONS = 1

    # --- Brightness control parameters for the lapse trajectory (tunable) ---
    LAPSE_ALPHA_SCALE = 0.6   # Global intensity scale (0~1); smaller means darker
    LAPSE_HALF_LIFE   = 0    # Exponential decay half-life (in frames); smaller means faster decay; <=0 disables decay
    MAX_ALPHA_CAP     = 0.85  # Maximum per-frame alpha cap to avoid overexposure

    # --- Foreground detection threshold for REMOVE (tunable) ---
    FG_THR = 1e-3

    video_main = []
    video_interpolated = []
    mask_video = []
    src_video = []
    raw_comparison_video = []
    vace_video = []
    vace_mask = []
    video_gt = None
    ts_gt_idx_round = None
    gt_info = None

    src_video_lapse = []

    src_video_remove = []
    src_mask_remove = []

    src_video_bg = []    

    # GT
    if cfg.gt.gt_dir is not None:
        gt_info = load_gt_info(cfg.gt.gt_dir)
        if gt_info is not None:
            first_gt_idx, last_gt_idx, T_gt, gt_filename_map, available_gt_indices = gt_info
            video_gt = []
            t_start_gt_norm = first_gt_idx / T_gt
            t_end_gt_norm = last_gt_idx / T_gt
            duration_gt_norm = t_end_gt_norm - t_start_gt_norm
            time_step_gt_norm = duration_gt_norm / (cfg.num_replay_frames - 1) if cfg.num_replay_frames > 1 else 0.0
            t_start_render_gt_norm = t_start_gt_norm - cfg.num_past_extrap_frames * time_step_gt_norm
            t_end_render_gt_norm = t_end_gt_norm + cfg.num_future_extrap_frames * time_step_gt_norm
            total_render_frames = num_frames_render
            ts_gt_norm = torch.linspace(t_start_render_gt_norm, t_end_render_gt_norm, total_render_frames)
            ts_gt_idx_float = ts_gt_norm * T_gt
            ts_gt_idx_round = torch.round(ts_gt_idx_float).int().cpu().numpy()
            guru.info(f"GT: Calculated {len(ts_gt_idx_round)} GT frame indices.")
        else:
            guru.error("Failed to load GT info, gt.mp4 will not be generated.")

    known_replay_indices = np.linspace(0, cfg.num_replay_frames - 1, num_frames_trained).round().astype(int)
    vace_image_map = {replay_idx: img_idx for img_idx, replay_idx in enumerate(known_replay_indices)}
    if len(vace_image_map) != len(known_replay_indices):
        guru.warning("VACE: Mismatched known indices vs map; using overwrite policy.")
        vace_image_map = {}
        for img_idx, replay_idx in enumerate(known_replay_indices):
            vace_image_map[replay_idx] = img_idx

    guru.info(f"VACE: Mapping {num_frames_trained} images to {cfg.num_replay_frames} replay frames.")
    guru.info(f"VACE image map (replay_idx -> img_idx): {vace_image_map}")

    # Lapse accumulation buffers (reused per frame for re-accumulation; reinitialized each frame)
    H, W = img_wh[1], img_wh[0]
    decay_base = 1.0 if LAPSE_HALF_LIFE <= 0 else (0.5 ** (1.0 / LAPSE_HALF_LIFE))

    for i, (w2c_main, w2c_ref, w2c_interp_cam, t_val) in enumerate(
        tqdm(zip(w2cs_main_list, w2cs_ref_list, w2cs_interp_cam_list, ts),
             total=num_frames_render, desc="Rendering Frames")
    ):
        t_scalar = float(t_val.item())
        t_norm_single = normalize_time_idx_to_norm(t_val, num_frames_trained)  # scalar tensor

        is_extrapolated = (t_scalar < -1e-6) or (t_scalar > (last_frame_idx_trained + 1e-6))
        is_fractional = abs(t_scalar - round(t_scalar)) > 1e-6
        clamped_t_idx = max(0, min(int(round(t_scalar)), last_frame_idx_trained))

        # Main (FG spline + optional BG)
        with torch.inference_mode():
            # Build explicit FG+BG means/quats for the main render
            bases_main = continuous_motion_model.forward_extrap(t_norm_single.view(1, 1))  # [K,1,9]
            bases_main = bases_main[:, 0, :]
            rots_main = bases_main[:, :6]
            trans_main = bases_main[:, 6:]

            fg_coefs = renderer.model.fg.get_coefs()
            canonical_means = renderer.model.fg.params["means"]
            canonical_quats = renderer.model.fg.get_quats()

            rots_final = torch.einsum("gk,kd->gd", fg_coefs, rots_main)
            trans_final = torch.einsum("gk,kd->gd", fg_coefs, trans_main)

            rotmats_final = cont_6d_to_rmat(rots_final)
            means_fg_main = (torch.bmm(rotmats_final, canonical_means.unsqueeze(-1)).squeeze(-1) + trans_final)

            quats_transform_xyzw = roma.rotmat_to_unitquat(rotmats_final)
            quats_final_xyzw = roma.quat_product(quats_transform_xyzw, roma.quat_wxyz_to_xyzw(canonical_quats))
            quats_fg_main = F.normalize(roma.quat_xyzw_to_wxyz(quats_final_xyzw), p=2, dim=-1)

            if renderer.model.has_bg:
                means_bg = renderer.model.bg.params["means"]
                quats_bg = renderer.model.bg.get_quats()
                render_means_main = torch.cat([means_fg_main, means_bg], dim=0)
                render_quats_main = torch.cat([quats_fg_main, quats_bg], dim=0)
            else:
                render_means_main = means_fg_main
                render_quats_main = quats_fg_main

            render_args_main = {
                "t": None,
                "w2cs": w2c_main[None],
                "Ks": K0[None],
                "img_wh": img_wh,
                "return_depth": True,
                "means": render_means_main,
                "quats": render_quats_main,
            }

            # Interpolated baseline (camera from interpolated GT; FG linear blend when fractional)
            render_means_interp = None
            render_quats_interp = None
            render_t_interp = clamped_t_idx
            if is_fractional or is_extrapolated:
                t_0, t_1, alpha = 0, 1, t_scalar
                if t_scalar > last_frame_idx_trained:
                    t_0 = last_frame_idx_trained - 1; t_1 = last_frame_idx_trained
                    alpha = (t_scalar - float(t_0)) / max(float(t_1 - t_0), 1e-6)
                elif t_scalar >= 0:
                    t_0 = int(torch.floor(t_val).item())
                    t_1 = int(torch.ceil(t_val).item())
                    if t_0 == t_1:
                        t_1 = min(t_0 + 1, last_frame_idx_trained)
                    alpha = (t_scalar - float(t_0)) / max(float(t_1 - t_0), 1e-6)
                    if t_1 == t_0:
                        alpha = 0.0

                t_0 = max(0, min(t_0, last_frame_idx_trained))
                t_1 = max(0, min(t_1, last_frame_idx_trained))
                mb_params = renderer.model.motion_bases.params
                rots_floor = mb_params["rots"][:, t_0]
                rots_ceil = mb_params["rots"][:, t_1]
                trans_floor = mb_params["transls"][:, t_0]
                trans_ceil = mb_params["transls"][:, t_1]

                rots_interp = torch.lerp(rots_floor, rots_ceil, alpha)
                trans_interp = torch.lerp(trans_floor, trans_ceil, alpha)
                rots_final_lin = torch.einsum("gk,kd->gd", fg_coefs, rots_interp)
                trans_final_lin = torch.einsum("gk,kd->gd", fg_coefs, trans_interp)

                rotmats_lin = cont_6d_to_rmat(rots_final_lin)
                means_fg_lin = (torch.bmm(rotmats_lin, canonical_means.unsqueeze(-1)).squeeze(-1) + trans_final_lin)
                quats_transform_xyzw_lin = roma.rotmat_to_unitquat(rotmats_lin)
                quats_final_xyzw_lin = roma.quat_product(quats_transform_xyzw_lin, roma.quat_wxyz_to_xyzw(canonical_quats))
                quats_fg_lin = F.normalize(roma.quat_xyzw_to_wxyz(quats_final_xyzw_lin), p=2, dim=-1)

                if renderer.model.has_bg:
                    render_means_interp = torch.cat([means_fg_lin, means_bg], dim=0)
                    render_quats_interp = torch.cat([quats_fg_lin, quats_bg], dim=0)
                else:
                    render_means_interp = means_fg_lin
                    render_quats_interp = quats_fg_lin
                render_t_interp = None

            render_args_interp = {
                "t": render_t_interp,
                "w2cs": w2c_interp_cam[None],
                "Ks": K0[None],
                "img_wh": img_wh,
                "return_depth": False,
                "means": render_means_interp,
                "quats": render_quats_interp,
            }

            out_main = renderer.model.render(**render_args_main)
            img_main_t = out_main["img"][0]
            acc_tensor = out_main["acc"][0].squeeze(-1)
            depth_tensor = out_main["depth"][0].squeeze(-1)

            out_interp = renderer.model.render(**render_args_interp)
            img_interp_t = out_interp["img"][0]

        # Build occlusion/blank mask for other videos 
        acc_np = acc_tensor.cpu().numpy()
        depth_np = depth_tensor.cpu().numpy()
        mask_blank = acc_np < BLANK_THRESHOLD
        if BLANK_DILATE_ITERATIONS > 0:
            mask_blank = binary_dilation(mask_blank, iterations=BLANK_DILATE_ITERATIONS)
        try:
            sobel_d = sobel(depth_np)
            edges = sobel_d > EDGE_THRESHOLD
            mask_occ = binary_dilation(edges, iterations=DILATE_ITERATIONS) if DILATE_ITERATIONS > 0 else edges
            final_mask_np = np.logical_or(mask_blank, mask_occ)
        except Exception:
            final_mask_np = mask_blank

        mask_img = (final_mask_np * 255).astype(np.uint8)
        mask_video.append(mask_img)

        img_main_np = (img_main_t.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        img_interp_np = (img_interp_t.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

        # Raw reference (GT image)
        raw_idx = 0 if (is_extrapolated and t_scalar < 0) else (last_frame_idx_trained if is_extrapolated else clamped_t_idx)
        if gt_images_np:
            img_raw_np = gt_images_np[raw_idx] if 0 <= raw_idx < len(gt_images_np) else np.zeros_like(img_main_np)
        else:
            img_raw_np = np.zeros_like(img_main_np)

        # ==============================================================
        # Gaussian-level REMOVE + LAPSE
        # ==============================================================
        # 1) FG-only / BG-only at the current time
        fg_img_cur, fg_acc_cur, bg_img_cur = render_fg_bg_at_time(
            renderer=renderer,
            t_norm_single=t_norm_single,
            w2c=w2c_main,
            K=K0,
            img_wh=img_wh,
            continuous_motion_model=continuous_motion_model,
        )

        #  collect background-only frame
        src_video_bg.append(bg_img_cur)  

        # -set both "foreground + other regions needing fill (final_mask)" to gray 127 ---
        fg_mask_cur = (fg_acc_cur > FG_THR)  # HxW bool
        fill_mask = fg_mask_cur | final_mask_np
        remove_frame = bg_img_cur.copy()
        remove_frame[fill_mask] = 127
        src_video_remove.append(remove_frame)
        src_mask_remove.append((fg_acc_cur * 255.0).astype(np.uint8))

        # 2) LAPSE: accumulate foreground from historical frames 0..i, with decay and cap to avoid over-brightness
        accum_color = np.zeros((H, W, 3), dtype=np.float32)  # 0..1
        accum_alpha = np.zeros((H, W), dtype=np.float32)     # 0..1
        for k in range(i + 1):
            t_k_norm = normalize_time_idx_to_norm(ts[k], num_frames_trained).view(1, 1)
            fg_img_k, fg_acc_k, _ = render_fg_bg_at_time(
                renderer=renderer,
                t_norm_single=t_k_norm,
                w2c=w2c_main,   # Fix the current camera and replay historical FG
                K=K0,
                img_wh=img_wh,
                continuous_motion_model=continuous_motion_model,
            )
            Fcol = (fg_img_k.astype(np.float32) / 255.0)   # HxWx3
            a = fg_acc_k.astype(np.float32)                # HxW in [0,1]

            # Decay and global scaling + cap

            age = i - k
            if age == 0:
                # Current frame: keep the original fg_acc opacity
                # (optionally still cap it to MAX_ALPHA_CAP if needed)
                a = np.clip(a, 0.0, 1.0)
            else:
                decay = (decay_base ** age)
                a = np.clip(a * LAPSE_ALPHA_SCALE * decay, 0.0, MAX_ALPHA_CAP)


            # Over-composite onto the accumulated black background
            accum_color = accum_color * (1.0 - a)[..., None] + Fcol * a[..., None]
            accum_alpha = 1.0 - (1.0 - accum_alpha) * (1.0 - a)

        # 3) Over-composite the accumulated foreground onto the current BG-only image
        #    to obtain the final lapse image
        bg_lin = (bg_img_cur.astype(np.float32) / 255.0)
        lapse_lin = accum_color + (1.0 - accum_alpha)[..., None] * bg_lin
        lapse_img = (lapse_lin.clip(0.0, 1.0) * 255.0).astype(np.uint8)

        src_video_lapse.append(lapse_img)

        # --- Other existing outputs (kept) ---
        is_past = i < cfg.num_past_extrap_frames
        is_future = i >= (num_frames_render - cfg.num_future_extrap_frames)
        is_known_vace_frame = False
        frame_vace = None
        if not is_past and not is_future:
            replay_index = i - cfg.num_past_extrap_frames
            if replay_index in vace_image_map:
                image_index = vace_image_map[replay_index]
                if gt_images_np and image_index < len(gt_images_np):
                    frame_vace = gt_images_np[image_index]
                    is_known_vace_frame = True
        if is_known_vace_frame:
            frame_vace_mask = np.zeros((frame_vace.shape[0], frame_vace.shape[1]), dtype=np.uint8)
        else:
            frame_vace = np.full_like(img_main_np, 127)
            frame_vace_mask = np.full((img_main_np.shape[0], img_main_np.shape[1]), 255, dtype=np.uint8)

        video_main.append(img_main_np)
        video_interpolated.append(img_interp_np)
        raw_comparison_video.append(img_raw_np)
        vace_video.append(frame_vace)
        vace_mask.append(frame_vace_mask)

        if cfg.gt.gt_dir is not None and gt_info is not None and ts_gt_idx_round is not None:
            try:
                desired_gt_idx = ts_gt_idx_round[i]
                closest_map_idx = np.argmin(np.abs(available_gt_indices - desired_gt_idx))
                final_gt_idx = available_gt_indices[closest_map_idx]
                gt_filename = gt_filename_map[final_gt_idx]
                gt_frame_path = os.path.join(cfg.gt.gt_dir, gt_filename)
                gt_frame_np = iio.imread(gt_frame_path)
                if video_gt is None:
                    video_gt = []
                video_gt.append(gt_frame_np)
            except Exception as e:
                guru.error(f"GT: Failed to load frame {i} (idx {desired_gt_idx}): {e}")
                if video_gt is None:
                    video_gt = []
                video_gt.append(np.zeros_like(img_main_np))

        src_frame = img_main_np.copy()
        src_frame[final_mask_np] = 127
        src_video.append(src_frame)

    # -------------------------------------------------------------------------
    # 8. Save videos + config
    # -------------------------------------------------------------------------
    def save_video(filename, frames, fps, quality=8):
        path = f"{video_dir}/{filename}"
        guru.info(f"Saving {filename} ({len(frames)} frames)...")
        if not frames:
            guru.warning(f"No frames to save for {filename}")
            return
        try:
            iio.imwrite(
                path,
                np.stack(frames, 0),
                fps=fps,
                quality=quality,
                macro_block_size=2,
            )
        except Exception as e:
            guru.error(f"Failed to save {filename}: {e}")

  #  save_video("video_physics.mp4", video_main, cfg.fps)
   # save_video("video_interpolated_baseline.mp4", video_interpolated, cfg.fps)
    save_video("src_mask.mp4", mask_video, cfg.fps)
    save_video("src_video.mp4", src_video, cfg.fps)
   # save_video("video_raw_reference.mp4", raw_comparison_video, cfg.fps)
   # save_video("src_video_VACE.mp4", vace_video, cfg.fps)
   # save_video("src_mask_VACE.mp4", vace_mask, cfg.fps)
    if video_gt is not None:
        save_video("gt.mp4", video_gt, cfg.fps)

  #  save_video("src_video_lapse.mp4", src_video_lapse, cfg.fps)
  #  save_video("src_video_remove.mp4", src_video_remove, cfg.fps)
  #  save_video("src_mask_remove.mp4", src_mask_remove, cfg.fps)
  #  save_video("src_video_bg.mp4", src_video_bg, cfg.fps)  


    with open(f"{video_dir}/render_cfg.yaml", "w") as f:
        try:
            tyro.extras.dump_yaml(cfg, f)
        except Exception:
            yaml.dump(asdict(cfg), f, default_flow_style=None, sort_keys=False)

    guru.info(f"Saved videos and plots to {video_dir}")


if __name__ == "__main__":
    main(tyro.cli(VideoConfig))