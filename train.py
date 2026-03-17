import os
import os.path as osp
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Annotated, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tyro
import yaml
from loguru import logger as guru
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow3d.configs_4D_motion import (
    LossesConfig,
    OptimizerConfig,
    SceneLRConfig,
    Stage2Config,
    GeneralTrainingConfig,
    FGLRConfig, 
    BGLRConfig,
    MotionLRConfig,
    CameraPoseLRConfig,
    CameraScalesLRConfig
)
from flow3d.data import (
    BaseDataset,
    CustomDataConfig,
    get_train_val_datasets,
    iPhoneDataConfig,
    NvidiaDataConfig,
)
from flow3d.data.utils import to_device
from flow3d.init_utils import (
    init_bg,
    init_fg_from_tracks_3d,
    init_motion_params_with_procrustes,
    run_initial_optim,
    vis_init_params,
    init_trainable_poses,
)
from flow3d.scene_model import SceneModel
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.trainer import Trainer
from flow3d.validator import Validator
from flow3d.vis.utils import get_server
from flow3d.params import CameraScales

from flow3d.continuous_4D_motion import (
    ContinuousMotionBases,
    ContinuousCameraPose,
    t_norm_to_unit
)
from flow3d.transforms_4D_motion import (
    cont_6d_to_rmat_safe,
    rot_loss_6d_cosine,
    rot_loss_geodesic_safe,
    rmat_to_cont_6d_fallback,
    so3_log_safe
)

torch.set_float32_matmul_precision("high")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Stage 2 Helper Functions (Keep as before) ---
def normalize_time_idx_to_norm(t: torch.Tensor, T: int) -> torch.Tensor:
    # ... (rest of helper functions: load_discrete_from_ckpt, ls_init_from_discrete, sample_dense, _ensure_dir, plotting functions, ODE functions) ...
    if T <= 1:
        return torch.zeros_like(t, dtype=torch.float32)
    return (t.float() / (T - 1)) * 2.0 - 1.0

# ====================== read discrete supervision from Stage1 ckpt ======================
def load_discrete_from_ckpt(ckpt_path: str, device: torch.device):
    """
    parse from last.ckpt's state_dict:
      - discrete_bases_combined: [K, T, 9]  (rots6d + trans)
      - discrete_cam_poses_combined: [T, 9] (rots6d + trans)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    def _get(key):
        if key in sd:
            return sd[key]
        alt = f"model.{key}"
        if alt in sd:
            return sd[alt]
        raise KeyError(f"Key '{key}' not found in state_dict")

    # motion bases: rots(6D) + trans(3)
    rots6d = _get("motion_bases.params.rots").to(device)      # [K,T,6]
    trans  = _get("motion_bases.params.transls").to(device)      # [K,T,3]
    discrete_bases_combined = torch.cat([rots6d, trans], dim=-1)  # [K,T,9]

    # camera poses: R([T,3,3]) -> 6D
    Rs = _get("camera_poses.params.Rs").to(device)           # [T,3,3]
    ts = _get("camera_poses.params.ts").to(device)           # [T,3] or [T,3,1]
    if ts.dim() == 3 and ts.size(-1) == 1:
        ts = ts.squeeze(-1)                                   # [T,3]

    cam_r6 = rmat_to_cont_6d_fallback(Rs)                      # [T,6] 
    discrete_cam_poses_combined = torch.cat([cam_r6, ts], dim=-1)  # [T,9]

    discrete_bases_combined = discrete_bases_combined.float()
    discrete_cam_poses_combined = discrete_cam_poses_combined.float()

    return discrete_bases_combined, discrete_cam_poses_combined


# ====================== least-squares initialization of control points ======================
@torch.no_grad()
def ls_init_from_discrete(continuous_bases_model: ContinuousMotionBases,
                          continuous_camera_model: ContinuousCameraPose,
                          discrete_bases_combined: torch.Tensor,     # [K,T,9]
                          discrete_cam_poses_combined: torch.Tensor  # [T,9]
                          ):
    device = discrete_cam_poses_combined.device
    K, T, _ = discrete_bases_combined.shape

    # Build the basis matrix [T, m]
    t_all = torch.linspace(0, T-1, T, device=device).unsqueeze(-1)  # [T,1]
    t_norm = normalize_time_idx_to_norm(t_all, T)
    # t_norm_to_unit is imported from continuous_4D_motion
    B_full = continuous_bases_model.bspline(t_norm_to_unit(t_norm)) # [T, m]
    pinv   = torch.linalg.pinv(B_full)                           # [m, T]
    pinv_T = pinv.T                                               # [T, m]

    # motion: [K,9,T] @ [T,m] -> [K,9,m]
    C_motion = discrete_bases_combined.permute(0,2,1) @ pinv_T
    continuous_bases_model.control_points.copy_(C_motion)

    # camera: [9,T] @ [T,m] -> [9,m]
    C_cam = discrete_cam_poses_combined.T @ pinv_T
    continuous_camera_model.control_points.copy_(C_cam.unsqueeze(0))

# ====================== plotting and evaluation helper functions ======================

@torch.no_grad()
def sample_dense(mb_model, cam_model, T: int, S: int, device: torch.device, extrap_pct: float = 0.0):
    T1 = max(T - 1, 1)
    if extrap_pct > 0:
        t_start = -extrap_pct * T1
        t_end   = (1.0 + extrap_pct) * T1
    else:
        t_start = 0.0
        t_end   = T1

    u = torch.linspace(t_start, t_end, S, device=device).unsqueeze(-1)  # [S,1]
    u_norm = normalize_time_idx_to_norm(u, T)

    bases_S = mb_model.forward_extrap(u_norm).permute(1, 0, 2).contiguous()  # [S,K,9]
    cam_S   = cam_model.forward_extrap(u_norm)                              # [S,9]
    return (u.squeeze(-1).cpu().numpy(),
            bases_S.detach().cpu().numpy(),
            cam_S.detach().cpu().numpy())


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_series_with_scatter(t_dense, y_dense, t_disc, y_disc, title, labels, save_path):
    D = y_dense.shape[1]
    plt.figure(figsize=(11, 4.2))
    for d in range(D):
        plt.plot(t_dense, y_dense[:, d], linewidth=2, label=labels[d])
        plt.scatter(t_disc, y_disc[:, d], s=14, alpha=0.8)
    plt.xlabel("Frame idx")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(ncol=min(D, 6), fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def plot_3d_curve_with_scatter(xyz_dense, xyz_disc, title, save_path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz_dense[:, 0], xyz_dense[:, 1], xyz_dense[:, 2], linewidth=2)
    ax.scatter(xyz_disc[:, 0], xyz_disc[:, 1], xyz_disc[:, 2], s=12)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    try:
        max_range = (xyz_dense.max(0) - xyz_dense.min(0)).max()
        mid = (xyz_dense.max(0) + xyz_dense.min(0)) / 2
        ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
        ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
        ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def eval_and_plot(mb_model, cam_model,
                  disc_bases: torch.Tensor,   # [K,T,9]
                  disc_cam: torch.Tensor,     # [T,9]
                  cfg_s2: "Stage2Config",     # <-- use Stage2Config
                  epoch: int,
                  work_dir: str,
                  device: torch.device):
    K, T, _ = disc_bases.shape
    S = cfg_s2.dense_samples
    t_dense, bases_S, cam_S = sample_dense(
        mb_model, cam_model, T, S, device, extrap_pct=cfg_s2.extrap_pct
    )
    t_disc = np.arange(T)
    out_dir = osp.join(work_dir, f"plots/epoch_{epoch+1:04d}")
    _ensure_dir(out_dir)

    # ---- camera: rot6d & trans ----
    plot_series_with_scatter(
        t_dense, cam_S[:, :6], t_disc, disc_cam[:, :6].detach().cpu().numpy(),
        title="Camera Rotation (6D) — dense curve + discrete scatter",
        labels=[f"r6[{i}]" for i in range(6)],
        save_path=osp.join(out_dir, "camera_rot6d.png"),
    )
    plot_series_with_scatter(
        t_dense, cam_S[:, 6:9], t_disc, disc_cam[:, 6:9].detach().cpu().numpy(),
        title="Camera Translation (xyz) — dense curve + discrete scatter",
        labels=["tx", "ty", "tz"],
        save_path=osp.join(out_dir, "camera_trans.png"),
    )

    # ---- motion bases (plot only the first k) ----
    K_plot = min(cfg_s2.plot_top_k_bases, K)
    for k in range(K_plot):
        plot_series_with_scatter(
            t_dense, bases_S[:, k, :6],
            t_disc, disc_bases[k, :, :6].detach().cpu().numpy(),
            title=f"Motion Base {k} — Rotation 6D",
            labels=[f"r6[{i}]" for i in range(6)],
            save_path=osp.join(out_dir, f"base_{k:02d}_rot6d.png"),
        )
        plot_series_with_scatter(
            t_dense, bases_S[:, k, 6:9],
            t_disc, disc_bases[k, :, 6:9].detach().cpu().numpy(),
            title=f"Motion Base {k} — Translation (xyz)",
            labels=["tx", "ty", "tz"],
            save_path=osp.join(out_dir, f"base_{k:02d}_trans.png"),
        )
        if cfg_s2.plot_3d_translation:
            plot_3d_curve_with_scatter(
                bases_S[:, k, 6:9], disc_bases[k, :, 6:9].detach().cpu().numpy(),
                title=f"Motion Base {k} — 3D curve (dense) + points",
                save_path=osp.join(out_dir, f"base_{k:02d}_trans_3d.png"),
            )

def plot_gt_polylines(
    disc_bases: torch.Tensor,   # [K,T,9]
    disc_cam: torch.Tensor,     # [T,9]
    cfg_s2: Optional["Stage2Config"], # <-- use Stage2Config
    work_dir: str,
    subdir: str = "gt_lines",
):
    import numpy as np
    K, T, _ = disc_bases.shape
    time = np.arange(T)
    out_dir = osp.join(work_dir, f"plots/{subdir}")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Camera (GT) - Rotation (6D)")
    for j in range(6):
        ax.plot(time, disc_cam[:, j].detach().cpu().numpy(), label=f"rot6d[{j}]")
    ax.legend(ncol=3, fontsize=8); ax.set_xlabel("Frame"); ax.set_ylabel("Value")
    ax.grid(True, alpha=0.2); plt.tight_layout()
    plt.savefig(osp.join(out_dir, "camera_rot6d.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Camera (GT) - Translation (tx, ty, tz)")
    for j, lbl in enumerate(["tx","ty","tz"]):
        ax.plot(time, disc_cam[:, 6 + j].detach().cpu().numpy(), label=lbl)
    ax.legend(ncol=3, fontsize=8); ax.set_xlabel("Frame"); ax.set_ylabel("Value")
    ax.grid(True, alpha=0.2); plt.tight_layout()
    plt.savefig(osp.join(out_dir, "camera_trans.png")); plt.close(fig)

    K_plot = min(getattr(cfg_s2, "plot_top_k_bases", 3) if cfg_s2 is not None else 3, K)
    for k in range(K_plot):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"Motion Base {k} (GT) - Rotation (6D)")
        for j in range(6):
            ax.plot(time, disc_bases[k, :, j].detach().cpu().numpy(), label=f"rot6d[{j}]")
        ax.legend(ncol=3, fontsize=8); ax.set_xlabel("Frame"); ax.set_ylabel("Value")
        ax.grid(True, alpha=0.2); plt.tight_layout()
        plt.savefig(osp.join(out_dir, f"base_{k:02d}_rot6d.png")); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"Motion Base {k} (GT) - Translation (tx, ty, tz)")
        for j, lbl in enumerate(["tx","ty","tz"]):
            ax.plot(time, disc_bases[k, :, 6 + j].detach().cpu().numpy(), label=lbl)
        ax.legend(ncol=3, fontsize=8); ax.set_xlabel("Frame"); ax.set_ylabel("Value")
        ax.grid(True, alpha=0.2); plt.tight_layout()
        plt.savefig(osp.join(out_dir, f"base_{k:02d}_trans.png")); plt.close(fig)

    guru.info(f"[GT] Saved ground-truth polylines to {out_dir}")

# ====================== physics/ODE helper functions ======================

def _so3_angacc_from_rot6d(rot6d_fun, t_norm: torch.Tensor, h_norm: float) -> torch.Tensor:
    h = torch.full_like(t_norm, float(h_norm))
    r_p = rot6d_fun(t_norm + h)
    r_0 = rot6d_fun(t_norm)
    r_m = rot6d_fun(t_norm - h)
    R_p = cont_6d_to_rmat_safe(r_p)
    R_0 = cont_6d_to_rmat_safe(r_0)
    R_m = cont_6d_to_rmat_safe(r_m)
    w_plus  = so3_log_safe(R_p @ R_0.transpose(-1, -2)) / (h_norm + 1e-9)
    w_minus = so3_log_safe(R_0 @ R_m.transpose(-1, -2)) / (h_norm + 1e-9)
    d_omega = (w_plus - w_minus) / (h_norm + 1e-9)
    return d_omega

def _second_derivative_fd(fun, t_norm: torch.Tensor, h_norm: float):
    h = torch.full_like(t_norm, float(h_norm))
    y_p = fun(t_norm + h)
    y_0 = fun(t_norm)
    y_m = fun(t_norm - h)
    return (y_p - 2.0 * y_0 + y_m) / (h * h)

def _weighted_mean(res_per_sample: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w = w.clamp_min(1e-12)
    return (res_per_sample * w).sum() / w.sum()

# ===================================================================
# --- End of Added Helper Code ---
# ===================================================================


# --- TrainConfig uses GeneralTrainingConfig ---
@dataclass
class TrainConfig:
    """Main configuration holding specific configs and runtime paths."""
    work_dir: str # Keep work_dir here as it is often specified at runtime
    data: ( # Keep data config here for tyro subcommands
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")]
        | Annotated[CustomDataConfig, tyro.conf.subcommand(name="custom")]
        | Annotated[NvidiaDataConfig, tyro.conf.subcommand(name="nvidia")]
    )

    training: GeneralTrainingConfig = field(default_factory=GeneralTrainingConfig)

    lr: SceneLRConfig = field(default_factory=lambda: SceneLRConfig(
        fg=FGLRConfig(),                # create a default FGLRConfig
        bg=BGLRConfig(),                # create a default BGLRConfig
        motion_bases=MotionLRConfig(),    # create a default MotionLRConfig
        camera_poses=CameraPoseLRConfig(), # create a default CameraPoseLRConfig
        camera_scales=CameraScalesLRConfig() # create a default CameraScalesLRConfig
    ))
    loss: LossesConfig = field(default_factory=LossesConfig) # these can directly use the class name
    optim: OptimizerConfig = field(default_factory=OptimizerConfig) # because they already have default values
    stage2: Stage2Config = field(default_factory=Stage2Config)




    # --- Parameters removed from here and moved to GeneralTrainingConfig ---
    # num_fg: int = 80_000
    # num_bg: int = 80_000
    # num_motion_bases: int = 10
    # num_epochs: int = 4000
    # port: int | None = None
    # vis_debug: bool = False
    # batch_size: int = 16
    # num_dl_workers: int = 4
    # validate_every: int = 50
    # save_videos_every: int = 50
    # use_2dgs: bool = False


# --- main function uses cfg.training.* ---
def main(cfg: TrainConfig):
    backup_code(cfg.work_dir)
    train_dataset, train_video_view, val_img_dataset, val_kpt_dataset = (
        get_train_val_datasets(cfg.data, load_val=True)
    )
    guru.info(f"Training dataset has {train_dataset.num_frames} frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg.work_dir, exist_ok=True)
    # Save the *entire* config object, which now includes the nested 'training' part

    with open(f"{cfg.work_dir}/cfg.yaml", "w") as f:
        # directly use asdict and yaml.dump, no longer attempt tyro.extras
            guru.info("Saving configuration using PyYAML...")
            try:
             # default_flow_style=None lets PyYAML choose the style automatically, usually more readable
                yaml.dump(asdict(cfg), f, default_flow_style=None, sort_keys=False)
            except Exception as e:
                guru.error(f"Failed to save config: {e}")
            # as a fallback, try printing it to the log
                guru.warning(f"Config object: {asdict(cfg)}")

    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    initialize_and_checkpoint_model(
        cfg, train_dataset, device, ckpt_path # Pass the whole cfg
    )

    # ============ Stage 1 ============ #
    guru.info("--- Starting Stage 1 Training ---")

    trainer, start_epoch = Trainer.init_from_checkpoint(
        ckpt_path,
        device,
        cfg.training.use_2dgs, # <-- Use cfg.training.*
        cfg.lr, # Pass LR config group
        cfg.loss, # Pass Loss config group
        cfg.optim, # Pass Optim config group
        work_dir=cfg.work_dir,
        port=cfg.training.vis_port, # <-- Use cfg.training.*
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size_s1, # <-- Use cfg.training.*
        num_workers=cfg.training.num_dl_workers, # <-- Use cfg.training.*
        persistent_workers=(cfg.training.num_dl_workers > 0), # Added condition
        collate_fn=BaseDataset.train_collate_fn,
    )

    validator = Validator(
        model=trainer.model,
        device=device,
        train_loader=(DataLoader(train_video_view, batch_size=1) if train_video_view else None),
        val_img_loader=(DataLoader(val_img_dataset, batch_size=1) if val_img_dataset else None),
        val_kpt_loader=(DataLoader(val_kpt_dataset, batch_size=1) if val_kpt_dataset else None),
        save_dir=cfg.work_dir,
    )

    # <-- Use cfg.training.num_epochs_s1 -->
    for epoch in tqdm(range(start_epoch, cfg.training.num_epochs_s1)):
        trainer.set_epoch(epoch)

        ema = {}
        ema_beta = 0.9
        step = 0
        log_every = getattr(cfg.training, "log_every_s1", 200)  # use default 200 if not set

        for batch in train_loader:   # no progress bar
            step += 1
            batch = to_device(batch, device)

            # support multiple returns: dict / (loss, logs) / scalar
            out = trainer.train_step(batch)

            if isinstance(out, tuple) and len(out) == 2:
                loss_val, logs = out
                logs = dict(logs)
                if "loss" not in logs and loss_val is not None:
                    logs["loss"] = loss_val
            elif isinstance(out, dict):
                logs = dict(out)
            else:
                logs = {"loss": out}  # scalar or tensor

            # convert tensors to float for printing
            clean_logs = {}
            for k, v in logs.items():
                try:
                    if isinstance(v, torch.Tensor):
                        v = float(v.detach().cpu().item())
                    elif hasattr(v, "item"):
                        v = float(v.item())
                    else:
                        v = float(v)
                except Exception:
                    continue
                clean_logs[k] = v

            # EMA smoothing
            for k, v in clean_logs.items():
                if k not in ema:
                    ema[k] = v
                else:
                    ema[k] = ema_beta * ema[k] + (1 - ema_beta) * v

            # write logs every N steps (no progress bar, logs only)
            if step % log_every == 0 or step == 1:
                # only print common key items; you can also directly print all keys in clean_logs
                keys_to_show = ("loss", "loss_motion", "loss_camera", "loss_reg", "total")
                msg = " ".join([f"{k}={ema.get(k, clean_logs.get(k)):.6f}"
                                for k in keys_to_show if (k in ema) or (k in clean_logs)])
                if not msg:
                    msg = " ".join([f"{k}={v:.6f}" for k, v in clean_logs.items()])
                guru.info(f"[S1 e{epoch+1} s{step}] {msg}")

        # validation and saving
        if (epoch + 1) % cfg.training.validate_every_s1 == 0 or epoch == cfg.training.num_epochs_s1 - 1:
            val_logs = validator.validate()
            trainer.log_dict(val_logs)

        if (epoch + 1) % cfg.training.save_videos_every_s1 == 0 or epoch == cfg.training.num_epochs_s1 - 1:
            validator.save_train_videos(epoch)

        trainer.save_checkpoint(ckpt_path)





    guru.info("Stage 1 finished. Model saved to last.ckpt")


    # ============ Stage 2 ============ #
    guru.info("--- Starting Stage 2 Training (from new logic) ---")

    with open(osp.join(cfg.work_dir, "stage2_fit_cfg.yaml"), "w") as f:
        yaml.dump(asdict(cfg.stage2), f)

    guru.info(f"Loading discrete supervision from {ckpt_path}")
    disc_bases, disc_cam = load_discrete_from_ckpt(ckpt_path, device)
    K, T, _ = disc_bases.shape

    mb_model = ContinuousMotionBases(
        num_bases=K,
        num_frames=T,
        num_control_points=cfg.stage2.num_control_points,
        degree=cfg.stage2.spline_degree
    ).to(device)

    cam_model = ContinuousCameraPose(
        num_frames=T,
        num_control_points=cfg.stage2.num_control_points,
        degree=cfg.stage2.spline_degree
    ).to(device)

    ls_init_from_discrete(mb_model, cam_model, disc_bases, disc_cam)

    plot_gt_polylines( # Pass cfg.stage2 here now
        disc_bases.to(device),
        disc_cam.to(device),
        cfg_s2=cfg.stage2, # <-- Pass stage2 config
        work_dir=cfg.work_dir,
        subdir="gt_lines",
    )

    mb_model.eval(); cam_model.eval()
    with torch.no_grad():
        eval_and_plot( # Pass cfg.stage2 here now
            mb_model, cam_model,
            disc_bases=disc_bases,
            disc_cam=disc_cam,
            cfg_s2=cfg.stage2, # <-- Pass stage2 config
            epoch=-1,
            work_dir=cfg.work_dir,
            device=device,
        )
    guru.info("[Init] Plotted LS-initialized curves vs GT at plots/epoch_0000/")

    with torch.no_grad():
        mb_model.eval(); cam_model.eval()
        t_all = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(-1)
        t_norm_all = normalize_time_idx_to_norm(t_all, T)
        pred_bases_all = mb_model(t_norm_all)
        pred_cam_all   = cam_model(t_norm_all)
        loss_motion0 = F.mse_loss(pred_bases_all, disc_bases)
        pred_r6_0 = pred_cam_all[..., :6]; pred_t_0  = pred_cam_all[..., 6:9]
        tgt_r6_0  = disc_cam[..., :6];   tgt_t_0   = disc_cam[..., 6:9]
        loss_cam_t0 = F.l1_loss(pred_t_0, tgt_t_0)
        loss_cam_r0 = rot_loss_6d_cosine(pred_r6_0, tgt_r6_0)
        loss_camera0 = loss_cam_t0 + cfg.stage2.w_rot_loss_camera * loss_cam_r0
        total0 = (cfg.stage2.w_data_loss_motion * loss_motion0 +
                  cfg.stage2.w_data_loss_camera * loss_camera0)
        guru.info(
            f"[0000/{cfg.stage2.num_epochs}] init_total={float(total0):.6f} | "
            f"motion={float(loss_motion0):.6f} | camera={float(loss_camera0):.6f} | "
            f"cam_rot6d={float(loss_cam_r0):.6f} | cam_transL1={float(loss_cam_t0):.6f}"
        )

    opt = torch.optim.Adam([
        {"params": mb_model.parameters(), "lr": cfg.stage2.lr_motion_model},
        {"params": cam_model.parameters(), "lr": cfg.stage2.lr_camera_model},
    ])
    switch_epoch = max(1, int(cfg.stage2.rot_switch_pct * cfg.stage2.num_epochs))

    for epoch in tqdm(range(cfg.stage2.num_epochs), desc="Stage 2"):
        mb_model.train(); cam_model.train()
        opt.zero_grad(set_to_none=True)
        t_idx = torch.randint(0, T, (cfg.stage2.batch_size,), device=device)
        t_norm = normalize_time_idx_to_norm(t_idx, T).unsqueeze(-1)
        pred_bases = mb_model(t_norm)
        tgt_bases = disc_bases[:, t_idx]
        loss_motion = F.mse_loss(pred_bases, tgt_bases)
        pred_cam = cam_model(t_norm)
        tgt_cam  = disc_cam[t_idx]
        pred_r6, pred_t = pred_cam[..., :6], pred_cam[..., 6:]
        tgt_r6,  tgt_t  = tgt_cam[...,  :6], tgt_cam[...,  6:]
        loss_cam_t = F.l1_loss(pred_t, tgt_t)
        if epoch < switch_epoch:
            loss_cam_r = rot_loss_6d_cosine(pred_r6, tgt_r6)
        else:
            R_pred = cont_6d_to_rmat_safe(pred_r6)
            R_tgt  = cont_6d_to_rmat_safe(tgt_r6)
            loss_cam_r = rot_loss_geodesic_safe(R_pred, R_tgt)
        loss_camera = loss_cam_t + cfg.stage2.w_rot_loss_camera * loss_cam_r

        loss_ode = torch.zeros((), device=device)
        if cfg.stage2.w_ode > 0.0:
            T1 = max(T - 1, 1)
            tmin = -cfg.stage2.extrap_pct * T1
            tmax = (1.0 + cfg.stage2.extrap_pct) * T1
            t_ext = torch.rand(cfg.stage2.batch_size, 1, device=device) * (tmax - tmin) + tmin
            t_ext_norm = normalize_time_idx_to_norm(t_ext, T)
            h_norm = (cfg.stage2.ode_step_frames / T1) * 2.0
            d2_bases = _second_derivative_fd(mb_model.forward_extrap, t_ext_norm, h_norm)
            d2_bases_for_loss = d2_bases[..., 6:9]
            res_mb_B = d2_bases_for_loss.pow(2).mean(dim=(0, -1))
            d2_cam = _second_derivative_fd(cam_model.forward_extrap, t_ext_norm, h_norm)
            res_cam_trans_B = d2_cam[..., 6:9].pow(2).mean(dim=-1)
            if cfg.stage2.ode_on_rotation:
                dω_cam = _so3_angacc_from_rot6d(
                    lambda tn: cam_model.forward_extrap(tn)[..., :6], t_ext_norm, h_norm
                )
                res_cam_rot_B = (dω_cam ** 2).mean(dim=-1)
            else:
                res_cam_rot_B = torch.zeros_like(res_cam_trans_B)
            res_cam_B = res_cam_trans_B + res_cam_rot_B
            outside = ((t_ext < 0.0) | (t_ext > T1)).float().squeeze(-1)
            w = 1.0 + (cfg.stage2.ode_tail_boost - 1.0) * outside
            res_ode_B = 0.5 * (res_mb_B + res_cam_B)
            loss_ode = _weighted_mean(res_ode_B, w)

        base_total = (cfg.stage2.w_data_loss_motion * loss_motion +
                      cfg.stage2.w_data_loss_camera * loss_camera)
        if cfg.stage2.w_ode > 0.0:
            total = base_total + cfg.stage2.w_ode * loss_ode
        else:
            total = base_total
        total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=-1e6)

        total.backward()
        torch.nn.utils.clip_grad_norm_(list(mb_model.parameters()) + list(cam_model.parameters()), 1.0)
        opt.step()

        if (epoch+1) % cfg.stage2.log_every == 0 or epoch == cfg.stage2.num_epochs-1:
            guru.info(f"[{epoch+1}/{cfg.stage2.num_epochs}] "
                      f"Total={float(total):.6f} | motion={float(loss_motion):.6f} | "
                      f"camera={float(loss_camera):.6f} | ODE={float(loss_ode):.6f} | "
                      f"rot_mode={'6D' if epoch < switch_epoch else 'geo'}")

        if (epoch + 1) % cfg.stage2.val_plot_every == 0 or epoch == cfg.stage2.num_epochs - 1:
            eval_and_plot( # Pass cfg.stage2 here now
                mb_model, cam_model,
                disc_bases=disc_bases,
                disc_cam=disc_cam,
                cfg_s2=cfg.stage2, # <-- Pass stage2 config
                epoch=epoch,
                work_dir=cfg.work_dir,
                device=device,
            )

        s2_ckpt_dir = f"{cfg.work_dir}/checkpoints"
        os.makedirs(s2_ckpt_dir, exist_ok=True)
        s2_bases_ckpt = osp.join(s2_ckpt_dir, "continuous_motion_bases.ckpt")
        s2_cam_ckpt = osp.join(s2_ckpt_dir, "continuous_camera_pose.ckpt")

        if (epoch+1) % cfg.stage2.save_every == 0 or epoch == cfg.stage2.num_epochs-1:
            torch.save(mb_model.state_dict(), s2_bases_ckpt)
            torch.save(cam_model.state_dict(), s2_cam_ckpt)

    guru.info("Fitting done. Dumping dense samples for inspection...")
    mb_model.eval(); cam_model.eval()
    with torch.no_grad():
        u = torch.linspace(0, T-1, cfg.stage2.dense_samples, device=device).unsqueeze(-1)
        u_norm = normalize_time_idx_to_norm(u, T)
        bases_dense = mb_model(u_norm).permute(1,0,2).contiguous()
        cam_dense   = cam_model(u_norm)
    out_np = {
        "bases_dense": bases_dense.detach().cpu().numpy(),
        "cam_dense": cam_dense.detach().cpu().numpy(),
        "S": int(cfg.stage2.dense_samples),
        "K": int(K), "T": int(T),
        "degree": int(cfg.stage2.spline_degree),
        "num_control_points": int(cfg.stage2.num_control_points),
    }
    np.savez(osp.join(cfg.work_dir, "continuous_fits_dense.npz"), **out_np)
    guru.info(f"Saved: {s2_bases_ckpt}")
    guru.info(f"Saved: {s2_cam_ckpt}")
    guru.info(f"Saved dense samples: {cfg.work_dir}/continuous_fits_dense.npz")
    guru.info("Stage 2 finished.")


# ---initialize_and_checkpoint_model uses cfg.training.* ---
def initialize_and_checkpoint_model(
    cfg: TrainConfig, # Takes the main TrainConfig
    train_dataset: BaseDataset,
    device: torch.device,
    ckpt_path: str,
):
    if os.path.exists(ckpt_path):
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return

    fg_params, motion_bases, bg_params, tracks_3d = init_model_from_tracks(
        train_dataset,
        cfg.training.num_fg_points, # <-- Use cfg.training.*
        cfg.training.num_bg_points, # <-- Use cfg.training.*
        cfg.training.num_motion_bases, # <-- Use cfg.training.*
        vis=cfg.training.vis_debug_init, # <-- Use cfg.training.*
        port=cfg.training.vis_port, # <-- Use cfg.training.*
    )

    Ks = train_dataset.get_Ks().to(device)
    w2cs = train_dataset.get_w2cs().to(device)
    run_initial_optim(fg_params, motion_bases, tracks_3d, Ks, w2cs)

    if cfg.training.vis_debug_init and cfg.training.vis_port is not None: # <-- Use cfg.training.*
        server = get_server(port=cfg.training.vis_port) # <-- Use cfg.training.*
        vis_init_params(server, fg_params, motion_bases)

    camera_poses = init_trainable_poses(w2cs)

    model = SceneModel(
        Ks,
        w2cs,
        fg_params,
        motion_bases,
        camera_poses,
        bg_params,
        cfg.training.use_2dgs, # <-- Use cfg.training.*
    )

    guru.info(f"Saving initialization to {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "global_step": 0}, ckpt_path)


# --- init_model_from_tracks uses renamed parameters ---
def init_model_from_tracks(
    train_dataset,
    num_fg_points: int, # <-- Renamed param
    num_bg_points: int, # <-- Renamed param
    num_motion_bases: int, # <-- Keep name
    vis: bool = False,
    port: int | None = None,
):
    # Pass num_fg_points to get_tracks_3d
    tracks_3d = TrackObservations(*train_dataset.get_tracks_3d(num_fg_points))
    print(
        f"{tracks_3d.xyz.shape=} {tracks_3d.visibles.shape=} "
        f"{tracks_3d.invisibles.shape=} {tracks_3d.confidences.shape} "
        f"{tracks_3d.colors.shape}"
    )
    if not tracks_3d.check_sizes():
        import ipdb
        ipdb.set_trace()

    rot_type = "6d"
    cano_t = int(tracks_3d.visibles.sum(dim=0).argmax().item())

    # Use renamed params in log message
    guru.info(f"{cano_t=} {num_fg_points=} {num_bg_points=} {num_motion_bases=}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t, vis=vis, port=port
    )
    motion_bases = motion_bases.to(device)

    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs)
    fg_params = fg_params.to(device)

    bg_params = None
    if num_bg_points > 0: # <-- Use renamed param
        # Pass num_bg_points to get_bkgd_points
        bg_points = StaticObservations(*train_dataset.get_bkgd_points(num_bg_points))
        assert bg_points.check_sizes()
        bg_params = init_bg(bg_points)
        bg_params = bg_params.to(device)

    tracks_3d = tracks_3d.to(device)
    return fg_params, motion_bases, bg_params, tracks_3d


def backup_code(work_dir):
    root_dir = osp.abspath(osp.join(osp.dirname(__file__)))
    tracked_dirs = [osp.join(root_dir, dirname) for dirname in ["flow3d", "scripts"]]
    dst_dir = osp.join(work_dir, "code", datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    for tracked_dir in tracked_dirs:
        if osp.exists(tracked_dir):
            shutil.copytree(tracked_dir, osp.join(dst_dir, osp.basename(tracked_dir)))


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))