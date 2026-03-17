from dataclasses import dataclass


@dataclass
class FGLRConfig:
    means: float = 1.6e-4
    opacities: float = 1e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2
    motion_coefs: float = 1e-2


@dataclass
class BGLRConfig:
    means: float = 1.6e-4
    opacities: float = 5e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2


@dataclass
class MotionLRConfig:
    rots: float = 1.6e-4
    transls: float = 1.6e-4

@dataclass
class CameraScalesLRConfig:
    camera_scales: float = 1e-4

@dataclass
class CameraPoseLRConfig:
    Rs: float = 1e-3
    ts: float = 1e-3

@dataclass
class SceneLRConfig:
    fg: FGLRConfig
    bg: BGLRConfig
    motion_bases: MotionLRConfig
    camera_poses: CameraPoseLRConfig
    camera_scales: CameraScalesLRConfig



@dataclass
class LossesConfig:
    w_rgb: float = 1.0
    w_depth_reg: float = 0.5
    w_depth_const: float = 0.1
    w_depth_grad: float = 1
    w_track: float = 2.0
    w_mask: float = 1.0
    w_smooth_bases: float = 1.0
    w_smooth_tracks: float = 2.0
    w_scale_var: float = 0.01
    w_z_accel: float = 1.0


@dataclass
class OptimizerConfig:
    max_steps: int = 5000
    ## Adaptive gaussian control
    warmup_steps: int = 200
    control_every: int = 100
    reset_opacity_every_n_controls: int = 30
    stop_control_by_screen_steps: int = 4000
    stop_control_steps: int = 4000
    ### Densify.
    densify_xys_grad_threshold: float = 0.0002
    densify_scale_threshold: float = 0.01
    densify_screen_threshold: float = 0.05
    stop_densify_steps: int = 15000
    ### Cull.
    cull_opacity_threshold: float = 0.1
    cull_scale_threshold: float = 0.5
    cull_screen_threshold: float = 0.15

# --- Stage 1 General Config ---
# Renamed from GeneralTrainingConfig, removed _s1 suffixes
@dataclass
class GeneralTrainingConfig:
    """General configuration for training setup, mostly for Stage 1."""
    num_epochs_s1: int = 2000         # Renamed from num_epochs for clarity (Stage 1)
    num_fg_points: int = 80000       # Renamed from num_fg
    num_bg_points: int = 80000       # Renamed from num_bg
    num_motion_bases: int = 10
    batch_size_s1: int = 32           # Renamed from batch_size for clarity (Stage 1)
    num_dl_workers: int = 4
    validate_every_s1: int = 100       # Renamed from validate_every
    save_videos_every_s1: int = 100    # Renamed from save_videos_every
    use_2dgs: bool = False
    # Visualization
    vis_port: int | None = None       # Renamed from port
    vis_debug_init: bool = False      # Renamed from vis_debug


@dataclass
class Stage2Config:
    """Configuration for the second stage of training (fitting continuous motion)."""
    # General
    num_epochs: int = 1000       # Stage 2 total epochs (example value)
    batch_size: int = 64        # Batch size for Stage 2 sampling

    # B-spline Parameters
    num_control_points: int = 8  # Number of control points (m)
    spline_degree: int = 3       # Spline degree (p=3 -> cubic)

    # Learning Rates
    lr_motion_model: float = 1e-4 # Motion bases model LR
    lr_camera_model: float = 1e-5 # Camera pose model LR

    # Data Fitting Loss Weights
    w_data_loss_motion: float = 10.0 # Weight for fitting discrete B(t)
    w_data_loss_camera: float = 10.0 # Weight for fitting discrete w2c(t)
    w_rot_loss_camera: float = 5.0  # Specific weight for camera rotation loss component

    # Rotation Loss Switching
    rot_switch_pct: float = 0.2     # Percentage of epochs to use 6D cosine loss before switching

    # Physics (ODE) Prior Loss Weights & Params
    w_ode: float = 1e-5              # Weight for the ODE (y'' ~ 0) prior loss
    extrap_pct: float = 0.20         # Percentage to extrapolate time for ODE loss (e.g., 0.1 = +/- 10%)
    ode_step_frames: float = 1.0     # Finite difference step size (in frames) for derivatives
    ode_tail_boost: float = 0.1      # Multiplier for ODE loss weight in extrapolated regions
    ode_on_rotation: bool = True     # Apply ODE loss to rotation (SO(3) angular acceleration)



    # Logging, Saving, Plotting
    log_every: int = 50              # Log loss frequency
    save_every: int = 200            # Save checkpoint frequency
    val_plot_every: int = 200        # Generate validation plot frequency
    dense_samples: int = 300         # Number of samples for final dense output (.npz)
    plot_top_k_bases: int = 3        # Max number of motion bases to include in plots
    plot_3d_translation: bool = False # Whether to generate 3D trajectory plots

