import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState

from flow3d.scene_model import SceneModel
from flow3d.vis.utils import draw_tracks_2d_th, get_server
from flow3d.vis.viewer import DynamicViewer


class Renderer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        # Logging.
        work_dir: str,
        port: int | None = None,
    ):
        self.device = device

        self.model = model
        self.num_frames = model.num_frames

        self.work_dir = work_dir
        self.global_step = 0
        self.epoch = 0

        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server, self.render_fn, model.num_frames, work_dir, mode="rendering"
            )


        self.tracks_3d = self.model.compute_poses_fg(
            #  torch.arange(max(0, t - 20), max(1, t), device=self.device),
            torch.arange(self.num_frames, device=self.device),
            inds=torch.arange(10, device=self.device),
        )[0]

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, use_2dgs: bool = False, *args, **kwargs
    ) -> "Renderer":
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, weights_only=False)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model.use_2dgs = use_2dgs
        model = model.to(device)
        print(f"num gs: {model.fg.num_gaussians + model.bg.num_gaussians}")
        renderer = Renderer(model, device, *args, **kwargs)
        renderer.global_step = ckpt.get("global_step", 0)
        renderer.epoch = ckpt.get("epoch", 0)
        return renderer

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        # If the viewer is not ready, return a white image to avoid crashing
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh

        # Construct camera intrinsics K based on current interactive camera state
        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0],
             [0.0,  focal, H / 2.0],
             [0.0,  0.0,   1.0]],
            device=self.device,
        )

        # world-to-camera
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )

        # Current rendering time index:
        # canonical mode -> t=None
        # playback mode  -> t=frame index
        if not self.viewer._canonical_checkbox.value:
            t = int(self.viewer._playback_guis[0].value)
        else:
            t = None

        # First render RGB
        self.model.training = False
        out = self.model.render(t, w2c[None], K[None], img_wh)
        img = out["img"][0]  # torch, [H, W, 3], 0~1

        # (1) If viewer does not have this checkbox, return RGB directly
        if not hasattr(self.viewer, "_render_track_checkbox"):
            return (img.cpu().numpy() * 255.0).astype(np.uint8)

        # (2) Checkbox exists but not enabled -> return RGB
        if not self.viewer._render_track_checkbox.value:
            return (img.cpu().numpy() * 255.0).astype(np.uint8)

        # (3) Checkbox enabled but in canonical mode (t=None)
        #     No specific time frame -> cannot project temporal tracks
        if t is None:
            return (img.cpu().numpy() * 255.0).astype(np.uint8)

        # (4) Checkbox enabled and t is valid
        #     Now render tracks; use try/except to avoid crashing
        try:
            if self.tracks_3d is None:
                # If you want lazy computation instead of precomputing in __init__, do it here
                num_show = min(10, self.model.fg.num_gaussians)
                ts_full = torch.arange(self.num_frames, device=self.device)
                inds = torch.arange(num_show, device=self.device)
                self.tracks_3d = self.model.compute_poses_fg(ts_full, inds=inds)[0]
                # self.tracks_3d: [num_show, T, 3]

            # Take last 20 frames of trajectory and project to current camera
            t0 = max(0, t - 20)
            t1 = max(1, t)
            tracks_3d = self.tracks_3d[:, t0:t1]  # [N, L, 3]

            tracks_3d_h = F.pad(tracks_3d, (0, 1), value=1.0)  # -> [N, L, 4] homogeneous
            tracks_2d = torch.einsum(
                "ij,jk,nlk->nli",   # (K)(w2c)(xyz1)
                K,                  # [3,3]
                w2c[:3],            # [3,4]
                tracks_3d_h,        # [N,L,4]
            )
            tracks_2d = tracks_2d[..., :2] / tracks_2d[..., 2:]

            img = draw_tracks_2d_th(img, tracks_2d)
            return img  # draw_tracks_2d_th usually returns uint8 or torch depending on implementation
        except Exception as e:
            guru.warning(f"[render_fn] track overlay failed: {e}")
            return (img.cpu().numpy() * 255.0).astype(np.uint8)



    @torch.inference_mode()
    def render_fn_old(self, camera_state: CameraState, img_wh: tuple[int, int]):
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = (
            int(self.viewer._playback_guis[0].value)
            if not self.viewer._canonical_checkbox.value
            else None
        )
        self.model.training = False
        img = self.model.render(t, w2c[None], K[None], img_wh)["img"][0]
        if not self.viewer._render_track_checkbox.value:
            img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        else:
            assert t is not None
            tracks_3d = self.tracks_3d[:, max(0, t - 20) : max(1, t)]
            tracks_2d = torch.einsum(
                "ij,jk,nbk->nbi", K, w2c[:3], F.pad(tracks_3d, (0, 1), value=1.0)
            )
            tracks_2d = tracks_2d[..., :2] / tracks_2d[..., 2:]
            img = draw_tracks_2d_th(img, tracks_2d)
        return img