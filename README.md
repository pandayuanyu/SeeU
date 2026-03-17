This repository is the official implementation of [SeeU](https://arxiv.org/abs/2512.03350).

> **[CVPR 2026] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation** <br>
> [Yu Yuan](https://yuyuan-space.github.io/), [Tharindu Wickremasinghe](https://tharindu-nirmal.github.io/), [Zeeshan Nadir](https://www.linkedin.com/in/zeeshan-nadir), [Xijun Wang](https://xijun-w.github.io/), [Yiheng Chi](https://engineering.purdue.edu/ChanGroup/people.html), [Stanley H. Chan](https://engineering.purdue.edu/ChanGroup/stanleychan.html)<br>

[![Arxiv](https://img.shields.io/badge/arXiv-2512.03350-b31b1b.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2512.03350)
[![Project Page](https://img.shields.io/badge/Project-Page-green?style=for-the-badge)](https://yuyuan-space.github.io/SeeU/)
[![SeeU45 Dataset](https://img.shields.io/badge/SeeU45%20-Dataset-FF4F1D.svg?style=for-the-badge&logo=Huggingface)](https://huggingface.co/datasets/pandaphd/SeeU45)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)

![SeeU Example](project_page/SeeU.gif)



## 🔥 Latest News!
* [Mar 17, 2026]: Release code.
* [Feb 21, 2026]: SeeU has been accepted by CVPR 2026!
* [Dec 3, 2025]: Paper available on [arXiv](https://arxiv.org/abs/2512.03350).
* [Dec 3, 2025]: Release [SeeU45 Dataset](https://huggingface.co/datasets/pandaphd/SeeU45).
* [Dec 1, 2025]: Release [Project Page](https://yuyuan-space.github.io/SeeU/).



## Quick Start

### 1. Environment
* CUDA 12.6, 64-bit Python 3.10 and PyTorch 2.6.0, other environments may also work
* Users can use the following commands to install the packages
```bash
conda create -n seeu python=3.10
conda activate seeu
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

git clone https://github.com/pandayuanyu/SeeU.git
cd SeeU
pip install -r requirements.txt
```

### 2. Download Pre-Processed Dataset
Please download our pre-processed SeeU45 Dataset from [Hugging Face](https://huggingface.co/datasets/pandaphd/SeeU45_PreProcessed/tree/main), and put it under folder `preproc`
```bash
pip install "huggingface_hub[hf_transfer]"

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download pandaphd/SeeU45_PreProcessed --local-dir preproc
```


### 3. Training (2D-->Discrete 4D-->Continuous 4D)
We use the `butterfly` scene as a demonstration to complete the two-stage training process:
(1) from 2D to discrete 4D, and
(2) from discrete 4D to continuous 4D.

```bash
python train.py --work-dir /path-to-work-dir/butterfly/  data:custom  --data.data-dir /path-to-preprocessed-data-folder/ --data.scene butterfly --data.depth-type megasam_depth
```


### 4. Inference: Projection to 2D

#### 4.1 Unseen Time (Temporal Rendering)
The following script demonstrates how to render projected frames and inpainting masks from the trained continuous 4D representation, given a predefined temporal configuration.

```bash 
python inference_video_lapse.py --work-dir /path-to-work-dir/butterfly/ --fps 15 --port 5005 --data.data-dir /path-to-preprocessed-data-folder/ --data.scene butterfly --camera.mode continuous  --gt.gt-dir /path/SeeU/dataset/SeeU45_GT/butterfly/
```

* You can modify the temporal setup in `inference_video_lapse.py` under:
```bash 
class VideoConfig:
    num_past_extrap_frames: int = 5 # number of extrapolated frames in the past
    num_replay_frames: int = 71 # number of in-between frames
    num_future_extrap_frames: int = 5 # number of extrapolated future frames
    # We recommend using 81 total frames for final experiments, to stay consistent with the following video inpainting stage.
```

* If ground-truth frames are not available, you can safely remove:
```bash 
--gt.gt-dir /path/SeeU/dataset/SeeU45_GT/butterfly/
```

#### 4.2 Unseen Space (Camera Trajectory Control)
The following script demonstrates rendering projected frames and inpainting masks from the trained continuous 4D representation, under different camera trajectories.
```bash 
python inference_video_lapse.py --work-dir /path-to-work-dir/butterfly/ --fps 15 --port 5005 --data.data-dir path-to-preprocessed-data-folder/ --data.scene butterfly --camera.mode reference --camera.traj dolly-right  --gt.gt-dir /path/SeeU/dataset/SeeU45_GT/butterfly/
```
* Supported camera trajectories via `--camera.traj`, current options include: `fixed` `tilt-up` `pan-right` `dolly-up` `dolly-right` `dolly-out`
* You can extend or customize trajectories by modifying `flow3d/trajectories_4D_motion.py`
* If ground-truth frames are not available, you can safely remove:
```bash 
--gt.gt-dir /path/SeeU/dataset/SeeU45_GT/butterfly/
```


#### 4.3 4D Dynamics/Tracks Visualization
We also provide additional scripts for visualizing 4D tracks.
```bash 
python render_tracks.py --work-dir /path-to-work-dir/butterfly/ --data.data-dir path-to-preprocessed-data-folder/ --data.scene butterfly
```
* You can modify the visualization setup in `render_tracks.py` under:
```bash 

class VideoConfig:
    # View 2 Config
    view2: CameraView2Config = field(default_factory=CameraView2Config)

    # tracks selection
    grid_selection_step: int = 80 # Sampling density (e.g., visualize one track per N 3D Gaussians)
    opacity_selection_threshold: float = 0.8 # Opacity threshold for filtering tracks
```


### 5. Inference: Video Inpainting
Once the projected frames (`src_video.mp4`) and inpainting masks (`src_mask.mp4`) are obtained from above Section 4.1 or 4.2, we perform in-context video inpainting based on a given text prompt using [VACE](https://ali-vilab.github.io/VACE-Page/).

Install VACE first (the environment is already integrated, so there is no need to create a separate environment for VACE). Then run the following scripts:
```bash
cd /path-to/VACE/
# inpainting inference
python vace/vace_wan_inference.py --src_video /path/SeeU/SeeU45_out/butterfly/videos/2026-xx-xx-xxxxxx_0p_81r_0f_Cam_Ref_dolly-out/src_video.mp4 --src_mask /path/SeeU/SeeU45_out/butterfly/videos/2026-xx-xx-xxxxxx_0p_81r_0f_Cam_Ref_dolly-out/src_mask.mp4 --prompt "A close-up video of a butterfly flapping its wings. The background has stones and gravel. Restore the masked regions of the video with the backgroound of the stones. Make the colors and background behind the butterfly realistic and continuous."
```




## Custom Data Pre-Processing
<details>
<summary>Click to expand</summary>

### 1. Overview
Our data preprocessing pipeline is primarily adapted from [Shape-of-Motion](https://github.com/vye16/shape-of-motion/blob/main/preproc/README.md), with few modifications.

The difference is that we additionally leverage [MegaSaM](https://github.com/mega-sam/mega-sam)
 to extract more robust camera parameters and depth maps, which improves the overall stability and quality of downstream 4D reconstruction.

### 2. PreProcessing Pipeline
```bash
# 1. create new venv   

# 2. 
cd /home/.../SeeU/preproc/

# 3. install dependencies
./setup_dependencies.sh

# 4. get foreground masks by SAM
python mask_app.py --root_dir /home/.../SeeU/preproc/data/

# 5. run
python process_custom.py --img-dirs /home/.../SeeU/preproc/data/images/** --gpus 0

# 6. install MegaSaM and 
cd /home/.../SeeU/preproc/mega-sam/

# 7. modeify the setting of the following scripts and run
bash ./mono_depth_scripts/run_mono-depth_demo.sh
bash ./tools/evaluate_demo.sh
bash ./cvd_opt/cvd_opt_demo.sh

# 8. After preprocessing, please follow the directory structure of [SeeU45_PreProcessed](https://huggingface.co/datasets/pandaphd/SeeU45_PreProcessed/tree/main) to organize your data.
```

### 3. Additional Notes

* We recommend creating a separate environment for data preprocessing, as it involves running multiple third-party models. Our base environment (compatible with all preprocessing models): CUDA 12.1, Python 3.10, PyTorch 2.2.0, torchvision 0.17.0, transformers 4.57.0, xformers 0.0.24.

* For efficient training, we suggest using custom datasets with 10–40 frames, where the data contains clear foreground motion.



</details>



## Evaluation
Please check the code is in the `comp_metrics` folder.


## Disclaimer
This project is released for academic use. We disclaim responsibility for user-generated content. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for, users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards. 


## ❣️ Acknowledgement
We thank [Shape-of-Motion](https://shape-of-motion.github.io/), [MegaSaM](https://github.com/mega-sam/mega-sam), and [VACE](https://ali-vilab.github.io/VACE-Page/) for their amazing jobs.


## 🌟 Citation
If you feel this project helpful/insightful, please cite our paper:
```bibtex
@article{Yuan_2025_SeeU,
  title={{SeeU}: Seeing the Unseen World via 4D Dynamics-aware Generation},
  author={Yuan, Yu and Wickremasinghe, Tharindu and Nadir, Zeeshan and Wang, Xijun and Chi, Yiheng and Chan, Stanley H.,
  journal={arXiv preprint arXiv: 2512.03350},
  year={2025}
}
```

## ✉️ Contact
If you have any questions or comments, feel free to contact me through email (mryuanyu@outlook.com). Suggestions and collaborations are also highly welcome!
