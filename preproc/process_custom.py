import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import tyro


def main(
    img_dirs: list[str],
    gpus: list[int],
    img_name: str = "images",
    mask_name: str = "masks",
    metric_depth_name: str = "unidepth_disp",
    intrins_name: str = "unidepth_intrins",
    mono_depth_model: str = "depth-anything",
    track_model: str = "bootstapir",
    tapir_torch: bool = True,
    # --- New parameter: name for MegaSAM output model ---
    camera_depth_model: str = "megasam-cam-depth",
):
    """
    Main function for parallel preprocessing of multiple video sequences.
    """
    # Fix validation logic: check if 'images' exists in the path instead of at the end
    if len(img_dirs) > 0 and img_name not in img_dirs[0]:
        raise ValueError(f"Expecting '{img_name}' to be in the path, but got: {img_dirs[0]}")

    mono_depth_name = mono_depth_model.replace("-", "_")
    with ProcessPoolExecutor(max_workers=len(gpus)) as exc:
        for i, img_dir in enumerate(img_dirs):
            gpu = gpus[i % len(gpus)]
            img_dir = img_dir.rstrip("/")
            
            # Infer base_dir and scene_name from img_dir
            scene_name = Path(img_dir).name
            base_dir = str(Path(img_dir).parent.parent)

            exc.submit(
                process_sequence,
                gpu,
                base_dir,
                scene_name,
                img_name,
                mask_name,
                metric_depth_name,
                intrins_name,
                mono_depth_name,
                track_model,
                tapir_torch,
                camera_depth_model,
            )


def process_sequence(
    gpu: int,
    base_dir: str,
    scene_name: str,
    img_folder_name: str,
    mask_folder_name: str,
    metric_depth_folder_name: str,
    intrins_folder_name: str,
    mono_depth_folder_name: str,
    track_model: str,
    tapir_torch: bool,
    camera_depth_model: str,
):
    """
    Execute all independent preprocessing steps for a single video sequence.
    """
    dev_arg = f"CUDA_VISIBLE_DEVICES={gpu}"
    
    # Construct all required paths
    img_dir = f"{base_dir}/{img_folder_name}/{scene_name}"
    mask_dir = f"{base_dir}/{mask_folder_name}/{scene_name}"
    metric_depth_dir = f"{base_dir}/{metric_depth_folder_name}/{scene_name}"
    # Fix intrins_name path construction: it is a file prefix, not a folder
    intrins_path_prefix = f"{base_dir}/{intrins_folder_name}/{scene_name}"
    mono_depth_dir = f"{base_dir}/{mono_depth_folder_name}/{scene_name}"
    aligned_depth_dir = f"{base_dir}/aligned_{mono_depth_folder_name}/{scene_name}"
    track_dir = f"{base_dir}/{track_model}/{scene_name}"
    
    print(f"\n--- Start processing scene: {scene_name} on GPU: {gpu} ---")

    # ==========================================================
    # --- Pipeline A: UniDepth -> Depth Anything -> TAPIR ---
    # ==========================================================
    print(f"[{scene_name}] Starting Pipeline A...")

    # Step A-1: Compute metric depth (UniDepth)
    print(f"[{scene_name}] -> Step A-1: Running metric depth estimation...")
    metric_depth_cmd = (
        f"{dev_arg} python compute_metric_depth.py --img-dir {img_dir} "
        f"--depth-dir {metric_depth_dir} --intrins-file {intrins_path_prefix}.json"
    )
    subprocess.run(metric_depth_cmd, shell=True, check=True, executable="/bin/bash")

    # Step A-2: Compute and align monocular depth (Depth Anything)
    print(f"[{scene_name}] -> Step A-2: Running monocular depth estimation...")
    mono_depth_cmd = (
        f"{dev_arg} python compute_depth.py --img_dir {img_dir} "
        f"--out_raw_dir {mono_depth_dir} --out_aligned_dir {aligned_depth_dir} "
        f"--model {mono_depth_folder_name.replace('_', '-')} --metric_dir {metric_depth_dir}"
    )
    subprocess.run(mono_depth_cmd, shell=True, check=True, executable="/bin/bash")

    # Step A-3: Compute 2D tracks (TAPIR)
    print(f"[{scene_name}] -> Step A-3: Running 2D tracking...")
    track_script = "compute_tracks_torch.py" if tapir_torch else "compute_tracks_jax.py"
    track_cmd = (
        f"{dev_arg} python {track_script} --image_dir {img_dir} "
        f"--mask_dir {mask_dir} --out_dir {track_dir} --model_type {track_model}"
    )
    subprocess.run(track_cmd, shell=True, check=True, executable="/bin/bash")

    

    print(f"--- Scene {scene_name} processing completed ---")


if __name__ == "__main__":
    tyro.cli(main)