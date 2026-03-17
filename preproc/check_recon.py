


import numpy as np
import matplotlib.pyplot as plt
import tyro
from pathlib import Path

def main(recon_path: str, output_path: str = "reconstruction.png"):
    """
    加载、打印并可视化 DROID-SLAM 的输出结果 (.npy 文件)，然后将图像直接保存到文件。
    """
    print(f"正在加载重建文件: {recon_path}")
    try:
        recon = np.load(recon_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"错误: 文件未找到! 请检查路径: {recon_path}")
        return

    # --- 提取核心数据 ---
    if 'traj_c2w' not in recon:
        print("错误: 在.npy文件中找不到 'traj_c2w' (相机轨迹)!")
        return
    camera_poses_c2w = recon['traj_c2w']
    camera_positions = camera_poses_c2w[:, :3, 3]
    points = recon.get('points')
    intrinsics = recon.get('intrinsics')
    img_shape = recon.get('img_shape')

    # vvvvvvvvvvvv 新增的打印部分 vvvvvvvvvvvvv
    print("\n" + "="*20 + " 从 .npy 文件中读取的相机参数 " + "="*20)

    # 1. 打印相机内参
    if intrinsics is not None:
        print("\n## 相机内参 (fx, fy, cx, cy):")
        print(intrinsics)
    else:
        print("\n## 相机内参: 未在 .npy 文件中找到。")

    # 2. 打印图像尺寸
    if img_shape is not None:
        print("\n## 图像尺寸 (高度, 宽度):")
        print(img_shape)
    else:
        print("\n## 图像尺寸: 未在 .npy 文件中找到。")
        
    # 3. 打印几个示例的相机位姿 (外参)
    num_poses = len(camera_poses_c2w)
    mid_index = num_poses // 2
    
    print("\n## 示例相机位姿 (Camera-to-World 4x4 矩阵):")
    print("\n--- 第一帧 (Frame 0) 的位姿 ---")
    print(camera_poses_c2w[0])
    
    print(f"\n--- 中间帧 (Frame {mid_index}) 的位姿 ---")
    print(camera_poses_c2w[mid_index])
    
    print(f"\n--- 最后一帧 (Frame {num_poses - 1}) 的位姿 ---")
    print(camera_poses_c2w[-1])

    print("\n" + "="*20 + " 参数打印结束 " + "="*20 + "\n")
    # ^^^^^^^^^^^^ 新增的打印部分结束 ^^^^^^^^^^^^

    # --- 核心修改：重塑'points'数组 ---
    if points is not None:
        points = points.reshape(-1, 3)
    
    # --- 开始绘图 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if points is not None and len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='gray', alpha=0.2, label=f'3D Point Cloud ({len(points)} points)')
        print(f"成功加载并绘制 {len(points)} 个三维点。")
    else:
        print("警告: 未找到或点云为空。")

    ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
            marker='o', markersize=3, linestyle='-', label='Camera Trajectory')
    
    ax.scatter(camera_positions[0, 0], camera_positions[0, 1], camera_positions[0, 2], c='green', s=50, label='Start')
    ax.scatter(camera_positions[-1, 0], camera_positions[-1, 1], camera_positions[-1, 2], c='red', s=50, label='End')
    
    print(f"成功加载并绘制 {len(camera_positions)} 个相机位姿。")

    # --- 设置图表样式 ---
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('DROID-SLAM Reconstruction Visualization')
    ax.legend()
    
    all_positions = camera_positions
    if points is not None and len(points) > 0:
        all_positions = np.concatenate([all_positions, points], axis=0)
    
    max_range = (all_positions.max(axis=0) - all_positions.min(axis=0)).max() / 2.0
    if max_range < 1e-3: max_range = 1.0
    
    mid = (all_positions.max(axis=0) + all_positions.min(axis=0)) * 0.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.savefig(output_path)
    print(f"绘图已成功保存到: {Path(output_path).resolve()}")

if __name__ == "__main__":
    tyro.cli(main)