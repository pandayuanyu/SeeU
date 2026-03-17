import os
import torch
import clip
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import cv2  


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)


def load_and_preprocess_frames_mp4(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: can not open {video_path}")
        return None
    
    frames = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break 
            
            # 1. Convert OpenCV BGR format to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Convert NumPy array to PIL image (preprocess requires PIL input)
            pil_image = Image.fromarray(rgb_frame)
            
            processed_frame = preprocess(pil_image)
            frames.append(processed_frame)
            
    finally:
        cap.release()
    
    if not frames:
        print(f"Error: no frames found in {video_path}")
        return None
        
    return torch.stack(frames).to(device)


@torch.no_grad()  
def get_average_video_features(video_path, batch_size=32):
    """
    Encode all frames of a video and return a single normalized average feature vector.
    """

    frame_batch = load_and_preprocess_frames_mp4(video_path)
    if frame_batch is None:
        return None
    
    all_features = []
    

    for i in tqdm(range(0, len(frame_batch), batch_size), desc=f"Encoding {os.path.basename(video_path)}"):
        batch = frame_batch[i:i+batch_size]
        
        features = model.encode_image(batch)
        all_features.append(features)
    
    if not all_features:
        return None
        
    video_features_tensor = torch.cat(all_features)
    
    average_features = torch.mean(video_features_tensor, dim=0, keepdim=True)
    
    average_features = F.normalize(average_features, p=2, dim=1)
    
    return average_features


def compute_video_to_video_similarity(video1_path, video2_path):
    """
    Compute CLIP similarity score between two MP4 videos.
    """
    print(f"Processing Video 1: {video1_path}")
    features_v1 = get_average_video_features(video1_path)
    
    print(f"Processing Video 2: {video2_path}")
    features_v2 = get_average_video_features(video2_path)
    
    if features_v1 is None or features_v2 is None:
        print("Error: failed to process one or both videos.")
        return None
        
    # 5. Compute cosine similarity between the two average feature vectors
    # features_v1 and features_v2 both have shape (1, D)
    similarity_score = F.cosine_similarity(features_v1, features_v2)
    
    return similarity_score.item()

# --- 3. Main execution ---
if __name__ == "__main__":
    
    # !! Important:
    # !! Please update the paths below to your actual MP4 files
    # !! ---------------------------------------------------
    video1_path = '/home/.../videos1.mp4'
    video2_path = '/home/.../video0.mp4'
    # !! ---------------------------------------------------


    # Check if files exist
    if not os.path.exists(video1_path):
        print(f"Error: file not found {video1_path}")
    elif not os.path.exists(video2_path):
         print(f"Error: file not found {video2_path}")
    else:
        # Compute and print score
        score = compute_video_to_video_similarity(video1_path, video2_path)
        if score is not None:
            print("\n" + "---"*10)
            print(f"Video-to-video CLIP similarity: {score:.4f}")
            print("---"*10)