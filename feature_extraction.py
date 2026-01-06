"""
Feature Extraction Script for Shoplifting Detection
Uses YOLO-Pose to extract pose keypoints from video frames
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import pickle
from tqdm import tqdm
import torch

# Configuration
DATASET_PATH = "Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad/Dataset"
OUTPUT_PATH = "extracted_features"
SEQUENCE_LENGTH = 30  # Number of frames per sequence
NUM_KEYPOINTS = 17    # COCO format has 17 keypoints
FRAME_SKIP = 2        # Process every nth frame for efficiency

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_yolo_pose_model():
    """Load YOLO-Pose model for keypoint detection"""
    # Use YOLO-Pose model (yolov8n-pose for efficiency)
    model = YOLO("yolov8n-pose.pt")
    
    # Check for MPS (Apple Silicon) or CUDA
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    return model, device

def extract_keypoints_from_frame(model, frame, device):
    """
    Extract pose keypoints from a single frame
    Returns: Array of shape (17, 3) for x, y, confidence of each keypoint
    """
    results = model(frame, device=device, verbose=False)
    
    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        # Get keypoints for the first detected person (most confident)
        keypoints = results[0].keypoints.data[0].cpu().numpy()  # Shape: (17, 3)
        return keypoints
    else:
        # Return zeros if no person detected
        return np.zeros((NUM_KEYPOINTS, 3))

def normalize_keypoints(keypoints, frame_width, frame_height):
    """
    Normalize keypoint coordinates to [0, 1] range
    Also center the skeleton around the hip center for translation invariance
    """
    normalized = keypoints.copy()
    
    # Normalize x, y coordinates
    normalized[:, 0] = normalized[:, 0] / frame_width
    normalized[:, 1] = normalized[:, 1] / frame_height
    
    # Center around hip center (keypoint 11 and 12 are left and right hip)
    if normalized[11, 2] > 0.3 and normalized[12, 2] > 0.3:
        hip_center_x = (normalized[11, 0] + normalized[12, 0]) / 2
        hip_center_y = (normalized[11, 1] + normalized[12, 1]) / 2
        normalized[:, 0] = normalized[:, 0] - hip_center_x + 0.5
        normalized[:, 1] = normalized[:, 1] - hip_center_y + 0.5
    
    return normalized

def extract_features_from_video(video_path, model, device):
    """
    Extract pose keypoints sequence from a video
    Returns: List of keypoint arrays, one per sampled frame
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    keypoints_sequence = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for efficiency
        if frame_idx % FRAME_SKIP == 0:
            keypoints = extract_keypoints_from_frame(model, frame, device)
            normalized_kp = normalize_keypoints(keypoints, frame_width, frame_height)
            keypoints_sequence.append(normalized_kp)
        
        frame_idx += 1
    
    cap.release()
    return keypoints_sequence

def create_sequences(keypoints_list, sequence_length=SEQUENCE_LENGTH):
    """
    Create fixed-length sequences from variable-length keypoint sequences
    Uses sliding window with overlap for data augmentation
    """
    sequences = []
    
    if len(keypoints_list) < sequence_length:
        # Pad with zeros if video is too short
        padded = keypoints_list + [np.zeros((NUM_KEYPOINTS, 3))] * (sequence_length - len(keypoints_list))
        sequences.append(np.array(padded[:sequence_length]))
    else:
        # Sliding window with 50% overlap
        stride = sequence_length // 2
        for i in range(0, len(keypoints_list) - sequence_length + 1, stride):
            seq = keypoints_list[i:i + sequence_length]
            sequences.append(np.array(seq))
    
    return sequences

def process_dataset():
    """
    Process all videos in the dataset and extract features
    """
    print("Loading YOLO-Pose model...")
    model, device = load_yolo_pose_model()
    
    all_features = []
    all_labels = []
    
    # Process Normal videos (label = 0)
    normal_path = os.path.join(DATASET_PATH, "Normal")
    normal_videos = [f for f in os.listdir(normal_path) if f.endswith('.mp4')]
    
    print(f"\nProcessing {len(normal_videos)} Normal videos...")
    for video_name in tqdm(normal_videos, desc="Normal videos"):
        video_path = os.path.join(normal_path, video_name)
        keypoints_seq = extract_features_from_video(video_path, model, device)
        
        if keypoints_seq is not None and len(keypoints_seq) > 0:
            sequences = create_sequences(keypoints_seq)
            for seq in sequences:
                all_features.append(seq)
                all_labels.append(0)  # Normal = 0
    
    # Process Shoplifting videos (label = 1)
    shoplifting_path = os.path.join(DATASET_PATH, "Shoplifting")
    shoplifting_videos = [f for f in os.listdir(shoplifting_path) if f.endswith('.mp4')]
    
    print(f"\nProcessing {len(shoplifting_videos)} Shoplifting videos...")
    for video_name in tqdm(shoplifting_videos, desc="Shoplifting videos"):
        video_path = os.path.join(shoplifting_path, video_name)
        keypoints_seq = extract_features_from_video(video_path, model, device)
        
        if keypoints_seq is not None and len(keypoints_seq) > 0:
            sequences = create_sequences(keypoints_seq)
            for seq in sequences:
                all_features.append(seq)
                all_labels.append(1)  # Shoplifting = 1
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n{'='*50}")
    print(f"Feature extraction complete!")
    print(f"Total sequences: {len(X)}")
    print(f"Feature shape: {X.shape}")  # (num_samples, sequence_length, num_keypoints, 3)
    print(f"Normal sequences: {np.sum(y == 0)}")
    print(f"Shoplifting sequences: {np.sum(y == 1)}")
    
    # Save features
    feature_file = os.path.join(OUTPUT_PATH, "pose_features.pkl")
    with open(feature_file, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
    
    print(f"\nFeatures saved to: {feature_file}")
    
    return X, y

if __name__ == "__main__":
    X, y = process_dataset()
