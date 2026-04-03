"""
FaceForensics++ (C23) Video to Image Dataset Processor

This script:
1. Extracts frames from videos using MTCNN with 1.15x margin
2. Resizes the face to 224x224
3. Splits the images into train/val/test sets with stratification
4. Saves the images into PyTorch-compatible directory structures

Output: 20,000 images total (10,000 real + 10,000 fake) split into train/val/test
"""

import cv2
import os
from pathlib import Path
from sklearn.externals.array_api_compat.numpy import False_
from tqdm import tqdm
import numpy as np
import torch
from facenet_pytorch import MTCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)


def extract_video_id(filename):
    """
    Extract the base (target) video ID from a filename.
    """

    stem = Path(filename).stem
    return stem.split('_')[0]   


def get_all_video_ids(base_dir):
    """
    Get all unique video IDs from the original videos directory.
    """

    original_dir = Path(base_dir) / "original"
    video_files = sorted(list(original_dir.glob("*.mp4")))
    video_ids = sorted(set([extract_video_id(vf.name) for vf in video_files]))
    return video_ids


def split_video_ids(video_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split video IDs into train/val/test sets SEQUENTIALLY.
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    video_ids_seq = sorted(video_ids.copy())
    
    total = len(video_ids_seq)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': video_ids_seq[:train_end],
        'val': video_ids_seq[train_end:val_end],
        'test': video_ids_seq[val_end:]
    }
    
    return splits


def crop_with_margin_and_resize(image, target_size=(224, 224), margin_percent=0.15):
    """
    Detect the largest face in image using MTCNN, crop with 1.15x margin, and resize to target_size.
    """

    h, w = image.shape[:2]
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes, probs = mtcnn.detect(image_rgb)
    
    if boxes is not None and len(boxes) > 0:
        x1, y1, x2, y2 = [int(b) for b in boxes[0]]
        
        face_w = x2 - x1
        face_h = y2 - y1
        
        margin_w = int(face_w * margin_percent)
        margin_h = int(face_h * margin_percent)
        
        x1 = x1 - margin_w
        y1 = y1 - margin_h
        x2 = x2 + margin_w
        y2 = y2 + margin_h
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        cropped_face = image[y1:y2, x1:x2]
        
        resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_LINEAR)
        
        return resized_face
    else:
        return None


def extract_frames_from_video(video_path, num_frames):
    """
    Extract frames from a video, only keeping frames with detected faces.
    """

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    frames = []
    
    frame_step = max(1, total_frames // (num_frames * 2))
    current_frame = 0
    
    max_attempts = total_frames
    attempts = 0
    
    while len(frames) < num_frames and current_frame < total_frames and attempts < max_attempts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        processed_face = crop_with_margin_and_resize(frame)
        
        if processed_face is not None:
            frames.append(processed_face)
        
        current_frame += frame_step
        attempts += 1
        
        if current_frame >= total_frames and len(frames) < num_frames and frame_step > 1:
            frame_step = max(1, frame_step // 2)
            current_frame = frame_step
    
    cap.release()
    return frames


def process_real_videos(base_dir, video_ids, output_dir, frames_per_video=10):
    """
    Process real (original) videos and extract frames.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    source_dir = Path(base_dir) / "original"
    frames_extracted = 0
    
    for video_id in tqdm(video_ids, desc="Processing real videos"):
        video_file = source_dir / f"{video_id}.mp4"
        
        if not video_file.exists():
            continue
            
        frames = extract_frames_from_video(video_file, frames_per_video)
        
        for frame_idx, frame in enumerate(frames):
            output_filename = f"real_{video_id}_frame{frame_idx:04d}.jpg"
            output_filepath = output_path / output_filename
            cv2.imwrite(str(output_filepath), frame)
            frames_extracted += 1
            
    return frames_extracted


def process_fake_videos(base_dir, video_ids, output_dir, target_fake_count, fake_methods):
    """
    Process fake videos, handling the target_source format.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_methods = len(fake_methods)
    frames_per_method = target_fake_count // num_methods
    remaining_frames = target_fake_count % num_methods
    
    total_fake_extracted = 0
    
    for method_idx, method in enumerate(fake_methods):
        method_dir = Path(base_dir) / method
        method_target = frames_per_method + (1 if method_idx < remaining_frames else 0)
        
        method_videos = []
        for video_id in video_ids:
            matching_videos = list(method_dir.glob(f"{video_id}_*.mp4"))
            if matching_videos:
                method_videos.append(matching_videos[0])
                
        if len(method_videos) == 0:
            print(f"Warning: No videos found for method {method}")
            continue
            
        frames_per_video = method_target // len(method_videos)
        remaining_method_frames = method_target % len(method_videos)
        
        method_extracted = 0
        for video_idx, video_file in enumerate(tqdm(method_videos, desc=f"Processing {method}", leave=False)):
            num_frames = frames_per_video + (1 if video_idx < remaining_method_frames else 0)
            
            if num_frames == 0:
                continue
                
            frames = extract_frames_from_video(video_file, num_frames)
            
            for frame_idx, frame in enumerate(frames):
                output_filename = f"fake_{method}_{video_file.stem}_frame{frame_idx:04d}.jpg"
                output_filepath = output_path / output_filename
                cv2.imwrite(str(output_filepath), frame)
                method_extracted += 1
                
        total_fake_extracted += method_extracted
        print(f"SUCCESS: {method}: {method_extracted} frames extracted")
        
    return total_fake_extracted


def main():
    print("=" * 80)
    print("FaceForensics++ (C23) Video to Image Dataset Processor")
    print("Sequential Split to Prevent Source-Face Leakage")
    print("=" * 80)
    
    SCRIPT_DIR = Path(__file__).parent.absolute()
    BASE_DIR = SCRIPT_DIR.parent / "raw_data" / "FaceForensics++_C23"
    OUTPUT_BASE_DIR = SCRIPT_DIR.parent / "data" / "image"
    
    print(f"\nInput directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    
    if not BASE_DIR.exists():
        print(f"\nERROR: Input directory does not exist: {BASE_DIR}")
        print("Please check the path and try again.")
        return
    TARGET_REAL_TOTAL = 10000
    TARGET_FAKE_TOTAL = 10000
    FRAMES_PER_REAL_VIDEO = 10
    
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    FAKE_METHODS = [
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
        "FaceShifter"
    ]
    
    print(f"\nMethods being processed: {FAKE_METHODS}")
    
    all_video_ids = get_all_video_ids(BASE_DIR)
    
    if len(all_video_ids) == 0:
        print(f"\nERROR: No video files found in {BASE_DIR / 'original'}")
        print("Please verify that the 'original' folder contains .mp4 files.")
        return
    
    print(f"Found {len(all_video_ids)} unique video IDs")
    
    splits = split_video_ids(all_video_ids, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    split_stats = {}
    for split_name, video_ids in splits.items():
        num_videos = len(video_ids)
        real_frames = num_videos * FRAMES_PER_REAL_VIDEO
        fake_frames = real_frames
        
        split_stats[split_name] = {
            'videos': num_videos,
            'real_frames': real_frames,
            'fake_frames': fake_frames
        }
    
    results = {}
    for split_name in ['train', 'val', 'test']:
        print(f"\n{'=' * 80}")
        print(f"Processing {split_name.upper()} Split")
        print(f"{'=' * 80}")
        
        video_ids = splits[split_name]
        split_output_dir = OUTPUT_BASE_DIR / split_name
        
        print(f"\n[1/2] Extracting REAL frames for {split_name}")
        real_output_dir = split_output_dir / "real"
        real_count = process_real_videos(BASE_DIR, video_ids, real_output_dir, FRAMES_PER_REAL_VIDEO)
        
        print(f"\n[2/2] Extracting FAKE frames for {split_name}")
        fake_output_dir = split_output_dir / "fake"
        fake_count = process_fake_videos(BASE_DIR, video_ids, fake_output_dir, split_stats[split_name]['fake_frames'], FAKE_METHODS)
        
        results[split_name] = {'real': real_count, 'fake': fake_count}
        
    print("\n" + "=" * 80)
    print("SUCCESS: Dataset processing complete! Sequential splits applied.")
    print("=" * 80)

if __name__ == "__main__":
    main()