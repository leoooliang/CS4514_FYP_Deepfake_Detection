"""
Process ArtiFact dataset into precomputed .pt tensors for the Image Detection Module.

This script:
1. Crops the largest face in the image using MTCNN with 1.15x margin
2. Resizes the face to 224x224
3. Splits the images into train/val/test sets with stratification
4. Saves the images into PyTorch-compatible directory structures

Output: 20,000 images total (10,000 real + 10,000 fake) split into train/val/test
"""


import cv2
import os
import random
from pathlib import Path
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)

SCRIPT_DIR = Path(__file__).parent.absolute()
BASE_DIR = SCRIPT_DIR.parent / "raw_data" / "ArtiFact"
OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "image"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET_COUNTS = {
    'real': {
        'ffhq': 5000,
        'celebahq': 5000
    },
    'fake': {
        'stable-diffusion': 2000,
        'stylegan': 2000,
        'stargan': 2000,
        'sfhq': 2000,
        'face_synthetics': 2000
    }
}


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


def collect_images(source_dir):
    """
    Collect all images from the source directory.
    """
    
    source_path = Path(source_dir)
    if not source_path.exists():
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    
    for ext in image_extensions:
        images.extend(source_path.glob(f"*{ext}"))
        images.extend(source_path.glob(f"*{ext.upper()}"))
    
    return sorted(images)


def sample_and_split_images(all_images, target_count):
    """
    Sample and split the images into train/val/test sets.
    """

    sampling_count = int(target_count * 1.5)
    
    if len(all_images) < sampling_count:
        if len(all_images) < target_count:
            print(f"WARNING: Only {len(all_images)} images available, requested {target_count}")
        sampled = all_images.copy()
    else:
        sampled = random.sample(all_images, sampling_count)
    
    sampled.sort()
    
    train_count = int(len(sampled) * TRAIN_RATIO)
    val_count = int(len(sampled) * VAL_RATIO)
    
    splits = {
        'train': sampled[:train_count],
        'val': sampled[train_count:train_count + val_count],
        'test': sampled[train_count + val_count:]
    }
    
    return splits


def process_and_copy_image(src_path, dst_path, target_size=(224, 224)):
    """
    Process and copy the image to the destination directory.
    """

    img = cv2.imread(str(src_path))
    
    if img is None:
        return False
    
    cropped_img = crop_with_margin_and_resize(img, target_size)
    
    if cropped_img is None:
        return False
    
    cv2.imwrite(str(dst_path), cropped_img)
    return True


def process_category(category_type, category_name, target_count):
    """
    Process a category of images and copy them to the destination directory.
    """

    print(f"\nProcessing {category_type}/{category_name} (target: {target_count} images)")
    
    source_dir = BASE_DIR / category_type / category_name
    
    if not source_dir.exists():
        print(f"ERROR: Directory not found: {source_dir}")
        return {'train': 0, 'val': 0, 'test': 0}
    
    all_images = collect_images(source_dir)
    print(f"  Found {len(all_images)} images in source directory")
    
    if len(all_images) == 0:
        print(f"ERROR: No images found in {source_dir}")
        return {'train': 0, 'val': 0, 'test': 0}
    
    splits = sample_and_split_images(all_images, target_count)
    
    target_per_split = {
        'train': int(target_count * TRAIN_RATIO),
        'val': int(target_count * VAL_RATIO),
        'test': int(target_count * TEST_RATIO)
    }
    
    results = {}
    
    for split_name, images in splits.items():
        output_path = OUTPUT_DIR / split_name / category_type
        output_path.mkdir(parents=True, exist_ok=True)
        
        split_target = target_per_split[split_name]
        processed_count = 0
        img_output_idx = 0
        img_source_idx = 0
        
        with tqdm(total=split_target, desc=f"  {split_name}", leave=False) as pbar:
            while processed_count < split_target and img_source_idx < len(images):
                img_path = images[img_source_idx]
                output_filename = f"{category_type}_{category_name}_{img_output_idx:06d}.jpg"
                output_filepath = output_path / output_filename
                
                if process_and_copy_image(img_path, output_filepath):
                    processed_count += 1
                    img_output_idx += 1
                    pbar.update(1)
                
                img_source_idx += 1
                pbar.set_postfix({'valid': processed_count, 'attempted': img_source_idx})
        
        results[split_name] = processed_count
        if processed_count < split_target:
            print(f"    {split_name}: WARNING: {processed_count}/{split_target} images processed (exhausted {img_source_idx} sampled images)")
        else:
            print(f"    {split_name}: SUCCESS: {processed_count}/{split_target} images processed (from {img_source_idx} attempted)")
    
    return results


def main():
    print("=" * 80)
    print("ArtiFact Dataset Processor")
    print("=" * 80)
    
    print(f"\nInput directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    if not BASE_DIR.exists():
        print(f"\nERROR: Input directory does not exist: {BASE_DIR}")
        print("Please check the path and try again.")
        return
    
    print(f"\nSplit ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    
    all_results = {}
    
    print("\n" + "=" * 80)
    print("Processing REAL images")
    print("=" * 80)
    
    for category_name, target_count in TARGET_COUNTS['real'].items():
        results = process_category('real', category_name, target_count)
        all_results[f"real_{category_name}"] = results
    
    print("\n" + "=" * 80)
    print("Processing FAKE images")
    print("=" * 80)
    
    for category_name, target_count in TARGET_COUNTS['fake'].items():
        results = process_category('fake', category_name, target_count)
        all_results[f"fake_{category_name}"] = results
    
    print("\n" + "=" * 80)
    print("SUCCESS: Dataset processing complete!")
    print("=" * 80)
    
    print("\nSummary by split:")
    for split_name in ['train', 'val', 'test']:
        total_real = sum(all_results[f"real_{cat}"][split_name] for cat in TARGET_COUNTS['real'].keys())
        total_fake = sum(all_results[f"fake_{cat}"][split_name] for cat in TARGET_COUNTS['fake'].keys())
        print(f"  {split_name.upper()}: {total_real} real + {total_fake} fake = {total_real + total_fake} total")
    
    print("\nSummary by category:")
    for category_type in ['real', 'fake']:
        for category_name in TARGET_COUNTS[category_type].keys():
            key = f"{category_type}_{category_name}"
            total = sum(all_results[key].values())
            print(f"  {category_type}/{category_name}: {total} images")


if __name__ == "__main__":
    main()
