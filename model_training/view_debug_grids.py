"""
Debug Grid Viewer for Video Feature Extraction

This script helps you quickly visualize the debug grids (15 cropped faces)
to verify that MTCNN face detection and cropping worked correctly.

Usage:
    python view_debug_grids.py --split train --label fake --num_samples 5
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import argparse


def view_debug_grids(debug_dir='../data/debug_crops', split='train', label='fake', num_samples=5):
    """
    Display debug grids for visual inspection.
    
    Args:
        debug_dir: Root directory containing debug grids
        split: train/val/test
        label: fake/real
        num_samples: Number of samples to display
    """
    debug_path = Path(debug_dir) / split / label
    
    if not debug_path.exists():
        print(f"❌ Debug directory not found: {debug_path}")
        return
    
    grid_files = sorted(list(debug_path.glob('*.jpg')))
    
    if len(grid_files) == 0:
        print(f"❌ No debug grids found in {debug_path}")
        return
    
    print(f"Found {len(grid_files)} debug grids in {debug_path}")
    print(f"Displaying {min(num_samples, len(grid_files))} samples...\n")
    
    # Display samples
    fig, axes = plt.subplots(min(num_samples, len(grid_files)), 1, 
                             figsize=(20, 4 * min(num_samples, len(grid_files))))
    
    if num_samples == 1:
        axes = [axes]
    
    for idx, grid_file in enumerate(grid_files[:num_samples]):
        # Read image
        img = cv2.imread(str(grid_file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"{grid_file.stem} ({split}/{label})", fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'debug_grid_preview_{split}_{label}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Preview saved to debug_grid_preview_{split}_{label}.png")


def check_quality_issues(debug_dir='../data/debug_crops', split='train', label='fake'):
    """
    Check for potential quality issues in debug grids.
    
    Looks for:
    - Grids with very small faces (face detection might have failed)
    - Grids with inconsistent face sizes (tracking might have failed)
    """
    debug_path = Path(debug_dir) / split / label
    
    if not debug_path.exists():
        print(f"❌ Debug directory not found: {debug_path}")
        return
    
    grid_files = sorted(list(debug_path.glob('*.jpg')))
    print(f"\nAnalyzing {len(grid_files)} debug grids for quality issues...")
    
    issues = []
    
    for grid_file in grid_files:
        img = cv2.imread(str(grid_file))
        h, w = img.shape[:2]
        
        # Each face should be 224x224, with 15 faces horizontally
        expected_width = 224 * 15
        expected_height = 224
        
        if w != expected_width or h != expected_height:
            issues.append({
                'file': grid_file.name,
                'issue': f'Unexpected dimensions: {w}x{h} (expected {expected_width}x{expected_height})',
                'type': 'dimension'
            })
    
    if len(issues) == 0:
        print("✓ No quality issues detected!")
    else:
        print(f"⚠ Found {len(issues)} potential issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue['file']}: {issue['issue']}")
        
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    
    return issues


def main():
    parser = argparse.ArgumentParser(description='View debug grids from video feature extraction')
    parser.add_argument('--debug_dir', type=str, default='../data/debug_crops',
                        help='Root directory containing debug grids')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Split to view (train/val/test)')
    parser.add_argument('--label', type=str, default='fake', choices=['fake', 'real'],
                        help='Label to view (fake/real)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to display')
    parser.add_argument('--check_quality', action='store_true',
                        help='Check for quality issues instead of viewing')
    
    args = parser.parse_args()
    
    if args.check_quality:
        check_quality_issues(args.debug_dir, args.split, args.label)
    else:
        view_debug_grids(args.debug_dir, args.split, args.label, args.num_samples)


if __name__ == '__main__':
    main()
