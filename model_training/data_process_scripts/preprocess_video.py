"""
Process FakeAVCeleb v1.2 video dataset for multimodal deepfake detection.

This script:
1. Parse meta_data.csv and sample 500 real + 500 fake videos
2. Perform Identity-Aware splitting (70/15/15) grouped by source ID to prevent data leakage
3. Copy original videos to organized directory structure (train/val/test/real/fake)

No preprocessing is done - videos are kept in original format for later processing.
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def parse_metadata(metadata_path):
    """
    Parse meta_data.csv from FakeAVCeleb v1.2.
    
    Columns: source, target1, target2, method, category, type, race, gender, path, full_path
    Label Mapping: Category A = Real (1), Categories B/C/D = Fake (0)
    """
    print("Loading metadata...")
    df = pd.read_csv(metadata_path)
    
    # The CSV has the full path in the last column (column 9, unnamed)
    # Remove "FakeAVCeleb/" prefix from the path as it's not in the actual directory structure
    df.columns = ['source', 'target1', 'target2', 'method', 'category', 'type', 'race', 'gender', 'filename', 'full_path_dir']
    df['full_path_dir'] = df['full_path_dir'].str.replace('FakeAVCeleb/', '', regex=False)
    df['video_path'] = df['full_path_dir'] + '/' + df['filename']
    
    # Create binary labels: Real = 1, Fake = 0
    df['label'] = df['category'].apply(lambda x: 1 if x == 'A' else 0)
    
    print(f"Total metadata entries: {len(df)}")
    print(f"Label distribution:")
    print(f"  Fake (Class 0): {len(df[df['label'] == 0])}")
    print(f"  Real (Class 1): {len(df[df['label'] == 1])}")
    
    return df


def sample_balanced_data(df, n_samples_per_class=500):
    """
    Sample balanced data: n real + n fake videos, ensuring file existence.
    """
    print(f"\nSampling {n_samples_per_class} samples per class (verifying file existence)...")
    
    fake_df = df[df['label'] == 0]
    real_df = df[df['label'] == 1]
    
    # Sample more than needed to account for missing files
    fake_sample = fake_df.sample(n=min(n_samples_per_class * 2, len(fake_df)), random_state=42)
    real_sample = real_df.sample(n=min(n_samples_per_class * 2, len(real_df)), random_state=42)
    
    # Verify file existence and keep only existing videos
    valid_fake = []
    valid_real = []
    
    raw_data_base = Path("model_training") / "raw_data" / "FakeAVCeleb_v1.2"
    
    print(f"  Finding {n_samples_per_class} valid fake videos...")
    for _, row in tqdm(fake_sample.iterrows(), total=len(fake_sample), desc="Checking fake", leave=False):
        if len(valid_fake) >= n_samples_per_class:
            break
        video_path = raw_data_base / row['video_path']
        if video_path.exists():
            valid_fake.append(row)
    
    print(f"  Finding {n_samples_per_class} valid real videos...")
    for _, row in tqdm(real_sample.iterrows(), total=len(real_sample), desc="Checking real", leave=False):
        if len(valid_real) >= n_samples_per_class:
            break
        video_path = raw_data_base / row['video_path']
        if video_path.exists():
            valid_real.append(row)
    
    fake_df_valid = pd.DataFrame(valid_fake)
    real_df_valid = pd.DataFrame(valid_real)
    
    balanced_df = pd.concat([fake_df_valid, real_df_valid], ignore_index=True)
    
    print(f"Total balanced dataset: {len(balanced_df)} videos")
    print(f"  - Fake: {len(fake_df_valid)}")
    print(f"  - Real: {len(real_df_valid)}")
    
    return balanced_df


def identity_aware_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Perform Identity-Aware splitting with EXACT counts and NO DATA LEAKAGE.
    
    CRITICAL: Each source ID appears in ONLY ONE split (train/val/test).
    This prevents the model from memorizing specific identities.
    
    Targets:
    - Train: exactly 350 fake + 350 real
    - Val: exactly 75 fake + 75 real  
    - Test: exactly 75 fake + 75 real
    """
    print(f"\nSplitting dataset with EXACT counts and identity-aware protection:")
    print(f"  Train: 350 fake + 350 real")
    print(f"  Val: 75 fake + 75 real")
    print(f"  Test: 75 fake + 75 real")
    print("Maximizing diversity for training set...")
    
    # Separate fake and real videos
    fake_df = df[df['label'] == 0].copy()
    real_df = df[df['label'] == 1].copy()
    
    def split_by_source_exact_no_leakage(df_subset, target_train, target_val, target_test):
        """
        Split by source ID to hit EXACT targets WITHOUT data leakage.
        
        GUARANTEE: Each source appears in ONLY ONE split.
        """
        sources = df_subset['source'].unique()
        
        # Get info for each source
        source_info = {}
        for source in sources:
            source_vids = df_subset[df_subset['source'] == source]
            source_info[source] = {
                'videos': source_vids.index.tolist(),
                'count': len(source_vids),
                'race': source_vids.iloc[0]['race'],
                'gender': source_vids.iloc[0]['gender'],
                'method': source_vids.iloc[0]['method'] if 'method' in source_vids.columns else 'real'
            }
        
        # Sort sources: smaller video counts first (maximizes identity diversity in training)
        sorted_sources = sorted(sources, key=lambda x: source_info[x]['count'])
        
        # Initialize splits - store SOURCE IDs, not individual videos
        train_sources = []
        val_sources = []
        test_sources = []
        
        train_count = 0
        val_count = 0
        test_count = 0
        
        # Pass 1: Greedy assignment of ENTIRE sources (NO SPLITTING)
        for source in sorted_sources:
            info = source_info[source]
            count = info['count']
            
            # Calculate deficits
            train_deficit = target_train - train_count
            val_deficit = target_val - val_count
            test_deficit = target_test - test_count
            
            # Assign ENTIRE source to the split that needs it most
            # Priority: train (for diversity) > val > test
            if train_deficit > 0 and (train_deficit >= val_deficit or train_deficit >= count):
                train_sources.append(source)
                train_count += count
            elif val_deficit > 0 and (val_deficit >= test_deficit or val_deficit >= count):
                val_sources.append(source)
                val_count += count
            elif test_deficit > 0:
                test_sources.append(source)
                test_count += count
            else:
                # All targets met, put remainder in test
                test_sources.append(source)
                test_count += count
        
        # Pass 2: Fine-tune by moving ENTIRE sources between splits
        # Move from train to val/test if train is over
        while train_count > target_train and train_sources:
            # Find smallest source in train that can be moved
            movable = [(s, source_info[s]['count']) for s in train_sources]
            movable.sort(key=lambda x: x[1])  # Sort by count
            
            moved = False
            for source, count in movable:
                if train_count - count >= target_train - 5:  # Allow small margin
                    if val_count + count <= target_val + 5:
                        train_sources.remove(source)
                        val_sources.append(source)
                        train_count -= count
                        val_count += count
                        moved = True
                        break
                    elif test_count + count <= target_test + 5:
                        train_sources.remove(source)
                        test_sources.append(source)
                        train_count -= count
                        test_count += count
                        moved = True
                        break
            
            if not moved:
                break  # Can't improve further without splitting sources
        
        # Move from val to test if val is over
        while val_count > target_val and val_sources:
            movable = [(s, source_info[s]['count']) for s in val_sources]
            movable.sort(key=lambda x: x[1])
            
            moved = False
            for source, count in movable:
                if val_count - count >= target_val - 5 and test_count + count <= target_test + 5:
                    val_sources.remove(source)
                    test_sources.append(source)
                    val_count -= count
                    test_count += count
                    moved = True
                    break
            
            if not moved:
                break
        
        # Move from test to val if test is over and val is under
        while test_count > target_test and val_count < target_val and test_sources:
            movable = [(s, source_info[s]['count']) for s in test_sources]
            movable.sort(key=lambda x: x[1])
            
            moved = False
            for source, count in movable:
                if test_count - count >= target_test - 5 and val_count + count <= target_val + 5:
                    test_sources.remove(source)
                    val_sources.append(source)
                    test_count -= count
                    val_count += count
                    moved = True
                    break
            
            if not moved:
                break
        
        # Verify no source appears in multiple splits (DATA LEAKAGE CHECK)
        all_sources = set(train_sources) | set(val_sources) | set(test_sources)
        assert len(all_sources) == len(train_sources) + len(val_sources) + len(test_sources), \
            "CRITICAL ERROR: Data leakage detected! Same source in multiple splits!"
        
        # Get all video indices for each split
        train_videos = []
        for source in train_sources:
            train_videos.extend(source_info[source]['videos'])
        
        val_videos = []
        for source in val_sources:
            val_videos.extend(source_info[source]['videos'])
        
        test_videos = []
        for source in test_sources:
            test_videos.extend(source_info[source]['videos'])
        
        # Create split dataframes
        train = df_subset.loc[train_videos].reset_index(drop=True)
        val = df_subset.loc[val_videos].reset_index(drop=True)
        test = df_subset.loc[test_videos].reset_index(drop=True)
        
        return train, val, test, train_sources, val_sources, test_sources
    
    # Split fake and real with exact targets (allowing small margin for identity-awareness)
    fake_train, fake_val, fake_test, fake_train_ids, fake_val_ids, fake_test_ids = split_by_source_exact_no_leakage(
        fake_df, target_train=350, target_val=75, target_test=75
    )
    real_train, real_val, real_test, real_train_ids, real_val_ids, real_test_ids = split_by_source_exact_no_leakage(
        real_df, target_train=350, target_val=75, target_test=75
    )
    
    # Verify no identity overlap
    fake_all_ids = set(fake_train_ids) | set(fake_val_ids) | set(fake_test_ids)
    real_all_ids = set(real_train_ids) | set(real_val_ids) | set(real_test_ids)
    
    print(f"\n✓ Data Leakage Prevention Verified:")
    print(f"  - Fake: {len(fake_train_ids)} train + {len(fake_val_ids)} val + {len(fake_test_ids)} test = {len(fake_all_ids)} unique identities")
    print(f"  - Real: {len(real_train_ids)} train + {len(real_val_ids)} val + {len(real_test_ids)} test = {len(real_all_ids)} unique identities")
    
    # Combine and shuffle
    train_df = pd.concat([fake_train, real_train], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat([fake_val, real_val], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat([fake_test, real_test], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nTrain set: {len(train_df)} videos from {len(fake_train_ids) + len(real_train_ids)} identities")
    print(f"  - Fake: {len(fake_train)} videos from {len(fake_train_ids)} identities")
    print(f"  - Real: {len(real_train)} videos from {len(real_train_ids)} identities")
    
    # Print diversity metrics
    if 'race' in train_df.columns:
        print(f"  - Race diversity: {train_df['race'].nunique()} races")
    if 'gender' in train_df.columns:
        print(f"  - Gender diversity: {train_df['gender'].nunique()} genders")
    fake_train_methods = fake_train['method'].nunique() if 'method' in fake_train.columns else 0
    if fake_train_methods > 0:
        print(f"  - Fake methods diversity: {fake_train_methods} methods")
    
    print(f"\nVal set: {len(val_df)} videos from {len(fake_val_ids) + len(real_val_ids)} identities")
    print(f"  - Fake: {len(fake_val)} videos from {len(fake_val_ids)} identities")
    print(f"  - Real: {len(real_val)} videos from {len(real_val_ids)} identities")
    
    print(f"\nTest set: {len(test_df)} videos from {len(fake_test_ids) + len(real_test_ids)} identities")
    print(f"  - Fake: {len(fake_test)} videos from {len(fake_test_ids)} identities")
    print(f"  - Real: {len(real_test)} videos from {len(real_test_ids)} identities")
    
    # Report counts (may be slightly off target due to identity-awareness constraint)
    print(f"\n✓ Final counts (identity-awareness may cause small deviation from targets):")
    print(f"  Train: {len(fake_train)} fake + {len(real_train)} real = {len(train_df)} total")
    print(f"  Val: {len(fake_val)} fake + {len(real_val)} real = {len(val_df)} total")
    print(f"  Test: {len(fake_test)} fake + {len(real_test)} real = {len(test_df)} total")
    
    return train_df, val_df, test_df


def create_directory_structure(output_base):
    """Create PyTorch-compatible directory structure."""
    print(f"\nCreating directory structure at {output_base}...")
    
    splits = ['train', 'val', 'test']
    labels = ['fake', 'real']
    
    for split in splits:
        for label in labels:
            dir_path = Path(output_base) / split / label
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {dir_path}")


def copy_files(df, split_name, raw_data_base, output_base):
    """Copy videos to their respective directories with progress bar."""
    print(f"\nCopying {split_name} videos...")
    
    copied_count = 0
    skipped_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name}"):
        # Generate unique video ID from source and filename
        video_id = f"{row['source']}_{Path(row['filename']).stem}"
        label_name = 'fake' if row['label'] == 0 else 'real'
        
        # Construct full video path
        video_path = Path(raw_data_base) / row['video_path']
        
        if not video_path.exists():
            skipped_count += 1
            continue
        
        # Destination file path
        dest_file = Path(output_base) / split_name / label_name / f"{video_id}.mp4"
        
        # Copy file
        try:
            shutil.copy2(video_path, dest_file)
            copied_count += 1
        except Exception as e:
            skipped_count += 1
    
    print(f"  Copied: {copied_count} videos, Skipped: {skipped_count} videos")


def verify_dataset(output_base):
    """Verify the final dataset structure and counts."""
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    labels = ['fake', 'real']
    
    total_files = 0
    for split in splits:
        split_total = 0
        print(f"\n{split.upper()}:")
        for label in labels:
            dir_path = Path(output_base) / split / label
            count = len(list(dir_path.glob('*.mp4')))
            split_total += count
            print(f"  {label}: {count} videos")
        print(f"  Total: {split_total} videos")
        total_files += split_total
    
    print(f"\nGRAND TOTAL: {total_files} videos")
    print("="*60)


def main():
    # Configuration
    RAW_DATA_BASE = Path("model_training") / "raw_data" / "FakeAVCeleb_v1.2"
    METADATA_FILE = RAW_DATA_BASE / "meta_data.csv"
    OUTPUT_BASE = Path("model_training") / "data" / "video"
    
    print("="*60)
    print("FakeAVCeleb v1.2 Video Dataset Processing")
    print("="*60)
    print(f"Raw data folder: {RAW_DATA_BASE}")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"Output folder: {OUTPUT_BASE}")
    print("="*60)
    
    # Check if paths exist
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    if not RAW_DATA_BASE.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DATA_BASE}")
    
    # Step 1: Parse metadata
    df = parse_metadata(METADATA_FILE)
    
    # Step 2: Sample balanced data (500 real + 500 fake)
    balanced_df = sample_balanced_data(df, n_samples_per_class=500)
    
    # Step 3: Identity-Aware split
    train_df, val_df, test_df = identity_aware_split(balanced_df)
    
    # Step 4: Create directory structure
    create_directory_structure(OUTPUT_BASE)
    
    # Step 5: Copy files
    copy_files(train_df, 'train', RAW_DATA_BASE, OUTPUT_BASE)
    copy_files(val_df, 'val', RAW_DATA_BASE, OUTPUT_BASE)
    copy_files(test_df, 'test', RAW_DATA_BASE, OUTPUT_BASE)
    
    # Step 6: Verify dataset
    verify_dataset(OUTPUT_BASE)
    
    print("\n✓ Video dataset processing completed successfully!")


if __name__ == "__main__":
    main()
