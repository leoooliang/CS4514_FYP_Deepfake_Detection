"""
Process ASVspoof 2021 audio dataset for PyTorch training.

This script:
1. Parses the trial_metadata.txt file to get the metadata of the audio files
2. Samples a balanced subset of 10,000 audio files (5500 real + 5500 fake)
3. Splits them into Train (70%), Validation (15%), and Test (15%) sets
4. Organizes files into PyTorch-compatible directory structures
"""

import os
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_metadata(metadata_path):
    """
    Parse the trial_metadata.txt file.
    """

    print("Loading metadata...")
    
    columns = [
        "speaker_id", 
        "utterance_id", 
        "codec", 
        "source_db", 
        "attack_id", 
        "label", 
        "trim", 
        "partition", 
        "vocoder", 
        "vcc_task", 
        "vcc_team", 
        "vcc_gender", 
        "vcc_language"
    ]
    
    df = pd.read_csv(metadata_path, sep=' ', header=None, names=columns)
    print(f"Total metadata entries: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def sample_balanced_data(df, flac_folder, n_samples_per_class=5500):
    """
    Sample balanced data: 5500 real + 5500 fake, ensuring files exist.
    """

    print(f"\nSampling {n_samples_per_class} samples per class (verifying file existence)...")
    
    def sample_valid_files(label_df, label_name, target_count):
        """
        Sample files for a specific label, skipping missing files."
        """

        valid_files = []
        attempted = 0
        shuffled_df = label_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n  Finding {target_count} valid {label_name} files...")
        for idx, row in tqdm(shuffled_df.iterrows(), 
                            total=len(shuffled_df), 
                            desc=f"  Checking {label_name}",
                            leave=False):
            attempted += 1
            utterance_id = row['utterance_id']
            source_file = Path(flac_folder) / f"{utterance_id}.flac"
            
            if source_file.exists():
                valid_files.append(row)
                if len(valid_files) >= target_count:
                    break
        
        result_df = pd.DataFrame(valid_files)
        missing_count = attempted - len(valid_files)
        print(f"  {label_name}: Found {len(result_df)} valid files (checked {attempted}, skipped {missing_count} missing)")
        
        if len(result_df) < target_count:
            print(f"  WARNING: Only found {len(result_df)} valid {label_name} files, target was {target_count}")
        
        return result_df
    
    bonafide_df = df[df['label'] == 'bonafide']
    valid_bonafide = sample_valid_files(bonafide_df, 'bonafide (real)', n_samples_per_class)
    
    spoof_df = df[df['label'] == 'spoof']
    valid_spoof = sample_valid_files(spoof_df, 'spoof (fake)', n_samples_per_class)
    
    balanced_df = pd.concat([valid_bonafide, valid_spoof], ignore_index=True)
    print(f"\nTotal balanced dataset: {len(balanced_df)} files")
    print(f"  - Real: {len(valid_bonafide)}")
    print(f"  - Fake: {len(valid_spoof)}")
    
    return balanced_df


def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets with stratification.
    """

    print(f"\nSplitting dataset: Train={train_ratio*100}%, Val={val_ratio*100}%, Test={test_ratio*100}%")
    
    labels = (df['label'] == 'spoof').astype(int)
    
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio), 
        stratify=labels, 
        random_state=42
    )
    
    temp_labels = (temp_df['label'] == 'spoof').astype(int)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        stratify=temp_labels, 
        random_state=42
    )
    
    print(f"Train set: {len(train_df)} files")
    print(f"  - Real: {len(train_df[train_df['label'] == 'bonafide'])}")
    print(f"  - Fake: {len(train_df[train_df['label'] == 'spoof'])}")
    
    print(f"Val set: {len(val_df)} files")
    print(f"  - Real: {len(val_df[val_df['label'] == 'bonafide'])}")
    print(f"  - Fake: {len(val_df[val_df['label'] == 'spoof'])}")
    
    print(f"Test set: {len(test_df)} files")
    print(f"  - Real: {len(test_df[test_df['label'] == 'bonafide'])}")
    print(f"  - Fake: {len(test_df[test_df['label'] == 'spoof'])}")
    
    return train_df, val_df, test_df


def create_directory_structure(output_base):
    """
    Create PyTorch-compatible directory structure.
    """

    print(f"\nCreating directory structure at {output_base}...")
    
    splits = ['train', 'val', 'test']
    classes = ['real', 'fake']
    
    for split in splits:
        for cls in classes:
            dir_path = Path(output_base) / split / cls
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {dir_path}")


def copy_files(df, split_name, flac_folder, output_base):
    """
    Copy files to their respective directories with progress bar.
    """

    print(f"\nCopying {split_name} files...")
    
    copied_count = 0
    skipped_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name}"):
        utterance_id = row['utterance_id']
        label = row['label']
        
        class_folder = 'real' if label == 'bonafide' else 'fake'
        
        source_file = Path(flac_folder) / f"{utterance_id}.flac"
        
        dest_file = Path(output_base) / split_name / class_folder / f"{utterance_id}.flac"
        
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            copied_count += 1
        else:
            skipped_count += 1
            print(f"\nWarning: Source file not found (unexpected): {source_file}")
    
    print(f"  Copied: {copied_count} files, Skipped: {skipped_count} files")


def verify_dataset(output_base):
    """
    Verify the final dataset structure and counts.
    """

    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    classes = ['real', 'fake']
    
    total_files = 0
    for split in splits:
        split_total = 0
        print(f"\n{split.upper()}:")
        for cls in classes:
            dir_path = Path(output_base) / split / cls
            count = len(list(dir_path.glob('*.flac')))
            split_total += count
            print(f"  {cls}: {count} files")
        print(f"  Total: {split_total} files")
        total_files += split_total
    
    print(f"\nGRAND TOTAL: {total_files} files")
    print("="*60)


def main():
    RAW_DATA_BASE = Path("model_training") / "raw_data"
    FLAC_FOLDER = RAW_DATA_BASE / "ASVspoof2021_DF_eval" / "flac"
    METADATA_FILE = RAW_DATA_BASE / "ASVspoof2021_DF_eval" / "trial_metadata.txt"
    OUTPUT_BASE = Path("model_training") / "data" / "audio"
    
    print("="*60)
    print("ASVspoof 2021 Audio Dataset Processing")
    print("="*60)
    print(f"Raw data folder: {RAW_DATA_BASE}")
    print(f"FLAC folder: {FLAC_FOLDER}")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"Output folder: {OUTPUT_BASE}")
    print("="*60)
    
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    if not FLAC_FOLDER.exists():
        raise FileNotFoundError(f"FLAC folder not found: {FLAC_FOLDER}")
    
    df = parse_metadata(METADATA_FILE)
    
    balanced_df = sample_balanced_data(df, FLAC_FOLDER, n_samples_per_class=5500)
    
    train_df, val_df, test_df = split_dataset(balanced_df)
    
    create_directory_structure(OUTPUT_BASE)
    
    copy_files(train_df, 'train', FLAC_FOLDER, OUTPUT_BASE)
    copy_files(val_df, 'val', FLAC_FOLDER, OUTPUT_BASE)
    copy_files(test_df, 'test', FLAC_FOLDER, OUTPUT_BASE)
    
    verify_dataset(OUTPUT_BASE)
    
    print("\nSUCCESS: Dataset processing completed successfully!")


if __name__ == "__main__":
    main()
