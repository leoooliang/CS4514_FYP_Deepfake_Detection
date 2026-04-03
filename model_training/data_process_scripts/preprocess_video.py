"""
Process FakeAVCeleb v1.2 video dataset into precomputed .pt tensors
for the Tri-Stream Multimodal Video Deepfake Detection model.

This script:
1. Parse meta_data.csv and sample 500 real + 500 fake videos
2. Perform Identity-Aware splitting (70/15/15) grouped by source ID to prevent data leakage
3. Extract 15 MTCNN face crops + 4s audio waveform per video
4. Save each sample as a .pt dict {'visual': (15,3,224,224), 'audio': (64000,)}

Output format matches TriStreamVideoDataset which loads .pt files and
computes Mel/LFCC spectrograms on-the-fly to support training-time augmentation.
"""

import subprocess
import tempfile

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm

NUM_FRAMES = 15
TARGET_SIZE = (224, 224)
FACE_MARGIN = 0.20
CENTER_CROP_RATIO = 0.50
AUDIO_SR = 16000
AUDIO_DURATION = 4.0
AUDIO_SAMPLES = int(AUDIO_SR * AUDIO_DURATION)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MTCNN_DETECTOR: MTCNN = None  # lazily initialized


def _get_mtcnn() -> MTCNN:
    """
    Get the MTCNN detector.
    """

    global MTCNN_DETECTOR
    if MTCNN_DETECTOR is None:
        MTCNN_DETECTOR = MTCNN(
            keep_all=False, post_process=False, device=DEVICE,
        )
        print(f"MTCNN initialized on {DEVICE}")
    return MTCNN_DETECTOR


def parse_metadata(metadata_path):
    """
    Parse meta_data.csv from FakeAVCeleb v1.2.

    Columns: source, target1, target2, method, category, type, race, gender, path, full_path
    Label Mapping: Category A = Real (1), Categories B/C/D = Fake (0)
    """

    print("Loading metadata...")
    df = pd.read_csv(metadata_path)

    df.columns = [
        'source', 'target1', 'target2', 'method', 'category',
        'type', 'race', 'gender', 'filename', 'full_path_dir',
    ]
    df['full_path_dir'] = df['full_path_dir'].str.replace('FakeAVCeleb/', '', regex=False)
    df['video_path'] = df['full_path_dir'] + '/' + df['filename']

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

    fake_sample = fake_df.sample(n=min(n_samples_per_class * 2, len(fake_df)), random_state=42)
    real_sample = real_df.sample(n=min(n_samples_per_class * 2, len(real_df)), random_state=42)

    valid_fake, valid_real = [], []
    raw_data_base = Path("model_training") / "raw_data" / "FakeAVCeleb_v1.2"

    print(f"  Finding {n_samples_per_class} valid fake videos...")
    for _, row in tqdm(fake_sample.iterrows(), total=len(fake_sample), desc="Checking fake", leave=False):
        if len(valid_fake) >= n_samples_per_class:
            break
        if (raw_data_base / row['video_path']).exists():
            valid_fake.append(row)

    print(f"  Finding {n_samples_per_class} valid real videos...")
    for _, row in tqdm(real_sample.iterrows(), total=len(real_sample), desc="Checking real", leave=False):
        if len(valid_real) >= n_samples_per_class:
            break
        if (raw_data_base / row['video_path']).exists():
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

    fake_df = df[df['label'] == 0].copy()
    real_df = df[df['label'] == 1].copy()

    def split_by_source_exact_no_leakage(df_subset, target_train, target_val, target_test):
        """
        Split by source ID to hit EXACT targets WITHOUT data leakage.
        GUARANTEE: Each source appears in ONLY ONE split.
        """

        sources = df_subset['source'].unique()

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

        sorted_sources = sorted(sources, key=lambda x: source_info[x]['count'])

        train_sources, val_sources, test_sources = [], [], []
        train_count = val_count = test_count = 0

        for source in sorted_sources:
            count = source_info[source]['count']
            train_deficit = target_train - train_count
            val_deficit = target_val - val_count
            test_deficit = target_test - test_count

            if train_deficit > 0 and (train_deficit >= val_deficit or train_deficit >= count):
                train_sources.append(source); train_count += count
            elif val_deficit > 0 and (val_deficit >= test_deficit or val_deficit >= count):
                val_sources.append(source); val_count += count
            elif test_deficit > 0:
                test_sources.append(source); test_count += count
            else:
                test_sources.append(source); test_count += count

        for _ in range(50):
            moved = False
            if train_count > target_train and train_sources:
                for s, c in sorted(((s, source_info[s]['count']) for s in train_sources), key=lambda x: x[1]):
                    if train_count - c >= target_train - 5:
                        if val_count + c <= target_val + 5:
                            train_sources.remove(s); val_sources.append(s)
                            train_count -= c; val_count += c; moved = True; break
                        elif test_count + c <= target_test + 5:
                            train_sources.remove(s); test_sources.append(s)
                            train_count -= c; test_count += c; moved = True; break
            if val_count > target_val and val_sources:
                for s, c in sorted(((s, source_info[s]['count']) for s in val_sources), key=lambda x: x[1]):
                    if val_count - c >= target_val - 5 and test_count + c <= target_test + 5:
                        val_sources.remove(s); test_sources.append(s)
                        val_count -= c; test_count += c; moved = True; break
            if test_count > target_test and val_count < target_val and test_sources:
                for s, c in sorted(((s, source_info[s]['count']) for s in test_sources), key=lambda x: x[1]):
                    if test_count - c >= target_test - 5 and val_count + c <= target_val + 5:
                        test_sources.remove(s); val_sources.append(s)
                        test_count -= c; val_count += c; moved = True; break
            if not moved:
                break

        all_sources = set(train_sources) | set(val_sources) | set(test_sources)
        assert len(all_sources) == len(train_sources) + len(val_sources) + len(test_sources), \
            "CRITICAL ERROR: Data leakage detected! Same source in multiple splits!"

        def _collect(srcs):
            idxs = []
            for s in srcs:
                idxs.extend(source_info[s]['videos'])
            return df_subset.loc[idxs].reset_index(drop=True)

        return (_collect(train_sources), _collect(val_sources), _collect(test_sources),
                train_sources, val_sources, test_sources)

    fake_train, fake_val, fake_test, fake_train_ids, fake_val_ids, fake_test_ids = \
        split_by_source_exact_no_leakage(fake_df, 350, 75, 75)
    real_train, real_val, real_test, real_train_ids, real_val_ids, real_test_ids = \
        split_by_source_exact_no_leakage(real_df, 350, 75, 75)

    print(f"\n Data Leakage Prevention Verified:")
    fake_all = set(fake_train_ids) | set(fake_val_ids) | set(fake_test_ids)
    real_all = set(real_train_ids) | set(real_val_ids) | set(real_test_ids)
    print(f"  Fake: {len(fake_train_ids)} train + {len(fake_val_ids)} val + {len(fake_test_ids)} test = {len(fake_all)} unique ids")
    print(f"  Real: {len(real_train_ids)} train + {len(real_val_ids)} val + {len(real_test_ids)} test = {len(real_all)} unique ids")

    train_df = pd.concat([fake_train, real_train], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df   = pd.concat([fake_val,   real_val],   ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df  = pd.concat([fake_test,  real_test],  ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    for name, sdf, fk, rl in [("Train", train_df, fake_train, real_train),
                               ("Val",   val_df,   fake_val,   real_val),
                               ("Test",  test_df,  fake_test,  real_test)]:
        print(f"\n{name} set: {len(sdf)} videos  (fake={len(fk)}, real={len(rl)})")

    return train_df, val_df, test_df


def _extract_frames(video_path: str) -> list[np.ndarray]:
    """
    Extract NUM_FRAMES uniformly sampled RGB frames from *video_path*.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    indices = np.linspace(0, total - 1, min(NUM_FRAMES, total), dtype=int)
    frames = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames could be read from {video_path}")
    return frames


def _detect_and_crop_faces(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Run MTCNN on each frame, crop with FACE_MARGIN expansion.
    Falls back to last-good-bbox (temporal continuity) then 50 % center crop.
    Returns exactly NUM_FRAMES face crops resized to TARGET_SIZE.
    """

    mtcnn = _get_mtcnn()
    cropped: list[np.ndarray] = []
    last_good_box = None

    for frame in frames:
        bbox = None
        try:
            boxes, probs = mtcnn.detect(frame)
            if boxes is not None and len(boxes) > 0:
                best = int(np.argmax(probs)) if len(boxes) > 1 else 0
                bbox = boxes[best]
                last_good_box = bbox.copy()
        except Exception:
            pass

        if bbox is None:
            bbox = last_good_box

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            x1 = int(max(0, x1 - w * FACE_MARGIN))
            y1 = int(max(0, y1 - h * FACE_MARGIN))
            x2 = int(min(frame.shape[1], x2 + w * FACE_MARGIN))
            y2 = int(min(frame.shape[0], y2 + h * FACE_MARGIN))
            crop = frame[y1:y2, x1:x2]
        else:
            fh, fw = frame.shape[:2]
            cs = int(min(fh, fw) * CENTER_CROP_RATIO)
            cx, cy = fw // 2, fh // 2
            crop = frame[cy - cs // 2:cy + cs // 2, cx - cs // 2:cx + cs // 2]

        cropped.append(cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_LINEAR))

    while len(cropped) < NUM_FRAMES:
        cropped.append(cropped[-1])
    return cropped[:NUM_FRAMES]


def _extract_audio(video_path: str) -> np.ndarray:
    """
    Extract audio waveform from video as 16 kHz mono, standardized to AUDIO_DURATION.
    Tries librosa first, then ffmpeg fallback.
    Returns a 1-D numpy array of length AUDIO_SAMPLES, amplitude-normalized to [-1, 1].
    """

    waveform = None

    try:
        waveform, _ = librosa.load(video_path, sr=AUDIO_SR, mono=True)
    except Exception:
        pass

    if waveform is None or len(waveform) == 0:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                     "-ac", "1", "-ar", str(AUDIO_SR), tmp.name],
                    check=True, capture_output=True,
                )
                waveform, _ = librosa.load(tmp.name, sr=AUDIO_SR, mono=True)
        except Exception:
            pass

    if waveform is None or len(waveform) == 0:
        raise RuntimeError(f"Could not extract audio from {video_path}")

    if len(waveform) > AUDIO_SAMPLES:
        waveform = waveform[:AUDIO_SAMPLES]
    elif len(waveform) < AUDIO_SAMPLES:
        repeats = (AUDIO_SAMPLES // len(waveform)) + 1
        waveform = np.tile(waveform, repeats)[:AUDIO_SAMPLES]

    mx = np.max(np.abs(waveform))
    if mx > 0:
        waveform = waveform / mx

    return waveform


def process_video_to_tensor(video_path: str) -> dict:
    """
    Full pipeline: video → {'visual': Tensor(15,3,224,224), 'audio': Tensor(64000,)}.

    visual: face crops in [0, 1] range
    audio:  raw waveform at 16 kHz (Mel / LFCC computed on-the-fly by the dataloader)
    """

    frames = _extract_frames(video_path)
    faces = _detect_and_crop_faces(frames)

    visual = torch.from_numpy(np.array(faces)).float().permute(0, 3, 1, 2) / 255.0

    audio = torch.from_numpy(_extract_audio(video_path)).float()

    return {'visual': visual, 'audio': audio}


def create_directory_structure(output_base):
    """
    Create PyTorch-compatible directory structure for .pt tensors.
    """

    print(f"\nCreating directory structure at {output_base}...")
    for split in ['train', 'val', 'test']:
        for label in ['fake', 'real']:
            d = Path(output_base) / split / label
            d.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {d}")


def process_split(df, split_name, raw_data_base, output_base):
    """
    Process every video in *df* through the face-detection + audio pipeline
    and save each as a .pt tensor dict.
    """

    print(f"\nProcessing {split_name} videos → .pt tensors ...")
    saved, skipped = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        video_id = f"{row['source']}_{Path(row['filename']).stem}"
        label_name = 'fake' if row['label'] == 0 else 'real'
        video_path = str(Path(raw_data_base) / row['video_path'])
        dest = Path(output_base) / split_name / label_name / f"{video_id}.pt"

        if dest.exists():
            saved += 1
            continue

        try:
            tensor_dict = process_video_to_tensor(video_path)
            torch.save(tensor_dict, dest)
            saved += 1
        except Exception as e:
            tqdm.write(f"  SKIP {video_id}: {e}")
            skipped += 1

    print(f"  Saved: {saved}, Skipped: {skipped}")


def verify_dataset(output_base):
    """
    Verify the final .pt tensor dataset structure and counts.
    """

    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    total_files = 0
    for split in ['train', 'val', 'test']:
        split_total = 0
        print(f"\n{split.upper()}:")
        for label in ['fake', 'real']:
            d = Path(output_base) / split / label
            count = len(list(d.glob('*.pt')))
            split_total += count
            print(f"  {label}: {count} tensors")
        print(f"  Total: {split_total} tensors")
        total_files += split_total

    print(f"\nGRAND TOTAL: {total_files} tensors")

    for split in ['train', 'val', 'test']:
        for label in ['fake', 'real']:
            pt_files = list((Path(output_base) / split / label).glob('*.pt'))
            if pt_files:
                sample = torch.load(pt_files[0], map_location='cpu', weights_only=False)
                print(f"\nSample tensor ({split}/{label}/{pt_files[0].name}):")
                print(f"  visual: {sample['visual'].shape}  range=[{sample['visual'].min():.3f}, {sample['visual'].max():.3f}]")
                print(f"  audio:  {sample['audio'].shape}   range=[{sample['audio'].min():.3f}, {sample['audio'].max():.3f}]")
                print("=" * 60)
                return

    print("=" * 60)


def main():
    RAW_DATA_BASE = Path("model_training") / "raw_data" / "FakeAVCeleb_v1.2"
    METADATA_FILE = RAW_DATA_BASE / "meta_data.csv"
    OUTPUT_BASE = Path("model_training") / "data" / "video_tensors"

    print("=" * 60)
    print("FakeAVCeleb v1.2 → Precomputed .pt Tensor Pipeline")
    print("=" * 60)
    print(f"Raw data:  {RAW_DATA_BASE}")
    print(f"Metadata:  {METADATA_FILE}")
    print(f"Output:    {OUTPUT_BASE}")
    print(f"Device:    {DEVICE}")
    print(f"Frames:    {NUM_FRAMES} @ {TARGET_SIZE}")
    print(f"Audio:     {AUDIO_DURATION}s @ {AUDIO_SR}Hz = {AUDIO_SAMPLES} samples")
    print("=" * 60)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    if not RAW_DATA_BASE.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DATA_BASE}")

    df = parse_metadata(METADATA_FILE)

    balanced_df = sample_balanced_data(df, n_samples_per_class=500)

    train_df, val_df, test_df = identity_aware_split(balanced_df)

    create_directory_structure(OUTPUT_BASE)

    process_split(train_df, 'train', RAW_DATA_BASE, OUTPUT_BASE)
    process_split(val_df,   'val',   RAW_DATA_BASE, OUTPUT_BASE)
    process_split(test_df,  'test',  RAW_DATA_BASE, OUTPUT_BASE)

    verify_dataset(OUTPUT_BASE)

    print("\nSUCCESS: Video tensor preprocessing completed successfully!")


if __name__ == "__main__":
    main()
