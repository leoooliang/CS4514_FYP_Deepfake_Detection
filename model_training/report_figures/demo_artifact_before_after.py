"""
Figure generator for reports: 
Figure 4.2: Image Preprocessing of A Sample Image with MTCNN and 15% Margin 

Output: artifact_preprocess_before_after.png
(The output image is not exactly the same as the figure in the report, but the overall effect is the same.)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_DIR = SCRIPT_DIR.parent
DATA_PROCESS_DIR = MODEL_TRAINING_DIR / "data_process_scripts"
if str(DATA_PROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PROCESS_DIR))

import process_artifact_dataset as artifact


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_all_raw_images(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    paths: list[Path] = []
    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    return sorted(paths)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_for_display(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / m
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def find_sample_with_face(
    candidates: list[Path],
    rng: random.Random,
    max_attempts: int,
) -> tuple[Path, np.ndarray, np.ndarray]:
    if not candidates:
        raise SystemExit(f"No images found under {artifact.BASE_DIR}")

    order = candidates.copy()
    rng.shuffle(order)
    for i, path in enumerate(order[:max_attempts]):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        out = artifact.crop_with_margin_and_resize(bgr, target_size=(224, 224))
        if out is not None:
            return path, bgr, out

    raise SystemExit(
        f"No face detected in first {min(max_attempts, len(order))} tried images. "
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Before/after figure for ArtiFact preprocessing.")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Specific raw image path; if omitted, picks randomly under raw ArtiFact tree.",
    )
    parser.add_argument("--seed", type=int, default=43, help="RNG seed.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=200,
        help="Max random files to try when no --image.",
    )
    parser.add_argument(
        "--max-display-side",
        type=int,
        default=1024,
        help="Max longest side (px) for the 'before' panel only.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=SCRIPT_DIR / "artifact_preprocess_before_after.png",
        help="Output PNG path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI for print.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.image is not None:
        if not args.image.is_file():
            raise SystemExit(f"File not found: {args.image}")
        bgr = cv2.imread(str(args.image))
        if bgr is None:
            raise SystemExit(f"Could not read image: {args.image}")
        out = artifact.crop_with_margin_and_resize(bgr, target_size=(224, 224))
        if out is None:
            raise SystemExit("No face detected in --image; choose another file.")
        src_path = args.image
    else:
        if not artifact.BASE_DIR.exists():
            raise SystemExit(
                f"Raw data folder missing: {artifact.BASE_DIR}\n"
            )
        all_imgs = collect_all_raw_images(artifact.BASE_DIR)
        src_path, bgr, out = find_sample_with_face(all_imgs, rng, args.max_attempts)

    before_show = resize_for_display(bgr, args.max_display_side)
    before_rgb = bgr_to_rgb(before_show)
    after_rgb = bgr_to_rgb(out)

    ah, aw = after_rgb.shape[0], after_rgb.shape[1]
    target_h = before_rgb.shape[0]
    scale = target_h / ah
    tw = max(1, int(round(aw * scale)))
    after_large = cv2.resize(
        out, (tw, target_h), interpolation=cv2.INTER_LINEAR
    )
    after_large_rgb = bgr_to_rgb(after_large)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 12,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    axes[0].imshow(before_rgb)
    axes[0].set_title("Before")
    axes[0].axis("off")

    axes[1].imshow(after_large_rgb)
    axes[1].set_title("After: MTCNN Face Crop + 15% margin")
    axes[1].axis("off")

    fig.suptitle(
        "Dataset preprocessing",
        fontsize=13,
        fontweight="600",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {args.output.resolve()}")
    print(f"Source: {src_path}")


if __name__ == "__main__":
    main()
