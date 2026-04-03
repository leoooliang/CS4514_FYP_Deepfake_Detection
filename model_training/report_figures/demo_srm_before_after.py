"""
Figure: random fake RGB image vs SRM (Spatial Rich Model) noise residual.

Matches training: image scaled to [0, 255], depthwise 5×5 SRM on each channel;
residual display uses clamp [-3, 3], ÷3 to [-1, 1], then mapped to [0, 1] for imshow.

Run from repo root or this folder:
  python model_training/report_figures/demo_srm_before_after.py
  python demo_srm_before_after.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_DIR = SCRIPT_DIR.parent / "utils"
DATA_PROCESS_DIR = SCRIPT_DIR.parent / "data_process_scripts"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
if str(DATA_PROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PROCESS_DIR))

from models import SRMConv2d  # noqa: E402

try:
    import process_artifact_dataset as artifact  # noqa: E402
except ImportError:
    artifact = None  # type: ignore

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FAKE_HINTS = {
    "fake",
    "deepfake",
    "stylegan",
    "stargan",
    "stable-diffusion",
    "face_synthetics",
    "sfhq",
}


def collect_images(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    paths: list[Path] = []
    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    return sorted(paths)


def is_fake_path(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    joined = "/".join(parts)
    return any(hint in joined for hint in FAKE_HINTS)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / m
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def srm_residual_to_display(srm_chw: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) or (3,H,W) float residual -> (H,W,3) in [0,1] for RGB imshow."""
    if srm_chw.dim() == 4:
        srm_chw = srm_chw.squeeze(0)
    x = torch.clamp(srm_chw, min=-3.0, max=3.0) / 3.0
    x = (x + 1.0) * 0.5
    return x.permute(1, 2, 0).cpu().numpy().astype(np.float32)


def pick_random_fake_image(data_dir: Path, rng: random.Random) -> Path:
    paths = [p for p in collect_images(data_dir) if is_fake_path(p)]
    if not paths:
        raise SystemExit(
            f"No fake images found under {data_dir}. "
            "Pass --image with a fake image, or point --data-dir to a fake subset."
        )
    return rng.choice(paths)


def main() -> None:
    default_dir = (
        artifact.BASE_DIR
        if artifact is not None and artifact.BASE_DIR.exists()
        else None
    )
    parser = argparse.ArgumentParser(
        description="Before/after figure for SRM filter on a fake image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Specific fake image path; if omitted, picks a random fake image under --data-dir.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_dir,
        help="Directory to sample a random fake image (default: ArtiFact raw if present).",
    )
    parser.add_argument("--seed", type=int, default=125, help="RNG seed.")
    parser.add_argument(
        "--max-side",
        type=int,
        default=512,
        help="Resize so longest side is at most this (px) before SRM.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=SCRIPT_DIR / "srm_before_after.png",
        help="Output PNG path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.image is not None:
        if not args.image.is_file():
            raise SystemExit(f"File not found: {args.image}")
        src = args.image
    else:
        if args.data_dir is None or not args.data_dir.is_dir():
            raise SystemExit(
                "Provide --image PATH to a fake image or a valid --data-dir with fake images "
                "(or place ArtiFact under model_training/raw_data/ArtiFact)."
            )
        src = pick_random_fake_image(args.data_dir, rng)

    bgr = cv2.imread(str(src))
    if bgr is None:
        raise SystemExit(f"Could not read image: {src}")

    bgr = resize_max_side(bgr, args.max_side)
    rgb_u8 = bgr_to_rgb(bgr)

    # [0, 255] float, NCHW — same convention as training notebooks
    x = torch.from_numpy(rgb_u8).permute(2, 0, 1).float().unsqueeze(0)
    srm = SRMConv2d(in_channels=3)
    srm.eval()
    with torch.no_grad():
        residual = srm(x)

    after_rgb = srm_residual_to_display(residual)
    before_rgb = rgb_u8.astype(np.float32) / 255.0

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 12,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    axes[0].imshow(np.clip(before_rgb, 0.0, 1.0))
    axes[0].set_title("Before (original)")
    axes[0].axis("off")

    axes[1].imshow(np.clip(after_rgb, 0.0, 1.0))
    axes[1].set_title("After (SRM noise residual, display-normalized)")
    axes[1].axis("off")

    fig.suptitle(
        "Spatial Rich Model (SRM) filter — 5×5 depthwise, [0,255] input",
        fontsize=13,
        fontweight="600",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {args.output.resolve()}")
    print(f"Source: {src}")


if __name__ == "__main__":
    main()
