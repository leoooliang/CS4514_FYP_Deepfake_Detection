"""
Generate report figures: Real vs Deepfake log-mel and LFCC via AudioPreprocessor.
Run from backend root:
  python scripts/visualize_audio_features_for_report.py --real path/real.wav --fake path/fake.wav
  python scripts/visualize_audio_features_for_report.py -r path/real.wav -f path/fake.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.preprocessing_pipelines.audio_preprocessor import AudioPreprocessor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Real vs Deepfake log-mel and LFCC (two panels each) for report figures."
        )
    )
    parser.add_argument(
        "-r",
        "--real",
        dest="real_path",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to real (bona fide) audio file",
    )
    parser.add_argument(
        "-f",
        "--fake",
        dest="fake_path",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to deepfake (spoof) audio file",
    )
    return parser.parse_args()


def _resolve_audio_path(p: Path) -> Path:
    p = p.expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f"File not found: {p}")
    return p


def _extract_mel_lfcc(
    pre: AudioPreprocessor, audio_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    mel_batch, lfcc_batch = pre.process(str(audio_path), segment=False)
    mel = mel_batch[0, 0].detach().cpu().numpy()
    lfcc = lfcc_batch[0, 0].detach().cpu().numpy()
    return mel, lfcc


def _time_axis(n_frames: int, pre: AudioPreprocessor) -> np.ndarray:
    return np.arange(n_frames) * pre.hop_length / pre.sample_rate


def main() -> None:
    args = _parse_args()
    real_path = _resolve_audio_path(args.real_path)
    fake_path = _resolve_audio_path(args.fake_path)

    out_dir = Path(__file__).resolve().parent / "report_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = AudioPreprocessor()
    mel_real, lfcc_real = _extract_mel_lfcc(pre, real_path)
    mel_fake, lfcc_fake = _extract_mel_lfcc(pre, fake_path)

    if mel_real.shape[1] != mel_fake.shape[1]:
        raise SystemExit(
            f"Time frames differ (real {mel_real.shape[1]} vs fake {mel_fake.shape[1]}); "
            "expected same segment length after preprocessing."
        )

    times = _time_axis(mel_real.shape[1], pre)
    t0, t1 = float(times[0]), float(times[-1])

    vmin_mel = min(float(mel_real.min()), float(mel_fake.min()))
    vmax_mel = max(float(mel_real.max()), float(mel_fake.max()))
    vmin_lfcc = min(float(lfcc_real.min()), float(lfcc_fake.min()))
    vmax_lfcc = max(float(lfcc_real.max()), float(lfcc_fake.max()))

    fig_mel, axes_mel = plt.subplots(
        1, 2, figsize=(12.0, 3.6), dpi=150, sharey=True, constrained_layout=True
    )
    for ax, mel, title in (
        (axes_mel[0], mel_real, "Real"),
        (axes_mel[1], mel_fake, "Deepfake"),
    ):
        im = ax.imshow(
            mel,
            aspect="auto",
            origin="lower",
            extent=(t0, t1, 0, mel.shape[0]),
            cmap="magma",
            interpolation="nearest",
            vmin=vmin_mel,
            vmax=vmax_mel,
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
    axes_mel[0].set_ylabel("Mel bin")
    fig_mel.colorbar(im, ax=axes_mel, fraction=0.035, pad=0.02)
    mel_path = out_dir / "log_mel_spectrogram_comparison.png"
    fig_mel.savefig(mel_path, dpi=300, bbox_inches="tight")
    plt.close(fig_mel)

    fig_lfcc, axes_lfcc = plt.subplots(
        1, 2, figsize=(12.0, 3.6), dpi=150, sharey=True, constrained_layout=True
    )
    for ax, lfcc, title in (
        (axes_lfcc[0], lfcc_real, "Real"),
        (axes_lfcc[1], lfcc_fake, "Deepfake"),
    ):
        im_l = ax.imshow(
            lfcc,
            aspect="auto",
            origin="lower",
            extent=(t0, t1, 0, lfcc.shape[0]),
            cmap="viridis",
            interpolation="nearest",
            vmin=vmin_lfcc,
            vmax=vmax_lfcc,
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
    axes_lfcc[0].set_ylabel("LFCC coeff.")
    fig_lfcc.colorbar(im_l, ax=axes_lfcc, fraction=0.035, pad=0.02)
    lfcc_path = out_dir / "lfcc_features_comparison.png"
    fig_lfcc.savefig(lfcc_path, dpi=300, bbox_inches="tight")
    plt.close(fig_lfcc)

    print(mel_path)
    print(lfcc_path)


if __name__ == "__main__":
    main()
