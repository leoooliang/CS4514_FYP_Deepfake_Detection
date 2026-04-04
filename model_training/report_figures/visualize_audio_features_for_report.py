"""
Figure generator for reports:
- Figure 2.3: Visualization of Log-Mel Spectrograms for Real vs. Deepfake Audio 
- Figure 2.4: Visualization of LFCC Features for Real vs. Deepfake Audio

Output: log_mel_spectrogram_comparison.png, lfcc_features_comparison.png
(The output image is not exactly the same as the figure in the report, but the overall effect is the same.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_DIR = SCRIPT_DIR.parent
if str(MODEL_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAINING_DIR))

from configs.config import AudioTrainConfig

# Matches AudioDataset Mel/LFCC band limits (not duplicated on AudioTrainConfig).
_F_MIN = 50.0
_F_MAX = 8000.0


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
    parser.add_argument(
        "--output-mel",
        type=Path,
        default=SCRIPT_DIR / "log_mel_spectrogram_comparison.png",
        help="Output path for log-mel comparison figure.",
    )
    parser.add_argument(
        "--output-lfcc",
        type=Path,
        default=SCRIPT_DIR / "lfcc_features_comparison.png",
        help="Output path for LFCC comparison figure.",
    )
    return parser.parse_args()


def _resolve_audio_path(p: Path) -> Path:
    p = p.expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f"File not found: {p}")
    return p


def _load_waveform(path: Path, cfg: AudioTrainConfig) -> torch.Tensor:
    """Load, pad/truncate to `target_duration`, peak-normalise — eval-style (no aug)."""
    target_samples = int(cfg.target_sr * cfg.target_duration)
    try:
        waveform_np, _ = librosa.load(str(path), sr=cfg.target_sr, mono=True)
        waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
    except Exception as e:
        raise SystemExit(f"Error loading {path}: {e}") from e

    current = waveform.shape[1]
    if current > target_samples:
        waveform = waveform[:, :target_samples]
    elif current < target_samples:
        repeats = (target_samples // current) + 1
        waveform = waveform.repeat(1, repeats)[:, :target_samples]

    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


def _build_transforms(cfg: AudioTrainConfig) -> tuple[torchaudio.transforms.MelSpectrogram, torchaudio.transforms.LFCC]:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.target_sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=_F_MIN,
        f_max=_F_MAX,
    )
    lfcc = torchaudio.transforms.LFCC(
        sample_rate=cfg.target_sr,
        n_filter=cfg.n_lfcc,
        n_lfcc=cfg.n_lfcc,
        f_min=_F_MIN,
        f_max=_F_MAX,
        speckwargs={"n_fft": cfg.n_fft, "hop_length": cfg.hop_length},
    )
    return mel, lfcc


def _extract_mel_lfcc(
    waveform: torch.Tensor,
    mel_t: torchaudio.transforms.MelSpectrogram,
    lfcc_t: torchaudio.transforms.LFCC,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        log_mel = torch.log(mel_t(waveform) + 1e-9)
        lfcc = lfcc_t(waveform)
    # (1, n_mels, T) / (1, n_lfcc, T) from torchaudio
    mel = log_mel[0].detach().cpu().numpy()
    lfcc_np = lfcc[0].detach().cpu().numpy()
    return mel, lfcc_np


def _time_axis(n_frames: int, cfg: AudioTrainConfig) -> np.ndarray:
    return np.arange(n_frames) * cfg.hop_length / cfg.target_sr


def main() -> None:
    args = _parse_args()
    real_path = _resolve_audio_path(args.real_path)
    fake_path = _resolve_audio_path(args.fake_path)

    cfg = AudioTrainConfig()
    mel_t, lfcc_t = _build_transforms(cfg)

    w_real = _load_waveform(real_path, cfg)
    w_fake = _load_waveform(fake_path, cfg)
    mel_real, lfcc_real = _extract_mel_lfcc(w_real, mel_t, lfcc_t)
    mel_fake, lfcc_fake = _extract_mel_lfcc(w_fake, mel_t, lfcc_t)

    if mel_real.shape[1] != mel_fake.shape[1]:
        raise SystemExit(
            f"Time frames differ (real {mel_real.shape[1]} vs fake {mel_fake.shape[1]}); "
            "expected same segment length after preprocessing."
        )

    times = _time_axis(mel_real.shape[1], cfg)
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
    args.output_mel.parent.mkdir(parents=True, exist_ok=True)
    fig_mel.savefig(args.output_mel, dpi=300, bbox_inches="tight")
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
    args.output_lfcc.parent.mkdir(parents=True, exist_ok=True)
    fig_lfcc.savefig(args.output_lfcc, dpi=300, bbox_inches="tight")
    plt.close(fig_lfcc)

    print(args.output_mel.resolve())
    print(args.output_lfcc.resolve())


if __name__ == "__main__":
    main()
