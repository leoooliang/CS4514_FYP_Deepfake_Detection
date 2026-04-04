"""
Video preprocessing pipeline for the Video Detection Module (Image Stream + Audio Stream + Sync Stream).

Pipeline:
    1. Load input -> Video file
    2. Extract frames -> (N, 3, 224, 224)
    3. Face detection -> crop with 15% margin (fail-fast if no face)
    4. Extract audio -> 16 kHz mono waveform
    5. Voice activity detection -> trim silence
    6. Normalise amplitude -> [-1, 1]
    7. Segment into fixed-length chunks -> (N, 1, 128, T)
    8. Extract Mel-spectrogram + LFCC features
"""

import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Tuple

import cv2
import librosa
import numpy as np
import torch
import torchaudio
from loguru import logger

from app.core.exceptions import NoFaceDetectedError, NoVoiceDetectedError

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logger.warning("facenet-pytorch not available, face detection disabled")


def _librosa_load(path: str, sr: int, mono: bool = True):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"librosa\.core\.audio")
        return librosa.load(path, sr=sr, mono=mono)


class VideoPreprocessor:

    def __init__(
        self,
        device: str = "cpu",
        num_frames: int = 15,
        target_size: Tuple[int, int] = (224, 224),
        face_margin: float = 0.15,
        center_crop_ratio: float = 0.50,
        audio_sample_rate: int = 16000,
        audio_duration: float = 4.0,
        n_mels: int = 128,
        n_lfcc: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        min_voice_duration: float = 0.5,
        vad_top_db: int = 30,
        min_face_detection_ratio: float = 0.5,
    ):
        self.device = torch.device(device)
        self.num_frames = num_frames
        self.target_size = target_size
        self.face_margin = face_margin
        self.center_crop_ratio = center_crop_ratio

        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.audio_samples = int(audio_sample_rate * audio_duration)
        self.min_voice_duration = min_voice_duration
        self.vad_top_db = vad_top_db
        self.min_face_detection_ratio = min_face_detection_ratio

        self.mtcnn = (
            MTCNN(keep_all=False, post_process=False, device=self.device)
            if MTCNN_AVAILABLE
            else None
        )

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max,
        )
        self.lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=audio_sample_rate, n_filter=n_lfcc, n_lfcc=n_lfcc,
            f_min=f_min, f_max=f_max,
            speckwargs={"n_fft": n_fft, "hop_length": hop_length},
        )

    def process(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            if not isinstance(video_path, str):
                raise ValueError("VideoPreprocessor requires a file path")

            raw_frames = self._extract_frames(video_path)
            face_crops = self._detect_and_crop_faces(raw_frames)

            while len(face_crops) < self.num_frames:
                face_crops.append(face_crops[-1])
            face_crops = face_crops[: self.num_frames]

            faces = torch.from_numpy(np.array(face_crops)).float().permute(0, 3, 1, 2) / 255.0

            waveform = self._extract_audio(video_path)
            wt = torch.from_numpy(waveform).float().unsqueeze(0)
            mel = torch.log(self.mel_spectrogram(wt) + 1e-9)
            lfcc = self.lfcc_transform(wt)

            return faces, mel, lfcc
        except (NoFaceDetectedError, NoVoiceDetectedError):
            raise
        except Exception as e:
            raise ValueError(f"Video preprocessing error: {e}") from e

    def _extract_frames(self, path: str) -> List[np.ndarray]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Video not found: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = (
            list(range(total))
            if total < self.num_frames
            else np.linspace(0, total - 1, self.num_frames, dtype=int)
        )

        frames: List[np.ndarray] = []
        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()

        if not frames:
            raise ValueError("No frames extracted from video")
        return frames

    def _detect_and_crop_faces(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if self.mtcnn is None:
            raise NoFaceDetectedError(
                user_message="Unable to process video.",
                technical_details="MTCNN not installed.",
            )

        crops: List[np.ndarray] = []
        last_box = None
        detections = 0

        for frame in frames:
            try:
                boxes, probs = self.mtcnn.detect(frame)
                if boxes is not None and len(boxes) > 0:
                    idx = int(np.argmax(probs)) if len(boxes) > 1 else 0
                    last_box = boxes[idx].copy()
                    detections += 1
                bbox = last_box
            except Exception:
                bbox = last_box

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                fh, fw = frame.shape[:2]
                c = (
                    int(max(0, x1 - w * self.face_margin)),
                    int(max(0, y1 - h * self.face_margin)),
                    int(min(fw, x2 + w * self.face_margin)),
                    int(min(fh, y2 + h * self.face_margin)),
                )
                crop = cv2.resize(frame[c[1]:c[3], c[0]:c[2]], self.target_size, interpolation=cv2.INTER_LINEAR)
                crops.append(crop)

        ratio = detections / len(frames)
        if ratio < self.min_face_detection_ratio or not crops:
            raise NoFaceDetectedError(
                user_message="No face detected in the provided video.",
                technical_details=f"{ratio:.0%} frames had faces (min {self.min_face_detection_ratio:.0%}).",
            )
        return crops

    def _extract_audio(self, video_path: str) -> np.ndarray:
        waveform = self._try_load_audio(video_path)

        self._validate_voice(waveform)

        if len(waveform) > self.audio_samples:
            waveform = waveform[: self.audio_samples]
        elif len(waveform) < self.audio_samples:
            waveform = np.tile(waveform, (self.audio_samples // len(waveform)) + 1)[: self.audio_samples]

        mx = np.max(np.abs(waveform))
        if mx > 0:
            waveform = waveform / mx
        return waveform

    def _try_load_audio(self, path: str) -> np.ndarray:
        try:
            wav, _ = _librosa_load(path, sr=self.audio_sample_rate, mono=True)
            if wav is not None and len(wav) > 0:
                return wav
        except Exception:
            pass

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", path, "-map", "a:0", "-vn",
                    "-acodec", "pcm_s16le", "-ac", "1",
                    "-ar", str(self.audio_sample_rate), tmp.name,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    stderr = (proc.stderr or "").lower()
                    if any(m in stderr for m in ("does not contain any stream", "matches no streams")):
                        raise NoVoiceDetectedError(
                            user_message="No voice detected in the provided video.",
                            technical_details="Video has no audio stream.",
                        )
                    raise RuntimeError(f"ffmpeg error: {proc.stderr}")
                wav, _ = _librosa_load(tmp.name, sr=self.audio_sample_rate, mono=True)
                if wav is not None and len(wav) > 0:
                    return wav
        except NoVoiceDetectedError:
            raise
        except Exception:
            pass

        raise NoVoiceDetectedError(
            user_message="Unable to process video audio.",
            technical_details="All audio decoders failed.",
        )

    def _validate_voice(self, waveform: np.ndarray) -> None:
        try:
            intervals = librosa.effects.split(waveform, top_db=self.vad_top_db)
            active = sum(e - s for s, e in intervals) / self.audio_sample_rate
        except Exception as e:
            raise NoVoiceDetectedError(technical_details=f"VAD failed: {e}") from e
        if active < self.min_voice_duration:
            raise NoVoiceDetectedError(
                user_message="No voice detected in the provided video.",
                technical_details=f"Only {active:.2f}s voice (min {self.min_voice_duration}s).",
            )
