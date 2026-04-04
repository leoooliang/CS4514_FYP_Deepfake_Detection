"""
Audio preprocessing pipeline for the Audio Detection Module (Mel + LFCC).

Pipeline:
    1. Load input -> Audio waveform
    2. Voice activity detection -> trim silence
    3. Normalise amplitude -> [-1, 1]
    4. Segment into fixed-length chunks -> (N, 1, 128, T)
    5. Extract Mel-spectrogram + LFCC features
"""

import io
from pathlib import Path
from typing import List, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio
from loguru import logger

from app.core.exceptions import NoVoiceDetectedError


class AudioPreprocessor:

    def __init__(
        self,
        sample_rate: int = 16000,
        segment_duration: float = 4.0,
        n_mels: int = 128,
        n_lfcc: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        min_voice_duration: float = 0.5,
        vad_top_db: int = 30,
    ):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_samples = int(sample_rate * segment_duration)
        self.n_mels = n_mels
        self.n_lfcc = n_lfcc
        self.min_voice_duration = min_voice_duration
        self.vad_top_db = vad_top_db

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max,
        )
        self.lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=sample_rate, n_filter=n_lfcc, n_lfcc=n_lfcc,
            f_min=f_min, f_max=f_max,
            speckwargs={"n_fft": n_fft, "hop_length": hop_length},
        )

    def process(
        self, input_data: Union[str, bytes, np.ndarray], segment: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            waveform = self._load_audio(input_data)
            self._validate_voice_activity(waveform)
            waveform = self._normalize(waveform)
            segments = self._segment(waveform) if segment else [self._pad_or_trim(waveform)]

            mels, lfccs = [], []
            for seg in segments:
                t = torch.from_numpy(seg).float().unsqueeze(0)
                mels.append(torch.log(self.mel_spectrogram(t) + 1e-9))
                lfccs.append(self.lfcc_transform(t))

            return torch.stack(mels), torch.stack(lfccs)
        except NoVoiceDetectedError:
            raise
        except Exception as e:
            raise ValueError(f"Audio preprocessing error: {e}") from e

    def _load_audio(self, data: Union[str, bytes, np.ndarray]) -> np.ndarray:
        if isinstance(data, str):
            if not Path(data).exists():
                raise FileNotFoundError(f"Audio not found: {data}")
            wav, _ = librosa.load(data, sr=self.sample_rate, mono=True)
            return wav
        if isinstance(data, bytes):
            wav, _ = librosa.load(io.BytesIO(data), sr=self.sample_rate, mono=True)
            return wav
        if isinstance(data, np.ndarray):
            if data.ndim > 1:
                data = data.mean(axis=0 if data.shape[0] < data.shape[1] else 1)
            return data
        raise ValueError(f"Unsupported input type: {type(data)}")

    def _validate_voice_activity(self, waveform: np.ndarray) -> None:
        try:
            intervals = librosa.effects.split(waveform, top_db=self.vad_top_db)
            active = sum(e - s for s, e in intervals) / self.sample_rate
        except Exception as e:
            raise NoVoiceDetectedError(technical_details=f"VAD failed: {e}") from e
        if active < self.min_voice_duration:
            raise NoVoiceDetectedError(
                technical_details=f"Only {active:.2f}s voice (min {self.min_voice_duration}s)"
            )

    def _normalize(self, wav: np.ndarray) -> np.ndarray:
        mx = np.max(np.abs(wav))
        return wav / mx if mx > 0 else wav

    def _pad_or_trim(self, wav: np.ndarray) -> np.ndarray:
        n = self.segment_samples
        if len(wav) > n:
            return wav[:n]
        if len(wav) < n:
            reps = (n // len(wav)) + 1
            return np.tile(wav, reps)[:n]
        return wav

    def _segment(self, wav: np.ndarray) -> List[np.ndarray]:
        n = self.segment_samples
        if len(wav) <= n:
            return [self._pad_or_trim(wav)]
        return [self._pad_or_trim(wav[i : i + n]) for i in range(0, len(wav), n)]
