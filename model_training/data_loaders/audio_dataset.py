"""
Audio dataset for deepfake detection training.

AudioDataset: 
1. Loads audio files
2. Resamples to 16 kHz
3. Pads/truncates to 4.0 s
4. Amplitude-normalises to [-1, 1]
5. (Training only) Gaussian white noise injection
6. Extracts Log-Mel Spectrogram (128 mels) and LFCC (128 coefficients)
7. (Training only) SpecAugment — frequency & time masking
"""

import os
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset


AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')


class AudioDataset(Dataset):
    """
    Dual-stream audio dataset for deepfake detection.
    """

    def __init__(self, root_dir, target_sr=16000, target_duration=4.0, is_train=False):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)
        self.is_train = is_train

        self.samples = []
        self.targets = []

        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(AUDIO_EXTENSIONS)]
            self.samples.extend(files)
            self.targets.extend([0] * len(files))

        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(AUDIO_EXTENSIONS)]
            self.samples.extend(files)
            self.targets.extend([1] * len(files))

        print(f"Loaded {len(self.samples)} audio files from {root_dir}")
        print(f"  Fake (Class 0): {self.targets.count(0)}, Real (Class 1): {self.targets.count(1)}")

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=50.0,
            f_max=8000.0,
        )

        self.lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=target_sr,
            n_filter=128,
            n_lfcc=128,
            f_min=50.0,
            f_max=8000.0,
            speckwargs={'n_fft': 1024, 'hop_length': 512},
        )

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path = self.samples[idx]
        label = self.targets[idx]

        try:
            waveform_np, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)
            waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            waveform = torch.zeros(1, self.target_samples)

        current = waveform.shape[1]
        if current > self.target_samples:
            start = torch.randint(0, current - self.target_samples + 1, (1,)).item()
            waveform = waveform[:, start:start + self.target_samples]
        elif current < self.target_samples:
            repeats = (self.target_samples // current) + 1
            waveform = waveform.repeat(1, repeats)[:, :self.target_samples]

        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        if self.is_train:
            noise = torch.randn_like(waveform)
            noise_level = torch.empty(1).uniform_(0.001, 0.01).item()
            waveform = torch.clamp(waveform + noise_level * noise, -1.0, 1.0)

        log_mel = torch.log(self.mel_spectrogram(waveform) + 1e-9)
        lfcc = self.lfcc_transform(waveform)

        if self.is_train:
            log_mel = self.time_mask(self.freq_mask(log_mel))
            lfcc = self.time_mask(lfcc)

        return log_mel, lfcc, label
