"""
Video dataset for deepfake detection training.

TriStreamVideoDataset: 
1. Loads precomputed .pt tensors (15 face crops + audio) for the Spatial Stream
2. Extracts dual-stream audio features (Log-Mel + LFCC) for the Audio Stream
3. Extracts sync features for the Sync Stream (CLIP ViT-B/32 + Cross-Attention + BiLSTM)
"""

import io
import torch
import torchaudio
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class TriStreamVideoDataset(Dataset):
    """
    Dataset for the Video Detection Module (Spatial Stream + Audio Stream + Sync Stream).
    """

    def __init__(self, root_dir, is_train=False, target_sr=16000):
        self.root_dir = root_dir
        self.is_train = is_train
        self.samples = []

        fake_dir = Path(root_dir) / 'fake'
        if fake_dir.exists():
            self.samples.extend([(f, 0) for f in sorted(fake_dir.glob('*.pt'))])

        real_dir = Path(root_dir) / 'real'
        if real_dir.exists():
            self.samples.extend([(f, 1) for f in sorted(real_dir.glob('*.pt'))])

        print(f"Loaded {len(self.samples)} samples from {root_dir}")
        print(f"  Fake (Class 0): {sum(1 for _, l in self.samples if l == 0)}")
        print(f"  Real (Class 1): {sum(1 for _, l in self.samples if l == 1)}")

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_fft=1024, hop_length=512,
            n_mels=128, f_min=50.0, f_max=8000.0,
        )
        self.lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=target_sr, n_filter=128, n_lfcc=128,
            f_min=50.0, f_max=8000.0,
            speckwargs={'n_fft': 1024, 'hop_length': 512},
        )

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.samples)

    def _augment_visual(self, visual):
        """
        Spatial augmentations for the Image Stream.
        """
        
        if torch.rand(1).item() < 0.5:
            visual = torch.flip(visual, dims=[3])

        if torch.rand(1).item() < 0.2:
            bfactor = 1.0 + (torch.rand(1).item() * 0.4 - 0.2)
            cfactor = 1.0 + (torch.rand(1).item() * 0.4 - 0.2)
            visual = TF.adjust_brightness(visual, bfactor)
            visual = TF.adjust_contrast(visual, cfactor)
            visual = visual.clamp(0.0, 1.0)

        if torch.rand(1).item() < 0.2:
            quality = int(torch.empty(1).uniform_(50, 100).item())
            frames = []
            for t in range(visual.shape[0]):
                pil = TF.to_pil_image(visual[t])
                buf = io.BytesIO()
                pil.save(buf, format='JPEG', quality=quality)
                buf.seek(0)
                frames.append(TF.to_tensor(Image.open(buf)))
            visual = torch.stack(frames)

        if torch.rand(1).item() < 0.2:
            sigma = torch.empty(1).uniform_(0.1, 2.0).item()
            visual = TF.gaussian_blur(visual, kernel_size=[3, 3], sigma=sigma)

        return visual

    def _augment_audio_waveform(self, audio):
        """
        Gaussian noise on raw waveform for the Audio Stream.
        """
        
        noise = torch.randn_like(audio)
        level = torch.empty(1).uniform_(0.001, 0.01).item()
        return torch.clamp(audio + level * noise, -1.0, 1.0)

    def _specaugment_mel(self, log_mel):
        """
        SpecAugment for Log-Mel for the Audio Stream.
        """

        log_mel = self.freq_mask(log_mel)
        log_mel = self.time_mask(log_mel)
        return log_mel

    def _specaugment_lfcc(self, lfcc):
        """
        SpecAugment for LFCC for the Audio Stream.
        """

        lfcc = self.time_mask(lfcc)
        return lfcc

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        visual = data['visual']
        audio = data['audio']

        if self.is_train:
            visual = self._augment_visual(visual)
            audio = self._augment_audio_waveform(audio)

        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        audio = audio.unsqueeze(0)
        log_mel = torch.log(self.mel_spectrogram(audio) + 1e-9)
        lfcc = self.lfcc_transform(audio)

        if self.is_train:
            log_mel = self._specaugment_mel(log_mel)
            lfcc = self._specaugment_lfcc(lfcc)

        return {
            'visual': visual,
            'mel': log_mel,
            'lfcc': lfcc,
            'label': label,
        }
