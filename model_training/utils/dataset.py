"""
Custom dataset classes for deepfake detection.

This module contains:
- PrecomputedSRMDataset: Dataset for loading precomputed SRM noise residuals
- PrecomputedCLIPDataset: Dataset for loading precomputed CLIP-preprocessed images
- AudioDataset: Dataset for loading and processing audio files with Dual-Stream features (Mel-Spectrograms + LFCCs)
"""

import os
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset


class PrecomputedCLIPDataset(Dataset):
    """
    Custom dataset that loads precomputed CLIP-preprocessed image tensors from .pt files.
    
    The tensors are already preprocessed (resized, normalized with CLIP mean/std),
    so no transforms are applied in __getitem__.
    
    This class is moved to a separate module to avoid multiprocessing deadlocks
    that can occur when the dataset class is defined in a notebook or __main__ context.
    """
    
    def __init__(self, root_dir):
        """
        Args:
            root_dir: Path to directory containing 'fake' and 'real' subdirectories
                     with precomputed .pt files
        """
        self.samples = []
        self.targets = []
        
        # Load fake images (label = 0)
        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                         if f.endswith('.pt')]
            self.samples.extend(fake_files)
            self.targets.extend([0] * len(fake_files))
        
        # Load real images (label = 1)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                         if f.endswith('.pt')]
            self.samples.extend(real_files)
            self.targets.extend([1] * len(real_files))
        
        print(f"Loaded {len(self.samples)} precomputed CLIP tensors from {root_dir}")
        print(f"  Fake: {self.targets.count(0)}")
        print(f"  Real: {self.targets.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load precomputed CLIP tensor
        tensor_path = self.samples[idx]
        clip_tensor = torch.load(tensor_path, weights_only=True)
        label = self.targets[idx]
        
        return clip_tensor, label


class PrecomputedSRMDataset(Dataset):
    """
    Custom dataset that loads precomputed RAW RGB tensors from .pt files.
    
    The tensors are only resized and converted to tensor format (no normalization, no SRM).
    Normalization and SRM filtering happen on GPU during training for maximum speed.
    
    This class is moved to a separate module to avoid multiprocessing deadlocks
    that can occur when the dataset class is defined in a notebook or __main__ context.
    """
    
    def __init__(self, root_dir):
        """
        Args:
            root_dir: Path to directory containing 'fake' and 'real' subdirectories
                     with precomputed .pt files
        """
        self.samples = []
        self.targets = []
        
        # Load fake images (label = 0) - Fake is Class 0 for AUC calculation
        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                         if f.endswith('.pt')]
            self.samples.extend(fake_files)
            self.targets.extend([0] * len(fake_files))
        
        # Load real images (label = 1) - Real is Class 1
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                         if f.endswith('.pt')]
            self.samples.extend(real_files)
            self.targets.extend([1] * len(real_files))
        
        print(f"Loaded {len(self.samples)} precomputed RGB tensors from {root_dir}")
        print(f"  Fake (Class 0): {self.targets.count(0)}")
        print(f"  Real (Class 1): {self.targets.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load precomputed RAW RGB tensor
        tensor_path = self.samples[idx]
        rgb_tensor = torch.load(tensor_path, weights_only=True)
        label = self.targets[idx]
        
        return rgb_tensor, label


class AudioDataset(Dataset):

    """

    Custom dataset for audio deepfake detection with Dual-Stream Architecture.

   

    This class loads audio files and processes them on the fly:

    - Resamples to target sample rate (default 16kHz)

    - Converts to mono

    - Standardizes duration with wrap-around padding

    - Extracts BOTH Log-Mel Spectrograms and LFCCs (Linear Frequency Cepstral Coefficients)

   

    Dual-Stream Rationale:

    - Mel-Spectrograms: Capture low-frequency human phonetic structures

    - LFCCs: Capture high-frequency synthetic vocoder artifacts

   

    Label mapping: Fake = 0, Real = 1

    """

   

    def __init__(self, root_dir, target_sr=16000, target_duration=4.0, is_train=False):

        """

        Args:

            root_dir: Path to directory containing 'fake' and 'real' subdirectories

            target_sr: Target sample rate (default: 16000 Hz)

            target_duration: Target duration in seconds (default: 4.0)

            is_train: If True, apply data augmentation (default: False)

        """

        self.root_dir = root_dir

        self.target_sr = target_sr

        self.target_duration = target_duration

        self.target_samples = int(target_sr * target_duration)  # 64,000 samples

        self.is_train = is_train

       

        self.samples = []

        self.targets = []

       

        # Load fake audio files (label = 0)

        fake_dir = os.path.join(root_dir, 'fake')

        if os.path.exists(fake_dir):

            fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)

                         if f.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]

            self.samples.extend(fake_files)

            self.targets.extend([0] * len(fake_files))

       

        # Load real audio files (label = 1)

        real_dir = os.path.join(root_dir, 'real')

        if os.path.exists(real_dir):

            real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)

                         if f.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]

            self.samples.extend(real_files)

            self.targets.extend([1] * len(real_files))

       

        print(f"Loaded {len(self.samples)} audio files from {root_dir}")

        print(f"  Fake (Class 0): {self.targets.count(0)}")

        print(f"  Real (Class 1): {self.targets.count(1)}")

       

        # DUAL-STREAM FEATURE EXTRACTORS

        # Both transforms use the same n_fft and hop_length to ensure time dimensions match

       

        # Initialize Mel-Spectrogram transform (captures low-frequency phonetic structures)

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(

            sample_rate=target_sr,

            n_fft=1024,

            hop_length=512,

            n_mels=128,

            f_min=50.0,

            f_max=8000.0

        )

       

        # Initialize LFCC transform (captures high-frequency vocoder artifacts)

        self.lfcc_transform = torchaudio.transforms.LFCC(

            sample_rate=target_sr,

            n_filter=128,

            n_lfcc=128,

            f_min=50.0,

            f_max=8000.0,

            speckwargs={'n_fft': 1024, 'hop_length': 512} # Pass them in a dict!

        )

       

        # Initialize SpecAugment transforms for data augmentation

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)

        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

   

    def __len__(self):

        return len(self.samples)

   

    def __getitem__(self, idx):

        audio_path = self.samples[idx]

        label = self.targets[idx]

       

        try:

            # OPTIMIZATION: Let librosa handle resampling and mono conversion instantly!

            # By passing sr=self.target_sr and mono=True, it returns a 1D array at 16kHz

            waveform_np, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

           

            # Convert numpy array to torch tensor and add the channel dimension

            # Shape goes from (Samples,) to (1, Samples)

            waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)

               

        except Exception as e:

            print(f"Error loading {audio_path}: {e}")

            # If a file is totally broken, return a silent tensor as a fallback

            waveform = torch.zeros(1, self.target_samples)

       

        # Standardize to exactly target_samples (e.g., 64,000 samples for 4.0 seconds)

        current_samples = waveform.shape[1]
        
       

        if current_samples > self.target_samples:

            # Randomly trim to target length

            start = torch.randint(0, current_samples - self.target_samples + 1, (1,)).item()

            waveform = waveform[:, start:start + self.target_samples]

           

        elif current_samples < self.target_samples:

            # Wrap-around padding (tiling/repeating)

            repeats = (self.target_samples // current_samples) + 1

            waveform = waveform.repeat(1, repeats)

            waveform = waveform[:, :self.target_samples]

           

        # Amplitude normalization to [-1.0, 1.0]

        max_val = torch.max(torch.abs(waveform))

        if max_val > 0:

            waveform = waveform / max_val

       

        # Apply Gaussian White Noise injection during training

        if self.is_train:

            noise = torch.randn_like(waveform)

            noise_level = torch.empty(1).uniform_(0.001, 0.01).item()

            waveform = waveform + noise_level * noise

            waveform = torch.clamp(waveform, -1.0, 1.0)

       

        # === DUAL-STREAM FEATURE EXTRACTION ===

       

        # Stream 1: Extract Mel-Spectrogram (captures low-frequency phonetic structures)

        mel_spec = self.mel_spectrogram(waveform)

        log_mel_spec = torch.log(mel_spec + 1e-9)  # Add small epsilon to avoid log(0)

       

        # Stream 2: Extract LFCC (captures high-frequency vocoder artifacts)

        lfcc = self.lfcc_transform(waveform)  # Shape: (1, n_lfcc, time_frames)

       

        # Apply SpecAugment during training to both features

        if self.is_train:

            log_mel_spec = self.freq_mask(log_mel_spec)

            log_mel_spec = self.time_mask(log_mel_spec)

            lfcc = self.time_mask(lfcc)

       

        # Return tuple: (mel_spectrogram, lfcc, label)

        return log_mel_spec, lfcc, label
