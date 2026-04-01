"""
============================================================================
Audio Preprocessing Pipeline for Dual-Stream Audio Deepfake Detection
============================================================================
This module provides a robust, reusable audio preprocessing pipeline for the
dual-stream (Mel-Spectrogram + LFCC) audio deepfake detection architecture.

Key Features:
    1. Waveform Loading & Normalization
       - Loads audio at 16kHz, mono
       - Standardizes duration to 4.0 seconds (64,000 samples)
       - Wrap-around padding for short clips
       - Amplitude normalization to [-1, 1]
       
    2. Dual-Stream Feature Extraction:
       - Stream A (Mel-Spectrogram): Captures low-frequency phonetic patterns
       - Stream B (LFCC): Captures high-frequency vocoder artifacts [NOVELTY]
       
    3. Robust Input Handling:
       - Supports: str (file path), np.ndarray (waveform), bytes (audio data)
       - Automatic resampling and format conversion
       
    4. Segmentation Support:
       - Splits long audio into fixed-duration segments
       - Enables batch processing for long audio files

Technical Details:
    - Sample Rate: 16kHz (mono)
    - Segment Duration: 4.0 seconds (64,000 samples)
    - Mel-Spectrogram: n_fft=1024, hop_length=512, n_mels=128, f_min=50, f_max=8000
    - LFCC: n_filter=128, n_lfcc=128, f_min=50, f_max=8000
    - Padding Strategy: Wrap-around (tiling) for short clips
    - Normalization: Amplitude normalization to [-1, 1]

Author: Senior ML Engineer
Date: 2026-03-24
============================================================================
"""

import io
from typing import Union, Tuple, List, Optional
from pathlib import Path

import torch
import torchaudio
import numpy as np
import librosa
from loguru import logger

from app.core.exceptions import NoVoiceDetectedError


class AudioPreprocessor:
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║           Audio Preprocessing Pipeline for Dual-Stream               ║
    ║              Mel-Spectrogram + LFCC Feature Extraction               ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Robust audio loading and dual-feature extraction for deepfake detection.
    
    Pipeline Overview:
        1. Load Audio → 16kHz mono waveform
        2. Standardize Duration → 4.0 seconds (wrap-around padding)
        3. Amplitude Normalize → [-1, 1]
        4. Segment (optional) → Multiple 4s segments for long audio
        5. Extract Features → (Mel-Spectrogram, LFCC)
        
    Stream A (Mel-Spectrogram):
        - torchaudio.transforms.MelSpectrogram
        - n_mels=128, n_fft=1024, hop_length=512
        - Log-scale transformation
        - Shape: (1, 128, time_frames)
        
    Stream B (LFCC):
        - torchaudio.transforms.LFCC
        - n_lfcc=128, n_filter=128, n_fft=1024, hop_length=512
        - Captures high-frequency vocoder artifacts
        - Shape: (1, 128, time_frames)
        
    Example:
        >>> preprocessor = AudioPreprocessor()
        >>> mel_batch, lfcc_batch = preprocessor.process("audio.wav")
        >>> print(mel_batch.shape, lfcc_batch.shape)
        torch.Size([num_segments, 1, 128, T]) torch.Size([num_segments, 1, 128, T])
    """
    
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
        vad_top_db: int = 30
    ):
        """
        Initialize AudioPreprocessor with dual-stream feature extractors.
        
        Args:
            sample_rate: Target sample rate in Hz (default: 16000)
            segment_duration: Duration of each segment in seconds (default: 4.0)
            n_mels: Number of mel frequency bins (default: 128)
            n_lfcc: Number of LFCC coefficients (default: 128)
            n_fft: FFT window size (default: 1024)
            hop_length: Hop length for STFT (default: 512)
            f_min: Minimum frequency for filterbank (default: 50.0 Hz)
            f_max: Maximum frequency for filterbank (default: 8000.0 Hz)
            min_voice_duration: Minimum voice activity duration in seconds (default: 0.5)
            vad_top_db: Top dB threshold for VAD (default: 30)
        """
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_samples = int(sample_rate * segment_duration)
        self.n_mels = n_mels
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.min_voice_duration = min_voice_duration
        self.vad_top_db = vad_top_db
        
        # =====================================================================
        # Stream A: Mel-Spectrogram Extractor
        # =====================================================================
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
        
        # =====================================================================
        # Stream B: LFCC Extractor
        # =====================================================================
        self.lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_filter=n_lfcc,
            n_lfcc=n_lfcc,
            f_min=f_min,
            f_max=f_max,
            speckwargs={'n_fft': n_fft, 'hop_length': hop_length}
        )
        
        logger.info(
            f"AudioPreprocessor initialized:\n"
            f"  Sample rate: {sample_rate}Hz\n"
            f"  Segment duration: {segment_duration}s ({self.segment_samples} samples)\n"
            f"  Stream A: Mel-Spectrogram ({n_mels} bins)\n"
            f"  Stream B: LFCC ({n_lfcc} coefficients) [NOVELTY]\n"
            f"  FFT: n_fft={n_fft}, hop_length={hop_length}\n"
            f"  Frequency range: [{f_min}, {f_max}] Hz\n"
            f"  VAD: min_voice_duration={min_voice_duration}s, top_db={vad_top_db}"
        )
    
    def _validate_voice_activity(self, waveform: np.ndarray) -> None:
        """
        Validate that audio contains sufficient voice activity using VAD.
        
        FAIL-FAST: Raises NoVoiceDetectedError if insufficient voice activity.
        
        Args:
            waveform: Audio waveform at target sample rate
            
        Raises:
            NoVoiceDetectedError: If total voice activity is below threshold
        """
        try:
            # Use librosa to detect non-silent intervals
            # top_db: threshold in dB below reference to consider as silence
            intervals = librosa.effects.split(waveform, top_db=self.vad_top_db)
            
            # Calculate total duration of active (non-silent) audio
            total_active_samples = sum(end - start for start, end in intervals)
            total_active_duration = total_active_samples / self.sample_rate
            
            logger.debug(
                f"Voice Activity Detection: "
                f"{len(intervals)} intervals, "
                f"total active duration: {total_active_duration:.2f}s "
                f"(threshold: {self.min_voice_duration}s)"
            )
            
            # FAIL-FAST: Reject audio without sufficient voice activity
            if total_active_duration < self.min_voice_duration:
                logger.warning(
                    f"Insufficient voice activity detected: "
                    f"{total_active_duration:.2f}s < {self.min_voice_duration}s"
                )
                raise NoVoiceDetectedError(
                    technical_details=(
                        f"Detected only {total_active_duration:.2f}s of voice activity "
                        f"(minimum required: {self.min_voice_duration}s)."
                    )
                )
            
            logger.debug(f"✓ Voice activity validation passed: {total_active_duration:.2f}s")
            
        except NoVoiceDetectedError:
            # Re-raise NoVoiceDetectedError as-is
            raise
        except Exception as e:
            logger.error(f"Voice activity detection failed: {str(e)}")
            raise NoVoiceDetectedError(
                user_message="Unable to process audio. Please try again later.",
                technical_details=f"Voice activity detection failed: {str(e)}"
            ) from e
    
    def _load_audio(self, input_data: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """
        Load and convert audio to mono waveform at target sample rate.
        
        Args:
            input_data: Audio in supported format
            
        Returns:
            np.ndarray: Mono waveform at target sample rate
            
        Raises:
            ValueError: If input type is unsupported
            FileNotFoundError: If file path doesn't exist
        """
        try:
            # Case 1: File path (string)
            if isinstance(input_data, str):
                file_path = Path(input_data)
                if not file_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {input_data}")
                
                waveform, sr = librosa.load(
                    input_data,
                    sr=self.sample_rate,
                    mono=True
                )
                logger.debug(
                    f"Loaded audio from path: {input_data} "
                    f"(duration: {len(waveform)/self.sample_rate:.2f}s)"
                )
                return waveform
            
            # Case 2: Bytes (audio file data)
            elif isinstance(input_data, bytes):
                with io.BytesIO(input_data) as bio:
                    waveform, sr = librosa.load(
                        bio,
                        sr=self.sample_rate,
                        mono=True
                    )
                logger.debug(
                    f"Loaded audio from bytes "
                    f"(duration: {len(waveform)/self.sample_rate:.2f}s)"
                )
                return waveform
            
            # Case 3: NumPy array (waveform)
            elif isinstance(input_data, np.ndarray):
                waveform = input_data
                
                # Ensure 1D
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=0 if waveform.shape[0] < waveform.shape[1] else 1)
                
                # Resample if needed (using librosa)
                if hasattr(self, '_source_sr'):
                    waveform = librosa.resample(
                        waveform,
                        orig_sr=self._source_sr,
                        target_sr=self.sample_rate
                    )
                
                logger.debug(
                    f"Using NumPy waveform "
                    f"(duration: {len(waveform)/self.sample_rate:.2f}s)"
                )
                return waveform
            
            else:
                raise ValueError(
                    f"Unsupported input type: {type(input_data)}. "
                    "Expected: str (file path), bytes, or np.ndarray"
                )
                
        except Exception as e:
            logger.error(f"Failed to load audio: {str(e)}")
            raise
    
    def _standardize_duration(self, waveform: np.ndarray) -> np.ndarray:
        """
        Standardize waveform to target duration using wrap-around padding.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            np.ndarray: Waveform of exactly segment_samples length
        """
        current_samples = len(waveform)
        
        if current_samples > self.segment_samples:
            # Trim to target length (take first segment)
            waveform = waveform[:self.segment_samples]
            logger.debug(
                f"Trimmed audio: {current_samples} → {self.segment_samples} samples"
            )
        
        elif current_samples < self.segment_samples:
            # Wrap-around padding (tiling)
            repeats = (self.segment_samples // current_samples) + 1
            waveform = np.tile(waveform, repeats)[:self.segment_samples]
            logger.debug(
                f"Applied wrap-around padding: {current_samples} → {self.segment_samples} samples"
            )
        
        return waveform
    
    def _normalize_amplitude(self, waveform: np.ndarray) -> np.ndarray:
        """
        Normalize waveform amplitude to [-1, 1] range.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            np.ndarray: Amplitude-normalized waveform
        """
        max_val = np.max(np.abs(waveform))
        
        if max_val > 0:
            waveform = waveform / max_val
            logger.debug(f"Normalized amplitude: max={max_val:.3f} → [-1, 1]")
        else:
            logger.warning("Silent audio detected (max amplitude = 0)")
        
        return waveform
    
    def _segment_audio(self, waveform: np.ndarray) -> List[np.ndarray]:
        """
        Split audio into fixed-duration segments.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            List[np.ndarray]: List of audio segments (each segment_samples long)
        """
        segments = []
        
        # If audio is shorter than one segment, return single standardized segment
        if len(waveform) <= self.segment_samples:
            segments.append(self._standardize_duration(waveform))
        else:
            # Split into non-overlapping segments
            for start in range(0, len(waveform), self.segment_samples):
                segment = waveform[start:start + self.segment_samples]
                
                # Standardize each segment (applies padding if needed)
                segment = self._standardize_duration(segment)
                segments.append(segment)
        
        logger.debug(f"Split audio into {len(segments)} segments")
        return segments
    
    def _extract_mel_spectrogram(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Extract log-Mel spectrogram from waveform.
        
        Args:
            waveform: Audio waveform (segment_samples long)
            
        Returns:
            torch.Tensor: Log-Mel spectrogram of shape (1, n_mels, time_frames)
        """
        # Convert to torch tensor
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, samples)
        
        # Extract Mel-Spectrogram
        mel_spec = self.mel_spectrogram(waveform_tensor)  # (1, n_mels, time_frames)
        
        # Convert to log scale
        log_mel_spec = torch.log(mel_spec + 1e-9)  # Add epsilon to avoid log(0)
        
        return log_mel_spec
    
    def _extract_lfcc(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Extract LFCC features from waveform.
        
        Args:
            waveform: Audio waveform (segment_samples long)
            
        Returns:
            torch.Tensor: LFCC features of shape (1, n_lfcc, time_frames)
        """
        # Convert to torch tensor
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, samples)
        
        # Extract LFCC
        lfcc = self.lfcc_transform(waveform_tensor)  # (1, n_lfcc, time_frames)
        
        return lfcc
    
    def process(
        self,
        input_data: Union[str, bytes, np.ndarray],
        segment: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete preprocessing pipeline: Load → Segment → Extract Features.
        
        This is the main entry point for preprocessing. It performs:
            1. Load audio file or waveform
            2. Segment into fixed-duration chunks (if segment=True)
            3. Normalize amplitude to [-1, 1]
            4. Extract Mel-Spectrogram and LFCC features
            
        Args:
            input_data: Audio in supported format (str, bytes, np.ndarray)
            segment: If True, split long audio into segments (default: True)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mel_batch, lfcc_batch)
                - mel_batch: (num_segments, 1, 128, T) - Log-Mel Spectrograms
                - lfcc_batch: (num_segments, 1, 128, T) - LFCC features
                
        Raises:
            ValueError: If input processing fails
            FileNotFoundError: If file path doesn't exist
            
        Example:
            >>> preprocessor = AudioPreprocessor()
            >>> mel_batch, lfcc_batch = preprocessor.process("audio.wav")
            >>> print(f"Mel: {mel_batch.shape}, LFCC: {lfcc_batch.shape}")
            Mel: torch.Size([2, 1, 128, 125]) LFCC: torch.Size([2, 1, 128, 125])
        """
        try:
            logger.debug("Starting audio preprocessing pipeline...")
            
            # Step 1: Load audio
            waveform = self._load_audio(input_data)
            
            # Step 2: FAIL-FAST Voice Activity Detection (VAD)
            self._validate_voice_activity(waveform)
            
            # Step 3: Normalize amplitude
            waveform = self._normalize_amplitude(waveform)
            
            # Step 4: Segment audio
            if segment:
                segments = self._segment_audio(waveform)
            else:
                # Single segment (standardize to target duration)
                segments = [self._standardize_duration(waveform)]
            
            # Step 5: Extract features for each segment
            mel_spectrograms = []
            lfcc_features = []
            
            for seg in segments:
                # Stream A: Extract Mel-Spectrogram
                mel_spec = self._extract_mel_spectrogram(seg)
                mel_spectrograms.append(mel_spec)
                
                # Stream B: Extract LFCC
                lfcc = self._extract_lfcc(seg)
                lfcc_features.append(lfcc)
            
            # Step 6: Stack into batch tensors
            mel_batch = torch.stack(mel_spectrograms)  # (num_segments, 1, 128, T)
            lfcc_batch = torch.stack(lfcc_features)    # (num_segments, 1, 128, T)
            
            logger.debug(
                f"Preprocessing complete:\n"
                f"  Mel-Spectrogram batch: shape={mel_batch.shape}, "
                f"range=[{mel_batch.min():.3f}, {mel_batch.max():.3f}]\n"
                f"  LFCC batch: shape={lfcc_batch.shape}, "
                f"range=[{lfcc_batch.min():.3f}, {lfcc_batch.max():.3f}]\n"
                f"  Number of segments: {len(segments)}"
            )
            
            return mel_batch, lfcc_batch
            
        except NoVoiceDetectedError:
            # Re-raise validation errors as-is (user-friendly messages)
            raise
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise ValueError(f"Preprocessing error: {str(e)}") from e


# =============================================================================
# Utility Functions
# =============================================================================

def preprocess_audio_dual_stream(
    input_data: Union[str, bytes, np.ndarray],
    segment: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function for dual-stream audio preprocessing.
    
    Args:
        input_data: Audio in supported format
        segment: If True, split long audio into segments
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (mel_batch, lfcc_batch)
        
    Example:
        >>> from app.preprocessing_pipelines.audio_preprocessor import preprocess_audio_dual_stream
        >>> mel, lfcc = preprocess_audio_dual_stream("audio.wav")
    """
    preprocessor = AudioPreprocessor()
    return preprocessor.process(input_data, segment=segment)


# =============================================================================
# Testing / Validation
# =============================================================================

if __name__ == "__main__":
    """
    Test AudioPreprocessor with dummy inputs.
    """
    logger.info("=== Testing AudioPreprocessor ===")
    
    # Test 1: Short audio (needs padding)
    logger.info("\n[Test 1] Short Audio (needs wrap-around padding)")
    short_audio = np.random.randn(32000)  # 2 seconds at 16kHz
    
    preprocessor = AudioPreprocessor()
    mel_batch, lfcc_batch = preprocessor.process(short_audio, segment=False)
    
    logger.info(f"✓ Mel batch: shape={mel_batch.shape}")
    logger.info(f"✓ LFCC batch: shape={lfcc_batch.shape}")
    
    assert mel_batch.shape[0] == 1, "Should have 1 segment!"
    assert mel_batch.shape[1] == 1, "Should have 1 channel!"
    assert mel_batch.shape[2] == 128, "Should have 128 mel bins!"
    assert lfcc_batch.shape[0] == 1, "Should have 1 segment!"
    assert lfcc_batch.shape[1] == 1, "Should have 1 channel!"
    assert lfcc_batch.shape[2] == 128, "Should have 128 LFCC coefficients!"
    
    logger.success("✓ Short audio test passed!\n")
    
    # Test 2: Long audio (needs segmentation)
    logger.info("[Test 2] Long Audio (needs segmentation)")
    long_audio = np.random.randn(160000)  # 10 seconds at 16kHz
    
    mel_batch2, lfcc_batch2 = preprocessor.process(long_audio, segment=True)
    
    logger.info(f"✓ Mel batch: shape={mel_batch2.shape}")
    logger.info(f"✓ LFCC batch: shape={lfcc_batch2.shape}")
    
    expected_segments = int(np.ceil(len(long_audio) / preprocessor.segment_samples))
    assert mel_batch2.shape[0] == expected_segments, f"Should have {expected_segments} segments!"
    assert lfcc_batch2.shape[0] == expected_segments, f"Should have {expected_segments} segments!"
    
    logger.success("✓ Long audio test passed!\n")
    
    logger.success("=== All AudioPreprocessor Tests Passed! ===")
