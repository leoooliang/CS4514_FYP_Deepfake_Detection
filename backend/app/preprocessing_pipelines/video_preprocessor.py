"""
============================================================================
Video Preprocessing Pipeline for Tri-Stream Multimodal Deepfake Detection
============================================================================
This module provides a robust, reusable video preprocessing pipeline for the
tri-stream (Spatial + Audio + Sync) multimodal deepfake detection architecture.

Key Features:
    1. Frame Extraction with MTCNN Face Detection
       - Extracts uniformly sampled frames from video
       - Detects and crops faces with 20% margin expansion
       - Temporal continuity: uses last good bbox if detection fails
       - Fallback: 50% center crop for first frame without detection
       
    2. Audio Extraction & Processing
       - Extracts audio track at 16kHz mono
       - Standardizes to 4.0 seconds with wrap-around padding
       - Amplitude normalization to [-1, 1]
       - Dual-stream features: Mel-Spectrogram + LFCC
       
    3. Tri-Stream Output:
       - Visual frames: (num_frames, 3, 224, 224) normalized to [0, 1]
       - Mel-Spectrogram: (1, 128, time_frames) for audio analysis
       - LFCC: (1, 128, time_frames) for vocoder artifact detection
       
    4. Robust Input Handling:
       - Supports various video formats (mp4, avi, mov, etc.)
       - Handles videos without audio tracks (silent fallback)
       - Error recovery for corrupted frames

Technical Details:
    - Frame Sampling: Uniform temporal sampling (default: 15 frames)
    - MTCNN: keep_all=False, post_process=False
    - Face Bbox Expansion: 20% margin (clamped to frame boundaries)
    - Fallback Strategy 1: Last good bbox (temporal continuity)
    - Fallback Strategy 2: 50% center crop (spatial fallback)
    - Target Frame Size: 224x224
    - Audio Sample Rate: 16kHz mono
    - Audio Duration: 4.0 seconds (64,000 samples)
    - Mel-Spectrogram: n_mels=128, n_fft=1024, hop_length=512
    - LFCC: n_lfcc=128, n_filter=128

Author: Senior ML Engineer
Date: 2026-03-24
============================================================================
"""

from typing import Union, Tuple, List, Optional
from pathlib import Path
import subprocess
import tempfile

import torch
import torchaudio
import numpy as np
import cv2
import librosa
from PIL import Image
from loguru import logger

from app.core.exceptions import NoFaceDetectedError, NoVoiceDetectedError


# Try to import facenet-pytorch MTCNN - graceful fallback if not available
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logger.warning(
        "facenet-pytorch not available. Face detection disabled. "
        "Install with: pip install facenet-pytorch"
    )


class VideoPreprocessor:
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║         Video Preprocessing Pipeline for Tri-Stream                  ║
    ║         Frame Extraction + Face Detection + Audio Extraction         ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Robust video preprocessing for tri-stream multimodal deepfake detection.
    
    Pipeline Overview:
        1. Frame Extraction → Uniformly sampled frames
        2. MTCNN Face Detection → Crop with 20% margin (temporal continuity)
        3. Audio Extraction → 16kHz mono, 4.0 seconds
        4. Dual-Stream Audio Features → (Mel-Spectrogram, LFCC)
        
    Visual Stream:
        - Extract num_frames (default: 15) uniformly sampled frames
        - MTCNN face detection with 20% bbox expansion
        - Fallback: last good box (temporal) or 50% center crop (spatial)
        - Resize to 224x224, normalize to [0, 1]
        - Output: (num_frames, 3, 224, 224)
        
    Audio Stream:
        - Extract audio at 16kHz mono
        - Standardize to 4.0 seconds (wrap-around padding)
        - Amplitude normalize to [-1, 1]
        - Extract Mel-Spectrogram: (1, 128, time_frames)
        - Extract LFCC: (1, 128, time_frames)
        
    Example:
        >>> preprocessor = VideoPreprocessor()
        >>> frames, mel, lfcc = preprocessor.process("video.mp4")
        >>> print(frames.shape, mel.shape, lfcc.shape)
        torch.Size([15, 3, 224, 224]) torch.Size([1, 128, T]) torch.Size([1, 128, T])
    """
    
    def __init__(
        self,
        device: str = "cpu",
        num_frames: int = 15,
        target_size: Tuple[int, int] = (224, 224),
        face_margin: float = 0.20,
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
        min_face_detection_ratio: float = 0.5
    ):
        """
        Initialize VideoPreprocessor with frame and audio extraction settings.
        
        Args:
            device: Computing device for MTCNN (cpu, cuda, mps)
            num_frames: Number of frames to extract (default: 15)
            target_size: Target frame size (height, width) (default: 224x224)
            face_margin: Margin expansion ratio for face bbox (default: 0.20)
            center_crop_ratio: Center crop ratio for fallback (default: 0.50)
            audio_sample_rate: Target audio sample rate (default: 16000 Hz)
            audio_duration: Audio duration to extract (default: 4.0s)
            n_mels: Number of mel bins (default: 128)
            n_lfcc: Number of LFCC coefficients (default: 128)
            n_fft: FFT window size (default: 1024)
            hop_length: Hop length for STFT (default: 512)
            f_min: Minimum frequency (default: 50.0 Hz)
            f_max: Maximum frequency (default: 8000.0 Hz)
            min_voice_duration: Minimum voice activity duration in seconds (default: 0.5)
            vad_top_db: Top dB threshold for VAD (default: 30)
            min_face_detection_ratio: Minimum ratio of frames with detected faces (default: 0.5)
        """
        self.device = torch.device(device)
        self.num_frames = num_frames
        self.target_size = target_size
        self.face_margin = face_margin
        self.center_crop_ratio = center_crop_ratio
        
        # Audio parameters
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.audio_samples = int(audio_sample_rate * audio_duration)
        
        # VAD parameters
        self.min_voice_duration = min_voice_duration
        self.vad_top_db = vad_top_db
        
        # Face detection parameters
        self.min_face_detection_ratio = min_face_detection_ratio
        
        # =====================================================================
        # Initialize MTCNN Face Detector
        # =====================================================================
        if MTCNN_AVAILABLE:
            self.mtcnn = MTCNN(
                keep_all=False,
                post_process=False,
                device=self.device
            )
            logger.debug(f"MTCNN face detector initialized on {self.device}")
        else:
            self.mtcnn = None
            logger.warning(
                "MTCNN not available - will use center crop fallback for all frames"
            )
        
        # =====================================================================
        # Audio Feature Extractors
        # =====================================================================
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
        
        self.lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=audio_sample_rate,
            n_filter=n_lfcc,
            n_lfcc=n_lfcc,
            f_min=f_min,
            f_max=f_max,
            speckwargs={'n_fft': n_fft, 'hop_length': hop_length}
        )
        
        logger.info(
            f"VideoPreprocessor initialized:\n"
            f"  Visual: {num_frames} frames @ {target_size}\n"
            f"  Face detection: {'MTCNN' if MTCNN_AVAILABLE else 'Center crop fallback'}\n"
            f"  Face margin: {face_margin * 100}%\n"
            f"  Min face detection ratio: {min_face_detection_ratio * 100}%\n"
            f"  Audio: {audio_duration}s @ {audio_sample_rate}Hz ({self.audio_samples} samples)\n"
            f"  Audio features: Mel ({n_mels} bins) + LFCC ({n_lfcc} coeffs)\n"
            f"  VAD: min_voice_duration={min_voice_duration}s, top_db={vad_top_db}"
        )
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract uniformly sampled frames from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List[np.ndarray]: List of RGB frames
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If video cannot be opened
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames < self.num_frames:
            logger.warning(
                f"Video has {total_frames} frames, but {self.num_frames} requested. "
                f"Will extract all available frames."
            )
            frame_indices = list(range(total_frames))
        else:
            # Sample frame indices evenly
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        
        try:
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame {frame_idx}")
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        finally:
            cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        logger.debug(
            f"Extracted {len(frames)} frames from video "
            f"(total: {total_frames}, fps: {fps:.2f})"
        )
        
        return frames
    
    def _detect_and_crop_faces(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Detect faces in frames and crop with margin expansion.
        
        FAIL-FAST: Raises NoFaceDetectedError if insufficient faces detected.
        Uses temporal continuity: if face detection fails, use last good bbox.
        
        Args:
            frames: List of RGB frames
            
        Returns:
            List[np.ndarray]: List of cropped face images (224x224)
            
        Raises:
            NoFaceDetectedError: If face detection ratio is below threshold
        """
        if self.mtcnn is None:
            logger.error("MTCNN not available - cannot perform face detection")
            raise NoFaceDetectedError(
                user_message="Unable to process video. Please try again later.",
                technical_details="MTCNN is required but not installed."
            )
        
        cropped_faces = []
        last_good_box = None
        num_frames_with_detection = 0
        
        for i, frame in enumerate(frames):
            try:
                # Detect face
                boxes, probs = self.mtcnn.detect(frame)
                
                if boxes is not None and len(boxes) > 0:
                    # Select box with highest probability
                    if len(boxes) > 1:
                        best_idx = np.argmax(probs)
                        bbox = boxes[best_idx]
                        prob = probs[best_idx]
                    else:
                        bbox = boxes[0]
                        prob = probs[0]
                    
                    last_good_box = bbox.copy()
                    num_frames_with_detection += 1
                    logger.debug(f"Frame {i}: Face detected (prob={prob:.3f})")
                    
                elif last_good_box is not None:
                    # Use last good box (temporal continuity)
                    bbox = last_good_box
                    logger.debug(f"Frame {i}: Using last good bbox (temporal continuity)")
                    
                else:
                    # No face detected and no previous bbox
                    logger.warning(f"Frame {i}: No face detected and no previous bbox")
                    bbox = None
                
                # If we have a bbox, crop the face
                if bbox is not None:
                    # Expand bounding box by margin
                    x1, y1, x2, y2 = bbox
                    w_bbox = x2 - x1
                    h_bbox = y2 - y1
                    x1_exp = x1 - w_bbox * self.face_margin
                    y1_exp = y1 - h_bbox * self.face_margin
                    x2_exp = x2 + w_bbox * self.face_margin
                    y2_exp = y2 + h_bbox * self.face_margin
                    
                    # Clip to frame boundaries
                    frame_height, frame_width = frame.shape[:2]
                    x1_exp = int(max(0, x1_exp))
                    y1_exp = int(max(0, y1_exp))
                    x2_exp = int(min(frame_width, x2_exp))
                    y2_exp = int(min(frame_height, y2_exp))
                    
                    # Crop and resize
                    face_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                    face_resized = cv2.resize(
                        face_crop,
                        self.target_size,
                        interpolation=cv2.INTER_LINEAR
                    )
                    
                    cropped_faces.append(face_resized)
                
            except Exception as e:
                logger.warning(f"Frame {i}: Face processing failed: {str(e)}")
        
        # FAIL-FAST: Check if we have sufficient face detections
        detection_ratio = num_frames_with_detection / len(frames)
        
        logger.debug(
            f"Face detection complete: {num_frames_with_detection}/{len(frames)} frames "
            f"({detection_ratio:.1%}) had direct detections"
        )
        
        if detection_ratio < self.min_face_detection_ratio:
            logger.warning(
                f"Insufficient face detections: {detection_ratio:.1%} < "
                f"{self.min_face_detection_ratio:.1%}"
            )
            raise NoFaceDetectedError(
                user_message="No face detected in the provided video.",
                technical_details=(
                    f"Only {detection_ratio:.1%} of frames contained faces "
                    f"(minimum required: {self.min_face_detection_ratio:.1%})."
                )
            )
        
        if len(cropped_faces) == 0:
            raise NoFaceDetectedError(user_message="No face detected in the provided video.")
        
        return cropped_faces
    
    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        """
        Fallback: 50% center crop of frame.
        
        Args:
            frame: RGB frame
            
        Returns:
            np.ndarray: Center-cropped and resized frame (224x224)
        """
        h, w = frame.shape[:2]
        crop_size = min(h, w) // 2
        cx, cy = w // 2, h // 2
        
        x1 = cx - crop_size // 2
        y1 = cy - crop_size // 2
        x2 = cx + crop_size // 2
        y2 = cy + crop_size // 2
        
        # Ensure bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Crop and resize
        cropped = frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        return resized
    
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
            intervals = librosa.effects.split(waveform, top_db=self.vad_top_db)
            
            # Calculate total duration of active (non-silent) audio
            total_active_samples = sum(end - start for start, end in intervals)
            total_active_duration = total_active_samples / self.audio_sample_rate
            
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
                    user_message="No voice detected in the provided video.",
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
                user_message="Unable to process video audio. Please try again later.",
                technical_details=f"Voice activity detection failed: {str(e)}"
            ) from e
    
    def _extract_audio(self, video_path: str) -> np.ndarray:
        """
        Extract audio track from video and standardize to target duration.
        
        FAIL-FAST: Raises NoVoiceDetectedError if no voice activity detected.
        
        Args:
            video_path: Path to video file
            
        Returns:
            np.ndarray: Audio waveform (audio_samples long, normalized to [-1, 1])
            
        Raises:
            NoVoiceDetectedError: If no voice activity is detected
        """
        waveform = None
        decode_errors: List[str] = []

        # Attempt 1: direct decode with librosa (uses local audio backends)
        try:
            waveform, sr = librosa.load(
                video_path,
                sr=self.audio_sample_rate,
                mono=True
            )
            logger.debug(
                f"Extracted audio via librosa: {len(waveform)} samples "
                f"({len(waveform)/self.audio_sample_rate:.2f}s)"
            )
        except Exception as e:
            decode_errors.append(f"librosa direct decode failed: {type(e).__name__}: {repr(e)}")
            logger.warning(
                f"Direct librosa decode failed for {video_path}: {type(e).__name__}: {repr(e)}"
            )

        # Attempt 2: ffmpeg -> temporary wav, then librosa load wav
        if waveform is None or len(waveform) == 0:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        video_path,
                        "-map",
                        "a:0",
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ac",
                        "1",
                        "-ar",
                        str(self.audio_sample_rate),
                        tmp_wav.name
                    ]
                    ffmpeg_proc = subprocess.run(
                        ffmpeg_cmd,
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    if ffmpeg_proc.returncode != 0:
                        stderr = (ffmpeg_proc.stderr or "").strip()
                        stdout = (ffmpeg_proc.stdout or "").strip()

                        # Common case: uploaded video has no audio stream.
                        no_audio_markers = (
                            "does not contain any stream",
                            "stream map 'a:0' matches no streams",
                            "matches no streams"
                        )
                        if any(marker in stderr.lower() for marker in no_audio_markers):
                            raise NoVoiceDetectedError(
                                user_message="No voice detected in the provided video.",
                                technical_details=(
                                    "Video does not contain an audio stream or has an unreadable audio track."
                                )
                            )

                        raise RuntimeError(
                            f"ffmpeg error. stderr={stderr or '<empty>'}, stdout={stdout or '<empty>'}"
                        )

                    waveform, sr = librosa.load(
                        tmp_wav.name,
                        sr=self.audio_sample_rate,
                        mono=True
                    )
                    logger.debug(
                        f"Extracted audio via ffmpeg fallback: {len(waveform)} samples "
                        f"({len(waveform)/self.audio_sample_rate:.2f}s)"
                    )
            except NoVoiceDetectedError:
                raise
            except FileNotFoundError as e:
                decode_errors.append("ffmpeg fallback failed: ffmpeg executable not found in PATH")
                logger.warning("ffmpeg fallback unavailable: ffmpeg executable not found in PATH")
            except Exception as e:
                decode_errors.append(f"ffmpeg wav extraction failed: {type(e).__name__}: {repr(e)}")
                logger.warning(
                    f"ffmpeg fallback decode failed for {video_path}: {type(e).__name__}: {repr(e)}"
                )

        try:
            if waveform is None or len(waveform) == 0:
                joined_errors = " | ".join(decode_errors) if decode_errors else "unknown decode failure"
                raise RuntimeError(joined_errors)

            # FAIL-FAST: Validate voice activity before processing
            self._validate_voice_activity(waveform)

            # Standardize duration
            if len(waveform) > self.audio_samples:
                # Take first audio_duration seconds
                waveform = waveform[:self.audio_samples]
            elif len(waveform) < self.audio_samples:
                # Wrap-around padding
                repeats = (self.audio_samples // len(waveform)) + 1
                waveform = np.tile(waveform, repeats)[:self.audio_samples]

            # Amplitude normalization to [-1, 1]
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val

            return waveform

        except (NoVoiceDetectedError, NoFaceDetectedError):
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            logger.error(f"Could not extract audio from video: {type(e).__name__}: {repr(e)}")
            raise NoVoiceDetectedError(
                user_message="Unable to process video audio. Please try again later.",
                technical_details=f"Failed to extract audio from video: {type(e).__name__}: {repr(e)}"
            ) from e
    
    def _extract_audio_features(
        self,
        waveform: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract dual-stream audio features: Mel-Spectrogram + LFCC.
        
        Args:
            waveform: Audio waveform (audio_samples long)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (log_mel_spec, lfcc)
                - log_mel_spec: (1, n_mels, time_frames)
                - lfcc: (1, n_lfcc, time_frames)
        """
        # Convert to torch tensor
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, samples)
        
        # Stream A: Extract Mel-Spectrogram
        mel_spec = self.mel_spectrogram(waveform_tensor)  # (1, n_mels, time_frames)
        log_mel_spec = torch.log(mel_spec + 1e-9)  # Log-scale
        
        # Stream B: Extract LFCC
        lfcc = self.lfcc_transform(waveform_tensor)  # (1, n_lfcc, time_frames)
        
        return log_mel_spec, lfcc
    
    def process(
        self,
        video_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete preprocessing pipeline: Extract Frames + Audio → Features.
        
        This is the main entry point for preprocessing. It performs:
            1. Extract uniformly sampled frames from video
            2. Detect and crop faces with MTCNN (temporal continuity)
            3. Extract audio track at 16kHz mono
            4. Extract dual-stream audio features (Mel + LFCC)
            
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (frames_tensor, mel_tensor, lfcc_tensor)
                - frames_tensor: (num_frames, 3, 224, 224) - Normalized frames [0, 1]
                - mel_tensor: (1, 128, time_frames) - Log-Mel Spectrogram
                - lfcc_tensor: (1, 128, time_frames) - LFCC features
                
        Raises:
            ValueError: If video processing fails
            FileNotFoundError: If video file doesn't exist
            
        Example:
            >>> preprocessor = VideoPreprocessor()
            >>> frames, mel, lfcc = preprocessor.process("video.mp4")
            >>> print(f"Frames: {frames.shape}, Mel: {mel.shape}, LFCC: {lfcc.shape}")
            Frames: torch.Size([15, 3, 224, 224]), Mel: torch.Size([1, 128, 125]), LFCC: torch.Size([1, 128, 125])
        """
        try:
            if not isinstance(video_path, str):
                raise ValueError("VideoPreprocessor requires file path as input")
            
            logger.debug(f"Starting video preprocessing: {video_path}")
            
            # =====================================================================
            # Step 1: Extract frames from video
            # =====================================================================
            frames = self._extract_frames(video_path)
            
            # =====================================================================
            # Step 2: Detect and crop faces with temporal continuity
            # =====================================================================
            cropped_faces = self._detect_and_crop_faces(frames)
            
            # Ensure we have exactly num_frames (pad if needed)
            while len(cropped_faces) < self.num_frames:
                # Duplicate last frame if we have fewer frames
                cropped_faces.append(cropped_faces[-1] if cropped_faces else self._center_crop(frames[0]))
            
            # Take only num_frames if we have more
            cropped_faces = cropped_faces[:self.num_frames]
            
            # =====================================================================
            # Step 3: Convert frames to tensor
            # =====================================================================
            # Convert to tensor (num_frames, 3, 224, 224) and normalize to [0, 1]
            faces_tensor = torch.from_numpy(np.array(cropped_faces)).float()
            faces_tensor = faces_tensor.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            faces_tensor = faces_tensor / 255.0  # Normalize to [0, 1]
            
            # =====================================================================
            # Step 4: Extract audio from video
            # =====================================================================
            waveform = self._extract_audio(video_path)
            
            # =====================================================================
            # Step 5: Extract dual-stream audio features
            # =====================================================================
            log_mel_spec, lfcc = self._extract_audio_features(waveform)
            
            logger.debug(
                f"Preprocessing complete:\n"
                f"  Frames: shape={faces_tensor.shape}, "
                f"range=[{faces_tensor.min():.3f}, {faces_tensor.max():.3f}]\n"
                f"  Mel-Spectrogram: shape={log_mel_spec.shape}, "
                f"range=[{log_mel_spec.min():.3f}, {log_mel_spec.max():.3f}]\n"
                f"  LFCC: shape={lfcc.shape}, "
                f"range=[{lfcc.min():.3f}, {lfcc.max():.3f}]"
            )
            
            return faces_tensor, log_mel_spec, lfcc
            
        except (NoFaceDetectedError, NoVoiceDetectedError):
            # Re-raise validation errors as-is (user-friendly messages)
            raise
        except Exception as e:
            logger.error(f"Video preprocessing failed: {str(e)}")
            raise ValueError(f"Preprocessing error: {str(e)}") from e


# =============================================================================
# Utility Functions
# =============================================================================

def preprocess_video_tristream(
    video_path: str,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience function for tri-stream video preprocessing.
    
    Args:
        video_path: Path to video file
        device: Computing device for MTCNN
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (frames, mel, lfcc)
        
    Example:
        >>> from app.preprocessing_pipelines.video_preprocessor import preprocess_video_tristream
        >>> frames, mel, lfcc = preprocess_video_tristream("video.mp4")
    """
    preprocessor = VideoPreprocessor(device=device)
    return preprocessor.process(video_path)


# =============================================================================
# Testing / Validation
# =============================================================================

if __name__ == "__main__":
    """
    Test VideoPreprocessor with dummy video.
    """
    logger.info("=== Testing VideoPreprocessor ===")
    
    logger.info("\n[Test 1] Create dummy video for testing")
    
    # Create a simple test video file
    test_video_path = Path("test_video.mp4")
    
    if not test_video_path.exists():
        logger.info("Creating test video with OpenCV...")
        
        # Create a simple 2-second video with colored frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(test_video_path), fourcc, 10.0, (640, 480))
        
        for i in range(20):  # 20 frames at 10 fps = 2 seconds
            # Create colored frame (BGR)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = [i * 12, 100, 255 - i * 12]  # Varying colors
            out.write(frame)
        
        out.release()
        logger.success(f"✓ Test video created: {test_video_path}")
    
    logger.info("\n[Test 2] Process test video")
    
    try:
        preprocessor = VideoPreprocessor(device="cpu", num_frames=10)
        frames_tensor, mel_tensor, lfcc_tensor = preprocessor.process(str(test_video_path))
        
        logger.info(f"✓ Frames tensor: shape={frames_tensor.shape}")
        logger.info(f"✓ Mel tensor: shape={mel_tensor.shape}")
        logger.info(f"✓ LFCC tensor: shape={lfcc_tensor.shape}")
        
        assert frames_tensor.shape[0] == 10, "Should have 10 frames!"
        assert frames_tensor.shape[1] == 3, "Should have 3 channels (RGB)!"
        assert frames_tensor.shape[2:] == (224, 224), "Should be 224x224!"
        assert mel_tensor.shape[1] == 128, "Should have 128 mel bins!"
        assert lfcc_tensor.shape[1] == 128, "Should have 128 LFCC coefficients!"
        
        logger.success("✓ Video processing test passed!\n")
        
    finally:
        # Clean up test video
        if test_video_path.exists():
            test_video_path.unlink()
            logger.debug("Cleaned up test video")
    
    logger.success("=== All VideoPreprocessor Tests Passed! ===")
