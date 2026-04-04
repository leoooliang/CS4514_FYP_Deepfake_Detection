"""
Pydantic request models (currently used for future batch / options endpoints).
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    return_metadata: bool = True
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class ImagePredictionOptions(BaseModel):
    detect_faces: bool = True
    return_heatmap: bool = False


class VideoPredictionOptions(BaseModel):
    fps_sample_rate: Optional[int] = Field(None, ge=1, le=30)
    max_frames: Optional[int] = Field(None, ge=1, le=500)
    analyze_audio: bool = False


class AudioPredictionOptions(BaseModel):
    segment_duration: Optional[int] = Field(None, ge=1, le=10)
    sample_rate: Optional[int] = None
    return_spectrogram: bool = False


class BatchPredictionRequest(BaseModel):
    file_urls: list[str]
    detection_type: str
    options: Optional[dict] = None

    @field_validator("detection_type")
    @classmethod
    def _validate_type(cls, v):
        if v not in ("image", "video", "audio"):
            raise ValueError("detection_type must be image, video, or audio")
        return v
