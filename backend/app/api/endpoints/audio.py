"""
Audio deepfake detection endpoint.

Upload an audio file to check for synthetic or cloned voice. 
Supported file extensions: MP3, WAV, FLAC, OGG, M4A.  Max file size: 100 MB.

Endpoint: /api/v1/predict/audio
Response model: AudioPredictionResponse
Tags: audio
""" 
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.core.validation import validate_audio_file
from app.db.database import get_db
from app.schemas.response import AudioPredictionResponse
from app.services.detection import run_detection_pipeline

router = APIRouter()


@router.post(
    "/audio",
    response_model=AudioPredictionResponse,
    summary="Detect deepfakes in audio",
    description=(
        "Upload an audio file to check for synthetic or cloned voice.  "
        "Supported: MP3, WAV, FLAC, OGG, M4A.  Max 100 MB."
    ),
    tags=["audio"],
)
async def predict_audio_deepfake(
    request: Request,
    file: UploadFile = File(..., description="Audio file to analyse"),
    session_id: Optional[str] = Form(None, description="Session ID for user tracking"),
    db: Session = Depends(get_db),
) -> AudioPredictionResponse:
    data = await run_detection_pipeline(
        request=request,
        file=file,
        session_id=session_id,
        db=db,
        detector_type="audio",
        allowed_extensions=settings.ALLOWED_AUDIO_EXTENSIONS,
        max_bytes=settings.max_audio_size_bytes,
        validate_fn=validate_audio_file,
    )
    return AudioPredictionResponse(**data)
