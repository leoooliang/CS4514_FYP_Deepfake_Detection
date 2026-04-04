"""
Video deepfake detection endpoint.

Upload a video to check for deepfake manipulation. 
Supported file extensions: MP4, AVI, MOV, MKV, WEBM.  Max file size: 100 MB.

Endpoint: /api/v1/predict/video
Response model: VideoPredictionResponse
Tags: video
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.core.validation import validate_video_file
from app.db.database import get_db
from app.schemas.response import VideoPredictionResponse
from app.services.detection import run_detection_pipeline

router = APIRouter()


@router.post(
    "/video",
    response_model=VideoPredictionResponse,
    summary="Detect deepfakes in videos",
    description=(
        "Upload a video to check for deepfake manipulation.  "
        "Supported: MP4, AVI, MOV, MKV, WEBM.  Max 100 MB."
    ),
    tags=["video"],
)
async def predict_video_deepfake(
    request: Request,
    file: UploadFile = File(..., description="Video file to analyse"),
    session_id: Optional[str] = Form(None, description="Session ID for user tracking"),
    db: Session = Depends(get_db),
) -> VideoPredictionResponse:
    data = await run_detection_pipeline(
        request=request,
        file=file,
        session_id=session_id,
        db=db,
        detector_type="video",
        allowed_extensions=settings.ALLOWED_VIDEO_EXTENSIONS,
        max_bytes=settings.max_video_size_bytes,
        validate_fn=validate_video_file,
        transcode_video=True,
    )
    return VideoPredictionResponse(**data)
