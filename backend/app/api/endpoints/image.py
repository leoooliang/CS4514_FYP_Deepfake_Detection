"""
Image deepfake detection endpoint.

Upload an image to check if it is real or a deepfake. 
Supported file extensions: JPG, JPEG, PNG, BMP, WEBP.  Max file size: 100 MB.

Endpoint: /api/v1/predict/image
Response model: ImagePredictionResponse
Tags: image
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from PIL import Image
from sqlalchemy.orm import Session

from app.config import settings
from app.core.validation import validate_image_file
from app.db.database import get_db
from app.schemas.response import ImagePredictionResponse
from app.services.detection import run_detection_pipeline

router = APIRouter()


@router.post(
    "/image",
    response_model=ImagePredictionResponse,
    summary="Detect deepfakes in images",
    description=(
        "Upload an image to check if it is real or a deepfake.  "
        "Supported: JPG, JPEG, PNG, BMP, WEBP.  Max 100 MB."
    ),
    tags=["image"],
)
async def predict_image_deepfake(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyse"),
    session_id: Optional[str] = Form(None, description="Session ID for user tracking"),
    db: Session = Depends(get_db),
) -> ImagePredictionResponse:
    data = await run_detection_pipeline(
        request=request,
        file=file,
        session_id=session_id,
        db=db,
        detector_type="image",
        allowed_extensions=settings.ALLOWED_IMAGE_EXTENSIONS,
        max_bytes=settings.max_image_size_bytes,
        validate_fn=validate_image_file,
        open_fn=lambda p: Image.open(p),
    )
    return ImagePredictionResponse(**data)
