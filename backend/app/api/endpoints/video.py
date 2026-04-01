"""
============================================================================
Video Detection Endpoint
============================================================================
API endpoint for video deepfake detection.

Endpoint: POST /api/v1/predict/video
Input: Multipart form data with video file
Output: JSON with prediction results

Author: Senior Full-Stack Engineer
Date: 2026-01-28
============================================================================
"""

import os
import tempfile
import time
import uuid
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, status, Depends, Form
from loguru import logger
from sqlalchemy.orm import Session

from app.config import settings
from app.schemas.response import VideoPredictionResponse
from app.core.validation import (
    validate_video_file as validate_video_file_full,
    sanitize_filename,
    validate_upload_safety
)
from app.core.exceptions import NoFaceDetectedError, NoVoiceDetectedError
from app.db.database import get_db
from app.db import crud


# Create router for video endpoints
router = APIRouter()


# =============================================================================
# Helper Functions
# =============================================================================

def validate_video_extension(file: UploadFile) -> None:
    """
    Validate uploaded video file extension.
    
    Args:
        file: Uploaded file object
    
    Raises:
        HTTPException: If validation fails
    """
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: {', '.join(settings.ALLOWED_VIDEO_EXTENSIONS)}"
        )
    
    logger.debug(f"Video file validation passed: {file.filename}")


async def save_video_file_tmp(upload_file: UploadFile) -> str:
    """
    Save uploaded video file to temporary location.
    
    Args:
        upload_file: FastAPI UploadFile object
    
    Returns:
        str: Path to temporary file
    """
    suffix = Path(upload_file.filename).suffix
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        # Read in chunks for large files
        chunk_size = 1024 * 1024  # 1 MB chunks
        total_size = 0
        
        while chunk := await upload_file.read(chunk_size):
            total_size += len(chunk)
            
            # Check size limit
            if total_size > settings.max_video_size_bytes:
                tmp_file.close()
                os.unlink(tmp_file.name)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Maximum size: {settings.MAX_VIDEO_SIZE_MB}MB"
                )
            
            tmp_file.write(chunk)
        
        tmp_path = tmp_file.name
    
    logger.debug(f"Saved video to temporary file: {tmp_path} ({total_size / (1024*1024):.2f} MB)")
    return tmp_path


def transcode_to_websafe_mp4(input_path: str, output_path: str) -> None:
    """
    Transcode a video to browser-safe MP4 (H.264 + AAC).

    Args:
        input_path: Source video path
        output_path: Target MP4 path

    Raises:
        RuntimeError: If ffmpeg is unavailable or transcoding fails
    """
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg is not installed or not available in PATH. "
            "Install FFmpeg to enable web-safe MP4 transcoding."
        ) from exc

    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip()[-800:]
        raise RuntimeError(f"FFmpeg transcoding failed: {stderr_tail}")


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/predict/video",
    response_model=VideoPredictionResponse,
    summary="Detect deepfakes in videos",
    description="""
    Upload a video to check if it contains deepfake manipulation.
    
    **Supported formats:** MP4, AVI, MOV, MKV, WEBM  
    **Maximum size:** 100 MB  
    **Processing time:** ~10-60 seconds (depends on video length)
    
    The endpoint analyzes video frames using:
    - Frame-by-frame deepfake detection
    - Temporal consistency analysis
    - Face tracking across frames
    - Artifact detection in motion
    
    Returns comprehensive analysis including per-frame insights.
    """,
    responses={
        200: {
            "description": "Successful detection",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": "deepfake",
                        "confidence": 0.87,
                        "probabilities": {
                            "real": 0.13,
                            "deepfake": 0.87
                        },
                        "processing_time_seconds": 12.45,
                        "inference_time_ms": 12000.0,
                        "metadata": {
                            "model_type": "VideoDeepfakeNet",
                            "num_frames_analyzed": 30
                        },
                        "record_id": "123e4567-e89b-12d3-a456-426614174000"
                    }
                }
            }
        },
        400: {"description": "Invalid file format or corrupted video"},
        413: {"description": "File too large"},
        500: {"description": "Internal server error during processing"}
    },
    tags=["video"]
)
async def predict_video_deepfake(
    request: Request,
    file: UploadFile = File(..., description="Video file to analyze"),
    session_id: Optional[str] = Form(None, description="Session ID for anonymous user tracking"),
    db: Session = Depends(get_db)
) -> VideoPredictionResponse:
    """
    Detect if an uploaded video contains deepfakes.
    
    Args:
        request: FastAPI request object
        file: Uploaded video file
    
    Returns:
        VideoPredictionResponse: Detection results with confidence scores
    
    Raises:
        HTTPException: If validation or processing fails
    """
    logger.info(f"Received video detection request: {file.filename}")
    
    # Security validation
    try:
        validate_upload_safety(file.filename)
        safe_filename = sanitize_filename(file.filename)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Validate file extension
    validate_video_extension(file)
    
    tmp_path = None
    record_id = None
    
    try:
        # Save uploaded file temporarily
        tmp_path = await save_video_file_tmp(file)
        file_size = os.path.getsize(tmp_path)
        
        # Comprehensive validation
        try:
            validate_video_file_full(tmp_path, file_size, safe_filename)
        except ValueError as e:
            # Clean up temp file before raising
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Get detector from factory
        factory = request.app.state.detector_factory
        video_detector = factory.get_detector("video")
        
        # Measure processing duration
        start_time = time.time()
        
        # Run detection (may raise NoFaceDetectedError or NoVoiceDetectedError during preprocessing)
        logger.debug(f"Running detection on video: {tmp_path}")
        try:
            result = video_detector.detect(tmp_path)
        except (NoFaceDetectedError, NoVoiceDetectedError) as e:
            # Clean up temp file before raising HTTP exception
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Calculate processing duration
        processing_duration = time.time() - start_time
        
        # Extract video-specific metadata with proper defaults
        metadata = result.metadata or {}
        
        # Determine classification and model version
        classification = "Fake" if result.prediction == "deepfake" else "Real"
        model_version = metadata.get("model_type", "VideoDeepfakeNet-v1.0")
        
        # Save file persistently with unique filename
        media_path = None
        try:
            # Generate unique filename to prevent collisions
            unique_id = str(uuid.uuid4())[:8]
            stem_name = Path(safe_filename).stem
            unique_filename = f"{unique_id}_{stem_name}.mp4"
            
            # Define persistent media directory
            persistent_media_dir = os.path.join(os.getcwd(), "persistent_media")
            os.makedirs(persistent_media_dir, exist_ok=True)
            
            # Transcode to browser-safe MP4 before storing for frontend preview.
            persistent_file_path = os.path.join(persistent_media_dir, unique_filename)
            transcode_to_websafe_mp4(tmp_path, persistent_file_path)
            
            # Store URL path for frontend access
            media_path = f"/media/{unique_filename}"
            
            logger.info(f"✓ Media file transcoded and saved: {media_path}")
            
        except Exception as e:
            logger.error(f"⚠️ Failed to transcode/save media file: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to transcode uploaded video to web-safe MP4. Ensure FFmpeg is installed and operational.",
            )
        
        # Log detection result to database
        try:
            record_data = {
                "file_name": safe_filename,
                "file_type": "video",
                "file_size": file_size,
                "detection_score": result.confidence,
                "classification": classification,
                "model_version": model_version,
                "processing_duration": processing_duration,
                "session_id": session_id,
                "media_path": media_path
            }
            
            db_record = crud.create_detection_record(db, record_data)
            record_id = db_record.id
            
            logger.info(
                f"✓ Detection logged to database with ID: {record_id}"
            )
            
        except Exception as e:
            logger.error(
                f"⚠️ Failed to log detection to database: {str(e)}"
            )
        
        # Convert to response model with standardized field names
        is_fake = result.prediction == "deepfake"
        inference_time_ms = result.processing_time * 1000.0
        
        response = VideoPredictionResponse(
            prediction=result.prediction,
            is_fake=is_fake,
            confidence=result.confidence,
            probabilities=result.probabilities,
            processing_time_seconds=processing_duration,
            inference_time_ms=inference_time_ms,
            metadata={
                **metadata,
                "model_type": metadata.get("model_type", "VideoDeepfakeNet"),
                "device": metadata.get("device", str(request.app.state.detector_factory._device))
            },
            record_id=record_id
        )
        
        logger.success(
            f"Video detection completed: {result.prediction} "
            f"({result.confidence:.2%} confidence)"
        )
        
        return response
        
    except HTTPException:
        raise
    
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Video validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
        
    except Exception as e:
        logger.error(f"Video detection failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video processing failed: {str(e)}"
        )
        
    finally:
        # Clean up temporary file (after copying to persistent storage)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {str(e)}")
