"""
============================================================================
Audio Detection Endpoint
============================================================================
API endpoint for audio deepfake detection.

Endpoint: POST /api/v1/predict/audio
Input: Multipart form data with audio file
Output: JSON with prediction results

Author: Senior Full-Stack Engineer
Date: 2026-01-28
============================================================================
"""

import os
import tempfile
import time
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, status, Depends, Form
from loguru import logger
from sqlalchemy.orm import Session

from app.config import settings
from app.schemas.response import AudioPredictionResponse
from app.core.validation import (
    validate_audio_file as validate_audio_file_full,
    sanitize_filename,
    validate_upload_safety
)
from app.core.exceptions import NoVoiceDetectedError
from app.db.database import get_db
from app.db import crud


# Create router for audio endpoints
router = APIRouter()


# =============================================================================
# Helper Functions
# =============================================================================

def validate_audio_extension(file: UploadFile) -> None:
    """
    Validate uploaded audio file extension.
    
    Args:
        file: Uploaded file object
    
    Raises:
        HTTPException: If validation fails
    """
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: {', '.join(settings.ALLOWED_AUDIO_EXTENSIONS)}"
        )
    
    logger.debug(f"Audio file validation passed: {file.filename}")


async def save_audio_file_tmp(upload_file: UploadFile) -> str:
    """
    Save uploaded audio file to temporary location.
    
    Args:
        upload_file: FastAPI UploadFile object
    
    Returns:
        str: Path to temporary file
    """
    suffix = Path(upload_file.filename).suffix
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await upload_file.read()
        
        # Validate file size
        if len(content) > settings.max_audio_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.MAX_AUDIO_SIZE_MB}MB"
            )
        
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    logger.debug(f"Saved audio to temporary file: {tmp_path}")
    return tmp_path


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/predict/audio",
    response_model=AudioPredictionResponse,
    summary="Detect deepfakes in audio",
    description="""
    Upload an audio file to check if it contains synthetic or cloned voice.
    
    **Supported formats:** MP3, WAV, FLAC, OGG, M4A  
    **Maximum size:** 100 MB  
    **Processing time:** ~2-5 seconds
    
    The endpoint analyzes audio using:
    - Mel-spectrogram analysis
    - Voice artifact detection
    - Speech synthesis patterns
    - Temporal consistency checks
    
    Detects:
    - Voice cloning (e.g., Lyrebird, Descript)
    - Speech synthesis (e.g., WaveNet, Tacotron)
    - Voice conversion attacks
    - Audio splicing manipulation
    """,
    responses={
        200: {
            "description": "Successful detection",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": "deepfake",
                        "confidence": 0.92,
                        "probabilities": {
                            "real": 0.08,
                            "deepfake": 0.92
                        },
                        "processing_time_seconds": 3.21,
                        "inference_time_ms": 3000.0,
                        "metadata": {
                            "model_type": "AudioDeepfakeNet",
                            "num_segments_analyzed": 4
                        },
                        "record_id": "123e4567-e89b-12d3-a456-426614174000"
                    }
                }
            }
        },
        400: {"description": "Invalid file format or corrupted audio"},
        413: {"description": "File too large"},
        500: {"description": "Internal server error during processing"}
    },
    tags=["audio"]
)
async def predict_audio_deepfake(
    request: Request,
    file: UploadFile = File(..., description="Audio file to analyze"),
    session_id: Optional[str] = Form(None, description="Session ID for anonymous user tracking"),
    db: Session = Depends(get_db)
) -> AudioPredictionResponse:
    """
    Detect if an uploaded audio file is a deepfake.
    
    Args:
        request: FastAPI request object
        file: Uploaded audio file
    
    Returns:
        AudioPredictionResponse: Detection results with segment analysis
    
    Raises:
        HTTPException: If validation or processing fails
    """
    logger.info(f"Received audio detection request: {file.filename}")
    
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
    validate_audio_extension(file)
    
    tmp_path = None
    record_id = None
    
    try:
        # Save uploaded file temporarily
        tmp_path = await save_audio_file_tmp(file)
        file_size = os.path.getsize(tmp_path)
        
        # Comprehensive validation
        try:
            validate_audio_file_full(tmp_path, file_size, safe_filename)
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
        audio_detector = factory.get_detector("audio")
        
        # Measure processing duration
        start_time = time.time()
        
        # Run detection (may raise NoVoiceDetectedError during preprocessing)
        logger.debug(f"Running detection on audio: {tmp_path}")
        try:
            result = audio_detector.detect(tmp_path)
        except NoVoiceDetectedError as e:
            # Clean up temp file before raising HTTP exception
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Calculate processing duration
        processing_duration = time.time() - start_time
        
        # Extract audio-specific metadata with proper defaults
        metadata = result.metadata or {}
        
        # Determine classification and model version
        classification = "Fake" if result.prediction == "deepfake" else "Real"
        model_version = metadata.get("model_type", "AudioDeepfakeNet-v1.0")
        
        # Save file persistently with unique filename
        media_path = None
        try:
            # Generate unique filename to prevent collisions
            unique_id = str(uuid.uuid4())[:8]
            file_extension = Path(safe_filename).suffix
            unique_filename = f"{unique_id}_{safe_filename}"
            
            # Define persistent media directory
            persistent_media_dir = os.path.join(os.getcwd(), "persistent_media")
            os.makedirs(persistent_media_dir, exist_ok=True)
            
            # Move file from temp to persistent storage
            persistent_file_path = os.path.join(persistent_media_dir, unique_filename)
            shutil.copy2(tmp_path, persistent_file_path)
            
            # Store URL path for frontend access
            media_path = f"/media/{unique_filename}"
            
            logger.info(f"✓ Media file saved persistently: {media_path}")
            
        except Exception as e:
            logger.error(f"⚠️ Failed to save media file persistently: {str(e)}")
        
        # Log detection result to database
        try:
            record_data = {
                "file_name": safe_filename,
                "file_type": "audio",
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
        
        response = AudioPredictionResponse(
            prediction=result.prediction,
            is_fake=is_fake,
            confidence=result.confidence,
            probabilities=result.probabilities,
            processing_time_seconds=processing_duration,
            inference_time_ms=inference_time_ms,
            metadata={
                **metadata,
                "model_type": metadata.get("model_type", "AudioDeepfakeNet"),
                "device": metadata.get("device", str(request.app.state.detector_factory._device))
            },
            record_id=record_id
        )
        
        logger.success(
            f"Audio detection completed: {result.prediction} "
            f"({result.confidence:.2%} confidence)"
        )
        
        return response
        
    except HTTPException:
        raise
    
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Audio validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
        
    except Exception as e:
        logger.error(f"Audio detection failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {str(e)}"
        )
        
    finally:
        # Clean up temporary file (after copying to persistent storage)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {str(e)}")
