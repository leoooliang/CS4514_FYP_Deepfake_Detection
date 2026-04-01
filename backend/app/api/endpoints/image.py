"""
============================================================================
Image Detection Endpoint
============================================================================
API endpoint for image deepfake detection.

Endpoint: POST /api/v1/predict/image
Input: Multipart form data with image file
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
from PIL import Image
from sqlalchemy.orm import Session

from app.config import settings
from app.schemas.response import ImagePredictionResponse
from app.core.validation import validate_image_file, sanitize_filename, validate_upload_safety
from app.core.exceptions import NoFaceDetectedError
from app.db.database import get_db
from app.db import crud


# Create router for image endpoints
router = APIRouter()


# =============================================================================
# Helper Functions
# =============================================================================

def validate_image_upload(file: UploadFile) -> None:
    """
    Validate uploaded image file (basic checks before saving).
    
    Checks:
        - File extension
    
    Args:
        file: Uploaded file object
    
    Raises:
        HTTPException: If validation fails
    """
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}"
        )
    
    logger.debug(f"Image file upload validation passed: {file.filename}")


async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """
    Save uploaded file to temporary location.
    
    Args:
        upload_file: FastAPI UploadFile object
    
    Returns:
        str: Path to temporary file
    """
    # Create temporary file with same extension
    suffix = Path(upload_file.filename).suffix
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        # Read and write in chunks to handle large files
        content = await upload_file.read()
        
        # Validate file size
        if len(content) > settings.max_image_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.MAX_IMAGE_SIZE_MB}MB"
            )
        
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    logger.debug(f"Saved upload to temporary file: {tmp_path}")
    return tmp_path


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/predict/image",
    response_model=ImagePredictionResponse,
    summary="Detect deepfakes in images",
    description="""
    Upload an image to check if it's real or a deepfake.
    
    **Supported formats:** JPG, JPEG, PNG, BMP, WEBP  
    **Maximum size:** 100 MB  
    **Processing time:** ~1-3 seconds
    
    The endpoint analyzes the image using a deep learning model trained to detect:
    - Face swaps
    - GAN-generated images
    - Facial manipulation
    - Image splicing
    
    Returns a prediction with confidence score and detailed analysis.
    """,
    responses={
        200: {
            "description": "Successful detection",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": "deepfake",
                        "is_fake": True,
                        "confidence": 0.95,
                        "probabilities": {
                            "real": 0.05,
                            "deepfake": 0.95
                        },
                        "processing_time_seconds": 1.23,
                        "inference_time_ms": 1200.0,
                        "metadata": {
                            "model_type": "ImageDeepfakeNet",
                            "device": "cuda"
                        },
                        "record_id": "123e4567-e89b-12d3-a456-426614174000"
                    }
                }
            }
        },
        400: {"description": "Invalid file format or corrupted file"},
        413: {"description": "File too large"},
        500: {"description": "Internal server error during processing"}
    },
    tags=["image"]
)
async def predict_image_deepfake(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze"),
    session_id: Optional[str] = Form(None, description="Session ID for anonymous user tracking"),
    db: Session = Depends(get_db)
) -> ImagePredictionResponse:
    """
    Detect if an uploaded image is a deepfake.
    
    Args:
        request: FastAPI request object (for accessing app state)
        file: Uploaded image file
    
    Returns:
        ImagePredictionResponse: Detection results with confidence scores
    
    Raises:
        HTTPException: If validation or processing fails
    """
    logger.info(f"Received image detection request: {file.filename}")
    
    # Security validation
    try:
        validate_upload_safety(file.filename)
        safe_filename = sanitize_filename(file.filename)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Validate file extension (basic check)
    validate_image_upload(file)
    
    tmp_path = None
    record_id = None
    
    try:
        # Save uploaded file temporarily
        tmp_path = await save_upload_file_tmp(file)
        file_size = os.path.getsize(tmp_path)
        
        # Comprehensive validation
        try:
            validate_image_file(tmp_path, file_size, safe_filename)
        except ValueError as e:
            # Clean up temp file before raising
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Open image for detection
        try:
            img = Image.open(tmp_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid or corrupted image file: {str(e)}"
            )
        
        # Get detector from factory
        factory = request.app.state.detector_factory
        image_detector = factory.get_detector("image")
        
        # Measure processing duration
        start_time = time.time()
        
        # Run detection (may raise NoFaceDetectedError during preprocessing)
        logger.debug(f"Running detection on image: {tmp_path}")
        try:
            result = image_detector.detect(img)
        except NoFaceDetectedError as e:
            # Clean up temp file before raising HTTP exception
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Calculate processing duration
        processing_duration = time.time() - start_time
        
        # Extract metadata for specific fields
        metadata = result.metadata or {}
        
        # Determine classification and model version
        classification = "Fake" if result.prediction == "deepfake" else "Real"
        model_version = metadata.get("model_type", "ImageDeepfakeNet-v1.0")
        
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
                "file_type": "image",
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
        
        response = ImagePredictionResponse(
            prediction=result.prediction,
            is_fake=is_fake,
            confidence=result.confidence,
            probabilities=result.probabilities,
            processing_time_seconds=processing_duration,
            inference_time_ms=inference_time_ms,
            metadata=metadata,
            record_id=record_id
        )
        
        logger.success(
            f"Image detection completed: {result.prediction} "
            f"({result.confidence:.2%} confidence)"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Image validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
        
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Image detection failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image processing failed: {str(e)}"
        )
        
    finally:
        # Clean up temporary file (after copying to persistent storage)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {str(e)}")


# =============================================================================
# Additional Endpoints (Future Enhancements)
# =============================================================================

@router.post(
    "/predict/image/batch",
    summary="Batch image detection",
    description="Analyze multiple images in a single request (coming soon)",
    tags=["image"],
    include_in_schema=False  # Hide from docs until implemented
)
async def predict_image_batch():
    """
    Batch processing endpoint for multiple images.
    
    TODO: Implement batch processing
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Batch processing not yet implemented"
    )
