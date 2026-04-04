"""
Detection service for the detection API.

Pipeline:
    1. Validate upload safety
    2. Sanitize filename
    3. Validate file extension
    4. Save uploaded file to temp storage
    5. Validate file content
    6. Prepare input for detector
    7. Run detector
    8. Persist media file
    9. Save detection record to database
    10. Return detection result
"""

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, List, Optional

from fastapi import HTTPException, Request, UploadFile, status
from loguru import logger
from sqlalchemy.orm import Session

from app.core.exceptions import NoFaceDetectedError, NoVoiceDetectedError
from app.core.validation import sanitize_filename, validate_upload_safety
from app.db import crud
from app.models.base import DetectionResult

PERSISTENT_MEDIA_DIR = os.path.join(os.getcwd(), "persistent_media")


async def save_upload_tmp(upload_file: UploadFile, max_bytes: int) -> str:
    suffix = Path(upload_file.filename).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        total = 0
        chunk_size = 1024 * 1024
        while chunk := await upload_file.read(chunk_size):
            total += len(chunk)
            if total > max_bytes:
                tmp.close()
                os.unlink(tmp.name)
                mb = max_bytes / (1024 * 1024)
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File too large (max {mb:.0f} MB)")
            tmp.write(chunk)
        return tmp.name


def validate_extension(filename: str, allowed: list) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid extension. Allowed: {', '.join(allowed)}")


def persist_media(tmp_path: str, safe_filename: str, *, transcode_video: bool = False) -> Optional[str]:
    try:
        os.makedirs(PERSISTENT_MEDIA_DIR, exist_ok=True)
        uid = str(uuid.uuid4())[:8]

        if transcode_video:
            stem = Path(safe_filename).stem
            unique_name = f"{uid}_{stem}.mp4"
            dest = os.path.join(PERSISTENT_MEDIA_DIR, unique_name)
            logger.debug("[PIPELINE] Transcoding video to web-safe MP4: {}", dest)
            _transcode_to_websafe_mp4(tmp_path, dest)
        else:
            unique_name = f"{uid}_{safe_filename}"
            dest = os.path.join(PERSISTENT_MEDIA_DIR, unique_name)
            shutil.copy2(tmp_path, dest)

        logger.debug("[PIPELINE] Media persisted to /media/{}", unique_name)
        return f"/media/{unique_name}"
    except Exception as e:
        logger.error("[PIPELINE] Failed to persist media '{}': {}", safe_filename, e)
        if transcode_video:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Failed to transcode video to web-safe MP4. Ensure FFmpeg is installed.",
            )
        return None


def log_to_db(
    db: Session,
    *,
    safe_filename: str,
    file_type: str,
    file_size: int,
    result: DetectionResult,
    processing_duration: float,
    session_id: Optional[str],
    media_path: Optional[str],
) -> Optional[str]:
    try:
        classification = "Deepfake" if result.prediction == "Deepfake" else "Real"
        model_version = (result.metadata or {}).get("model_type", f"{file_type.title()}DeepfakeNet-v1.0")
        record = crud.create_detection_record(db, {
            "file_name": safe_filename,
            "file_type": file_type,
            "file_size": file_size,
            "detection_score": result.confidence,
            "classification": classification,
            "model_version": model_version,
            "processing_duration": processing_duration,
            "session_id": session_id,
            "media_path": media_path,
        })
        logger.debug("[PIPELINE] Detection record saved: id={}, type={}, classification={}",
                     record.id, file_type, classification)
        return record.id
    except Exception as e:
        logger.error("[PIPELINE] Failed to save detection record for '{}': {}", safe_filename, e)
        return None


async def run_detection_pipeline(
    *,
    request: Request,
    file: UploadFile,
    session_id: Optional[str],
    db: Session,
    detector_type: str,
    allowed_extensions: List[str],
    max_bytes: int,
    validate_fn: Callable[[str, int, str], Any],
    open_fn: Optional[Callable[[str], Any]] = None,
    transcode_video: bool = False,
) -> dict:
    """
    Run the detection pipeline for a given file.
    """

    pipeline_start = time.time()
    logger.info("[PIPELINE] {} detection request received: file='{}'", detector_type.upper(), file.filename)

    try:
        validate_upload_safety(file.filename)
        safe_filename = sanitize_filename(file.filename)
    except ValueError as e:
        logger.warning("[PIPELINE] Upload rejected (security): {}", e)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    validate_extension(file.filename, allowed_extensions)

    tmp_path: Optional[str] = None
    try:
        logger.info("[PIPELINE] Step 1/6: Saving uploaded file to temp storage...")
        tmp_path = await save_upload_tmp(file, max_bytes)
        file_size = os.path.getsize(tmp_path)
        logger.info("[PIPELINE] Step 1/6 complete: temp_path={}, size={:.2f} MB",
                    tmp_path, file_size / (1024 * 1024))

        logger.info("[PIPELINE] Step 2/6: Validating file content...")
        try:
            validate_fn(tmp_path, file_size, safe_filename)
        except ValueError as e:
            logger.warning("[PIPELINE] Validation failed for '{}': {}", safe_filename, e)
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

        logger.info("[PIPELINE] Step 3/6: Preparing input for {} detector...", detector_type)
        detect_input = open_fn(tmp_path) if open_fn else tmp_path

        logger.info("[PIPELINE] Step 4/6 (inference): {} — '{}'", detector_type, safe_filename)
        factory = request.app.state.detector_factory
        detector = factory.get_detector(detector_type)
        start = time.time()
        try:
            result = detector.detect(detect_input)
        except (NoFaceDetectedError, NoVoiceDetectedError) as e:
            logger.warning("[PIPELINE] {} detection aborted for '{}': {}", detector_type, safe_filename, e)
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
        processing_duration = time.time() - start

        logger.info("[PIPELINE] Step 5/6: Persisting media file...")
        media_path = persist_media(tmp_path, safe_filename, transcode_video=transcode_video)

        logger.info("[PIPELINE] Step 6/6: Saving detection record to database...")
        record_id = log_to_db(
            db,
            safe_filename=safe_filename,
            file_type=detector_type,
            file_size=file_size,
            result=result,
            processing_duration=processing_duration,
            session_id=session_id,
            media_path=media_path,
        )

        total_time = time.time() - pipeline_start
        logger.info(
            "[PIPELINE] {} detection complete: file='{}', prediction={}, "
            "confidence={:.2%}, inference={:.3f}s, total_pipeline={:.3f}s, record_id={}",
            detector_type.upper(), safe_filename, result.prediction,
            result.confidence, result.processing_time, total_time, record_id,
        )

        return {
            "prediction": result.prediction,
            "is_deepfake": result.prediction == "Deepfake",
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "processing_time_seconds": processing_duration,
            "inference_time_ms": result.processing_time * 1000.0,
            "metadata": result.metadata or {},
            "record_id": record_id,
        }

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("[PIPELINE] {} request rejected for '{}': {}", detector_type, file.filename, e)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except Exception as e:
        logger.error("[PIPELINE] {} detection failed for '{}': {}", detector_type, file.filename, e, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"{detector_type.title()} processing failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _transcode_to_websafe_mp4(input_path: str, output_path: str) -> None:
    cmd = [
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
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg not found in PATH.") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {(proc.stderr or '').strip()[-800:]}")
