"""
============================================================================
Test Configuration - Pytest Fixtures and Setup
============================================================================
This module provides pytest fixtures for testing the FastAPI application.

Key Features:
    - In-memory SQLite database for isolated testing
    - FastAPI TestClient for API endpoint testing
    - Database session override to use test database
    - Automatic cleanup after tests

Author: Senior QA Engineer
Date: 2026-03-25
============================================================================
"""

import pytest
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.main import app
from app.db.database import Base, get_db
from app.db import models  # Import models to register them with Base


# ============================================================================
# Test Database Configuration
# ============================================================================

# Use file-based SQLite database for testing to avoid in-memory issues
# This ensures tests are isolated and don't affect the real database
# File will be created in temp directory and cleaned up after tests
import tempfile
import os

TEST_DB_FILE = os.path.join(tempfile.gettempdir(), "test_deepfake_detection.db")
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_FILE}"

# Create test engine with file-based database
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False  # Set to True for SQL query debugging
)

# Create session factory for test database
TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine
)


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def test_db() -> Generator[Session, None, None]:
    """
    Create a fresh test database for each test function.
    
    This fixture:
        1. Drops any existing tables (cleanup from previous test)
        2. Creates all tables in the in-memory database
        3. Yields a database session for the test
        4. Drops all tables after the test completes
    
    Scope: function (new database for each test)
    
    Usage:
        def test_something(test_db):
            # Use test_db session here
            record = crud.create_detection_record(test_db, {...})
            assert record.id is not None
    """
    # Drop any existing tables first (cleanup from previous tests)
    Base.metadata.drop_all(bind=test_engine)
    
    # Create all tables fresh
    Base.metadata.create_all(bind=test_engine)
    
    # Create a new session for the test
    db = TestingSessionLocal()
    
    try:
        yield db
    finally:
        db.close()
        # Drop all tables after test
        Base.metadata.drop_all(bind=test_engine)


# ============================================================================
# FastAPI Test Client Fixture
# ============================================================================

@pytest.fixture(scope="function")
def client(test_db: Session, mock_detector_factory) -> Generator[TestClient, None, None]:
    """
    Create a FastAPI TestClient for testing API endpoints.
    
    This fixture:
        1. Creates test database tables
        2. Overrides the database dependency to use test database
        3. Overrides the database engine for init_db()
        4. Mocks the detector factory to prevent real model loading
        5. Creates a TestClient instance
        6. Yields the client for testing
        7. Cleans up after the test
    
    The TestClient allows you to make HTTP requests to your FastAPI app
    without actually running a server.
    
    Usage:
        def test_root_endpoint(client):
            response = client.get("/")
            assert response.status_code == 200
    """
    # Override the database dependency to use test_db
    def _override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = _override_get_db
    
    # Override the database engine for init_db() to use test engine
    from app.db import database
    original_engine = database.engine
    database.engine = test_engine
    
    # Pre-set detector factory to mock BEFORE TestClient is created
    # This prevents the lifespan from loading real models
    original_factory = getattr(app.state, 'detector_factory', None)
    app.state.detector_factory = mock_detector_factory
    
    # Create the test client (lifespan will see the mocked factory)
    with TestClient(app) as test_client:
        yield test_client
    
    # Restore original engine and factory
    database.engine = original_engine
    if original_factory is not None:
        app.state.detector_factory = original_factory
    elif hasattr(app.state, 'detector_factory'):
        delattr(app.state, 'detector_factory')
    
    # Clear overrides after test
    app.dependency_overrides.clear()


# ============================================================================
# Mock Model Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_detection_result():
    """
    Create a mock detection result for testing.
    
    This fixture provides a sample DetectionResult object that can be used
    to mock model predictions without loading actual ML models.
    
    Usage:
        def test_prediction(mock_detection_result):
            # Use mock_detection_result as a return value for mocked detector
            pass
    """
    from app.models.base import DetectionResult
    
    return DetectionResult(
        prediction="deepfake",
        confidence=0.92,
        probabilities={
            "authentic": 0.08,
            "deepfake": 0.92
        },
        processing_time=0.123,
        metadata={
            "model_type": "MockDetector",
            "device": "cpu"
        }
    )


@pytest.fixture(scope="function")
def mock_authentic_result():
    """
    Create a mock detection result for authentic media.
    
    Usage:
        def test_authentic_prediction(mock_authentic_result):
            # Use mock_authentic_result for testing authentic media
            pass
    """
    from app.models.base import DetectionResult
    
    return DetectionResult(
        prediction="authentic",
        confidence=0.88,
        probabilities={
            "authentic": 0.88,
            "deepfake": 0.12
        },
        processing_time=0.098,
        metadata={
            "model_type": "MockDetector",
            "device": "cpu"
        }
    )


# ============================================================================
# Sample File Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def sample_image_file():
    """
    Create a sample image file for testing file uploads.
    
    This fixture creates a minimal valid PNG image in memory
    that can be used for testing file upload endpoints.
    
    Usage:
        def test_image_upload(client, sample_image_file):
            response = client.post(
                "/api/v1/predict/image",
                files={"file": sample_image_file}
            )
    """
    from io import BytesIO
    from PIL import Image
    
    # Create a simple 100x100 RGB image
    img = Image.new('RGB', (100, 100), color='red')
    
    # Save to BytesIO buffer
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Return as tuple (filename, file_object, content_type)
    return ("test_image.png", img_bytes, "image/png")


@pytest.fixture(scope="function")
def sample_invalid_file():
    """
    Create an invalid file for testing error handling.
    
    Usage:
        def test_invalid_upload(client, sample_invalid_file):
            response = client.post(
                "/api/v1/predict/image",
                files={"file": sample_invalid_file}
            )
            assert response.status_code == 400
    """
    from io import BytesIO
    
    # Create a text file disguised as an image
    invalid_content = BytesIO(b"This is not an image file")
    
    return ("fake_image.png", invalid_content, "image/png")


@pytest.fixture(scope="function")
def sample_jpg_image_file():
    """
    Create a sample JPG image file for testing.
    
    Usage:
        def test_jpg_upload(client, sample_jpg_image_file):
            response = client.post(
                "/api/v1/predict/image",
                files={"file": sample_jpg_image_file}
            )
    """
    from io import BytesIO
    from PIL import Image
    
    img = Image.new('RGB', (150, 150), color='blue')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG', quality=95)
    img_bytes.seek(0)
    
    return ("test_image.jpg", img_bytes, "image/jpeg")


@pytest.fixture(scope="function")
def sample_webp_image_file():
    """
    Create a sample WEBP image file for testing.
    
    Usage:
        def test_webp_upload(client, sample_webp_image_file):
            response = client.post(
                "/api/v1/predict/image",
                files={"file": sample_webp_image_file}
            )
    """
    from io import BytesIO
    from PIL import Image
    
    img = Image.new('RGB', (120, 120), color='green')
    img_bytes = BytesIO()
    img.save(img_bytes, format='WEBP', quality=90)
    img_bytes.seek(0)
    
    return ("test_image.webp", img_bytes, "image/webp")


@pytest.fixture(scope="function")
def sample_audio_file():
    """
    Create a sample audio file for testing audio uploads.
    
    This fixture creates a minimal valid WAV audio file in memory
    that can be used for testing audio upload endpoints.
    
    Usage:
        def test_audio_upload(client, sample_audio_file):
            response = client.post(
                "/api/v1/predict/audio",
                files={"file": sample_audio_file}
            )
    """
    from io import BytesIO
    import wave
    import struct
    
    # Create a simple 1-second mono audio file at 16kHz
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    
    # Generate a simple sine wave (440 Hz - A4 note)
    import math
    frequency = 440.0
    audio_data = []
    for i in range(num_samples):
        sample = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
        audio_data.append(sample)
    
    # Write WAV file to BytesIO
    audio_bytes = BytesIO()
    with wave.open(audio_bytes, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Pack audio data as 16-bit signed integers
        packed_data = struct.pack('<' + 'h' * len(audio_data), *audio_data)
        wav_file.writeframes(packed_data)
    
    audio_bytes.seek(0)
    
    # Return as tuple (filename, file_object, content_type)
    return ("test_audio.wav", audio_bytes, "audio/wav")


@pytest.fixture(scope="function")
def sample_video_file():
    """
    Create a sample video file for testing video uploads.
    
    This fixture creates a minimal valid MP4 video file in memory
    that can be used for testing video upload endpoints.
    
    Note: Creates a simple 2-second video with audio track.
    
    Usage:
        def test_video_upload(client, sample_video_file):
            response = client.post(
                "/api/v1/predict/video",
                files={"file": sample_video_file}
            )
    """
    from io import BytesIO
    import tempfile
    import os
    
    try:
        import cv2
        import numpy as np
        from moviepy.editor import VideoClip, AudioClip
        
        # Create a temporary file for the video
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp_path = tmp_file.name
        tmp_file.close()
        
        # Video parameters
        width, height = 224, 224
        fps = 10
        duration = 2.0
        num_frames = int(fps * duration)
        
        # Create video frames (simple colored frames)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
        
        for i in range(num_frames):
            # Create a frame with changing color
            color_value = int(255 * (i / num_frames))
            frame = np.full((height, width, 3), color_value, dtype=np.uint8)
            video_writer.write(frame)
        
        video_writer.release()
        
        # Read the video file into BytesIO
        with open(tmp_path, 'rb') as f:
            video_bytes = BytesIO(f.read())
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        video_bytes.seek(0)
        return ("test_video.mp4", video_bytes, "video/mp4")
        
    except ImportError:
        # If moviepy or cv2 not available, create a minimal MP4 container
        # This is a minimal valid MP4 file header
        minimal_mp4 = BytesIO(
            b'\x00\x00\x00\x20\x66\x74\x79\x70\x69\x73\x6f\x6d\x00\x00\x02\x00'
            b'\x69\x73\x6f\x6d\x69\x73\x6f\x32\x6d\x70\x34\x31\x00\x00\x00\x08'
            b'\x66\x72\x65\x65'
        )
        return ("test_video.mp4", minimal_mp4, "video/mp4")


@pytest.fixture(scope="function")
def sample_invalid_audio_file():
    """
    Create an invalid audio file for testing error handling.
    
    Usage:
        def test_invalid_audio_upload(client, sample_invalid_audio_file):
            response = client.post(
                "/api/v1/predict/audio",
                files={"file": sample_invalid_audio_file}
            )
            assert response.status_code == 400
    """
    from io import BytesIO
    
    # Create a text file disguised as an audio file
    invalid_content = BytesIO(b"This is not an audio file")
    
    return ("fake_audio.wav", invalid_content, "audio/wav")


@pytest.fixture(scope="function")
def sample_invalid_video_file():
    """
    Create an invalid video file for testing error handling.
    
    Usage:
        def test_invalid_video_upload(client, sample_invalid_video_file):
            response = client.post(
                "/api/v1/predict/video",
                files={"file": sample_invalid_video_file}
            )
            assert response.status_code == 400
    """
    from io import BytesIO
    
    # Create a text file disguised as a video file
    invalid_content = BytesIO(b"This is not a video file")
    
    return ("fake_video.mp4", invalid_content, "video/mp4")


@pytest.fixture(scope="function")
def mock_audio_detection_result():
    """
    Create a mock audio detection result for testing.
    
    Usage:
        def test_audio_prediction(mock_audio_detection_result):
            # Use as return value for mocked audio detector
            pass
    """
    from app.models.base import DetectionResult
    
    return DetectionResult(
        prediction="deepfake",
        confidence=0.89,
        probabilities={
            "authentic": 0.11,
            "deepfake": 0.89
        },
        processing_time=2.456,
        metadata={
            "model_type": "DualFeature_CNN_GRU",
            "architecture": {
                "stream_a": "Mel-Spectrogram → ResNet18",
                "stream_b": "LFCC → ResNet18",
                "fusion": "Attention-based",
                "temporal": "Bidirectional GRU"
            },
            "num_segments_analyzed": 3,
            "segment_duration": 4.0,
            "sample_rate": 16000,
            "device": "cpu"
        }
    )


@pytest.fixture(scope="function")
def mock_video_detection_result():
    """
    Create a mock video detection result for testing.
    
    Usage:
        def test_video_prediction(mock_video_detection_result):
            # Use as return value for mocked video detector
            pass
    """
    from app.models.base import DetectionResult
    
    return DetectionResult(
        prediction="deepfake",
        confidence=0.91,
        probabilities={
            "authentic": 0.09,
            "deepfake": 0.91
        },
        processing_time=8.234,
        metadata={
            "model_type": "TriStreamMultimodalNet",
            "architecture": {
                "spatial_stream": "CLIP + SRM/EfficientNet",
                "audio_stream": "DualCNN-GRU",
                "sync_stream": "CrossAttention + LSTM"
            },
            "num_frames": 15,
            "audio_duration": 4.0,
            "device": "cpu"
        }
    )


@pytest.fixture(scope="function")
def mock_audio_authentic_result():
    """
    Create a mock audio detection result for authentic audio.
    
    Usage:
        def test_audio_authentic(mock_audio_authentic_result):
            # Use for testing authentic audio detection
            pass
    """
    from app.models.base import DetectionResult
    
    return DetectionResult(
        prediction="authentic",
        confidence=0.86,
        probabilities={
            "authentic": 0.86,
            "deepfake": 0.14
        },
        processing_time=2.123,
        metadata={
            "model_type": "DualFeature_CNN_GRU",
            "num_segments_analyzed": 2,
            "device": "cpu"
        }
    )


@pytest.fixture(scope="function")
def mock_video_authentic_result():
    """
    Create a mock video detection result for authentic video.
    
    Usage:
        def test_video_authentic(mock_video_authentic_result):
            # Use for testing authentic video detection
            pass
    """
    from app.models.base import DetectionResult
    
    return DetectionResult(
        prediction="authentic",
        confidence=0.93,
        probabilities={
            "authentic": 0.93,
            "deepfake": 0.07
        },
        processing_time=7.891,
        metadata={
            "model_type": "TriStreamMultimodalNet",
            "num_frames": 15,
            "device": "cpu"
        }
    )


# ============================================================================
# Mock Detector Factory Fixture
# ============================================================================

@pytest.fixture(scope="function")
def mock_detector_factory(mock_detection_result):
    """
    Create a mock DetectorFactory for testing without loading real models.
    
    This fixture mocks the entire detector factory so tests don't need to:
        - Load heavy PyTorch models (saves time and memory)
        - Download CLIP models from HuggingFace
        - Perform actual GPU inference
    
    Usage:
        def test_with_mock_factory(client, mock_detector_factory):
            # The app will use the mocked factory instead of real models
            response = client.post("/api/v1/predict/image", ...)
    """
    from unittest.mock import Mock
    
    # Create mock detector
    mock_detector = Mock()
    mock_detector.detect.return_value = mock_detection_result
    
    # Create mock factory
    mock_factory = Mock()
    mock_factory.get_detector.return_value = mock_detector
    
    return mock_factory


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_files():
    """
    Clean up any temporary files created during testing.
    
    This runs once at the end of the entire test session.
    """
    yield
    
    # Cleanup logic here if needed
    # For example, remove temporary upload directories
    import shutil
    import os
    
    test_upload_dir = "test_uploads"
    if os.path.exists(test_upload_dir):
        shutil.rmtree(test_upload_dir)
    
    # Clean up test database file
    if os.path.exists(TEST_DB_FILE):
        try:
            os.remove(TEST_DB_FILE)
        except Exception:
            pass  # Ignore errors during cleanup
