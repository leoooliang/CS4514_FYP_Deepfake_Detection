"""
============================================================================
API Endpoint Tests - Pytest Test Suite
============================================================================
Comprehensive test suite for FastAPI endpoints.

Test Coverage:
    - Health check endpoints
    - Image detection endpoint
    - Telemetry/statistics endpoints
    - Error handling and validation
    - Database integration

Author: Senior QA Engineer
Date: 2026-03-25
============================================================================
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image

from app.models.base import DetectionResult
from app.db import crud


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthEndpoints:
    """Test suite for health check and root endpoints."""
    
    def test_root_endpoint(self, client: TestClient):
        """
        Test GET / - Root endpoint returns API information.
        """
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert "endpoints" in data
        
        # Verify status
        assert data["status"] == "operational"
        assert data["version"] == "1.0.0"
        
        # Verify endpoints are listed
        assert "image_detection" in data["endpoints"]
        assert "video_detection" in data["endpoints"]
        assert "audio_detection" in data["endpoints"]
    
    def test_health_check_endpoint(self, client: TestClient):
        """
        Test GET /health - Health check returns system status.
        """
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "timestamp" in data
        assert "models" in data
        
        # Verify status
        assert data["status"] == "healthy"
        
        # Verify models are listed
        assert "image_detector" in data["models"]
        assert "video_detector" in data["models"]
        assert "audio_detector" in data["models"]
        
        # Verify system metrics are present
        if "system" in data:
            assert "cpu_percent" in data["system"]
            assert "memory_percent" in data["system"]


# ============================================================================
# Telemetry Endpoint Tests
# ============================================================================

class TestTelemetryEndpoints:
    """Test suite for telemetry and statistics endpoints."""
    
    def test_get_stats_empty_database(self, client: TestClient, test_db):
        """
        Test GET /api/v1/telemetry/stats with empty database.
        """
        response = client.get("/api/v1/telemetry/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response schema
        assert "total_scans" in data
        assert "deepfakes_detected" in data
        assert "real_media_detected" in data
        assert "avg_processing_duration" in data
        assert "scans_by_type" in data
        assert "classification_breakdown" in data
        
        # Verify empty database returns zeros
        assert data["total_scans"] == 0
        assert data["deepfakes_detected"] == 0
        assert data["real_media_detected"] == 0
    
    def test_get_stats_with_data(self, client: TestClient, test_db):
        """
        Test GET /api/v1/telemetry/stats with sample data.
        """
        # Create sample detection records
        sample_records = [
            {
                "file_name": "test1.jpg",
                "file_type": "image",
                "file_size": 1024,
                "detection_score": 0.95,
                "classification": "Fake",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.5
            },
            {
                "file_name": "test2.jpg",
                "file_type": "image",
                "file_size": 2048,
                "detection_score": 0.85,
                "classification": "Real",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.2
            },
            {
                "file_name": "test3.mp4",
                "file_type": "video",
                "file_size": 5120,
                "detection_score": 0.78,
                "classification": "Fake",
                "model_version": "TestModel-v1.0",
                "processing_duration": 2.3
            }
        ]
        
        # Insert records into test database
        for record_data in sample_records:
            crud.create_detection_record(test_db, record_data)
        
        # Query stats endpoint
        response = client.get("/api/v1/telemetry/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify aggregated statistics
        assert data["total_scans"] == 3
        assert data["deepfakes_detected"] == 2
        assert data["real_media_detected"] == 1
        
        # Verify breakdown by type
        assert data["scans_by_type"]["image"] == 2
        assert data["scans_by_type"]["video"] == 1
        
        # Verify classification breakdown
        assert data["classification_breakdown"]["Fake"] == 2
        assert data["classification_breakdown"]["Real"] == 1
        
        # Verify average processing duration
        expected_avg = (1.5 + 1.2 + 2.3) / 3
        assert abs(data["avg_processing_duration"] - expected_avg) < 0.01
    
    def test_get_history_empty(self, client: TestClient, test_db):
        """
        Test GET /api/v1/telemetry/history with empty database.
        """
        response = client.get("/api/v1/telemetry/history")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total" in data
        assert "records" in data
        
        # Verify empty results
        assert data["total"] == 0
        assert len(data["records"]) == 0
    
    def test_get_history_with_limit(self, client: TestClient, test_db):
        """
        Test GET /api/v1/telemetry/history with limit parameter.
        """
        # Create 5 sample records
        for i in range(5):
            crud.create_detection_record(test_db, {
                "file_name": f"test{i}.jpg",
                "file_type": "image",
                "file_size": 1024 * (i + 1),
                "detection_score": 0.8 + (i * 0.02),
                "classification": "Fake" if i % 2 == 0 else "Real",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.0 + (i * 0.1)
            })
        
        # Query with limit=3
        response = client.get("/api/v1/telemetry/history?limit=3")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify limit is respected
        assert data["total"] == 3
        assert len(data["records"]) == 3
        
        # Verify records have required fields
        for record in data["records"]:
            assert "id" in record
            assert "file_name" in record
            assert "file_type" in record
            assert "detection_score" in record
            assert "classification" in record
            assert "timestamp" in record
    
    def test_get_result_by_id(self, client: TestClient, test_db):
        """
        Test GET /api/v1/telemetry/results/{id} - Get specific record.
        """
        # Create a sample record
        record = crud.create_detection_record(test_db, {
            "file_name": "test.jpg",
            "file_type": "image",
            "file_size": 2048,
            "detection_score": 0.92,
            "classification": "Fake",
            "model_version": "TestModel-v1.0",
            "processing_duration": 1.5
        })
        
        # Query by ID
        response = client.get(f"/api/v1/telemetry/results/{record.id}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify correct record is returned
        assert data["id"] == record.id
        assert data["file_name"] == "test.jpg"
        assert data["file_type"] == "image"
        assert data["classification"] == "Fake"
        assert data["detection_score"] == 0.92
    
    def test_get_result_not_found(self, client: TestClient, test_db):
        """
        Test GET /api/v1/telemetry/results/{id} with non-existent ID.
        """
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/v1/telemetry/results/{fake_id}")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


# ============================================================================
# Image Detection Endpoint Tests
# ============================================================================

class TestImageDetection:
    """Test suite for image deepfake detection endpoint."""
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_success(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test POST /api/v1/predict/image - Successful image detection.
        
        This test mocks the ImageDetector to avoid loading heavy PyTorch models.
        """
        # Configure mock detector
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        # Configure mock factory to return our mock detector
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        # Override app state with mock factory
        client.app.state.detector_factory = mock_factory
        
        # Make request with sample image
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Verify response schema
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "processing_time_seconds" in data
        assert "inference_time_ms" in data
        assert "metadata" in data
        
        # Verify prediction values
        assert data["prediction"] == "deepfake"
        assert data["confidence"] == 0.92
        assert data["is_fake"] is True
        
        # Verify probabilities
        assert "authentic" in data["probabilities"]
        assert "deepfake" in data["probabilities"]
        assert data["probabilities"]["deepfake"] == 0.92
        
        # Verify database record was created
        assert "record_id" in data
        assert data["record_id"] is not None
        
        # Verify detector was called
        mock_factory.get_detector.assert_called_once_with("image")
        mock_detector_instance.detect.assert_called_once()
    
    def test_predict_image_invalid_extension(self, client: TestClient):
        """
        Test POST /api/v1/predict/image with invalid file extension.
        """
        # Create a file with invalid extension
        invalid_file = BytesIO(b"fake content")
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("test.txt", invalid_file, "text/plain")}
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_predict_image_corrupted_file(self, client: TestClient, sample_invalid_file):
        """
        Test POST /api/v1/predict/image with corrupted image file.
        """
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_invalid_file}
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_authentic(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_authentic_result
    ):
        """
        Test POST /api/v1/predict/image - Detection of authentic image.
        """
        # Configure mock detector for authentic result
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_authentic_result
        
        # Configure mock factory
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        # Override app state
        client.app.state.detector_factory = mock_factory
        
        # Make request
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Verify authentic prediction
        assert data["prediction"] == "authentic"
        assert data["is_fake"] is False
        assert data["confidence"] == 0.88
        
        # Verify database record classification
        record_id = data["record_id"]
        record = crud.get_record_by_id(test_db, record_id)
        assert record is not None
        assert record.classification == "Real"
    
    def test_predict_image_no_file(self, client: TestClient):
        """
        Test POST /api/v1/predict/image without file parameter.
        """
        response = client.post("/api/v1/predict/image")
        
        # Should return 422 Unprocessable Entity (missing required field)
        assert response.status_code == 422
    
    @pytest.mark.parametrize("image_format,content_type", [
        (".jpg", "image/jpeg"),
        (".jpeg", "image/jpeg"),
        (".png", "image/png"),
        (".bmp", "image/bmp"),
        (".webp", "image/webp"),
    ])
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_various_formats(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result,
        image_format: str,
        content_type: str
    ):
        """
        Test image detection with various image formats.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        img = Image.new('RGB', (100, 100), color='blue')
        img_bytes = BytesIO()
        
        save_format = 'JPEG' if image_format in ['.jpg', '.jpeg'] else image_format[1:].upper()
        img.save(img_bytes, format=save_format)
        img_bytes.seek(0)
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": (f"test{image_format}", img_bytes, content_type)}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "deepfake"
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_different_sizes(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test image detection with various image dimensions.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        test_sizes = [(64, 64), (224, 224), (512, 512), (1920, 1080)]
        
        for width, height in test_sizes:
            img = Image.new('RGB', (width, height), color='green')
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            response = client.post(
                "/api/v1/predict/image",
                files={"file": (f"test_{width}x{height}.png", img_bytes, "image/png")}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == "deepfake"
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_grayscale(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test image detection with grayscale image.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        img = Image.new('L', (100, 100), color=128)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("grayscale.png", img_bytes, "image/png")}
        )
        
        assert response.status_code in [200, 400]
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_rgba(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test image detection with RGBA (transparent) image.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("transparent.png", img_bytes, "image/png")}
        )
        
        assert response.status_code in [200, 400]
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_high_confidence_deepfake(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file
    ):
        """
        Test image detection with very high confidence deepfake.
        """
        from app.models.base import DetectionResult
        
        high_confidence_result = DetectionResult(
            prediction="deepfake",
            confidence=0.99,
            probabilities={"authentic": 0.01, "deepfake": 0.99},
            processing_time=1.1,
            metadata={"model_type": "DualStreamFusion", "device": "cpu"}
        )
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = high_confidence_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["prediction"] == "deepfake"
        assert data["confidence"] == 0.99
        assert data["probabilities"]["deepfake"] == 0.99
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_low_confidence(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file
    ):
        """
        Test image detection with low confidence (near threshold).
        """
        from app.models.base import DetectionResult
        
        low_confidence_result = DetectionResult(
            prediction="deepfake",
            confidence=0.52,
            probabilities={"authentic": 0.48, "deepfake": 0.52},
            processing_time=1.0,
            metadata={"model_type": "DualStreamFusion", "device": "cpu"}
        )
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = low_confidence_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["prediction"] == "deepfake"
        assert data["confidence"] == 0.52
        assert 0.5 <= data["confidence"] <= 0.6
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_metadata_fields(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file
    ):
        """
        Test that image detection returns comprehensive metadata.
        """
        from app.models.base import DetectionResult
        
        detailed_result = DetectionResult(
            prediction="deepfake",
            confidence=0.87,
            probabilities={"authentic": 0.13, "deepfake": 0.87},
            processing_time=1.5,
            metadata={
                "model_type": "DualStreamFusion",
                "device": "cpu",
                "architecture": {
                    "clip_stream": "CLIP ViT-L/14",
                    "noise_stream": "SRM + EfficientNet-B0"
                },
                "image_size": [224, 224],
                "preprocessing": {
                    "normalization": "CLIP",
                    "resize_method": "center_crop"
                }
            }
        )
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = detailed_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        metadata = data["metadata"]
        assert "model_type" in metadata
        assert "device" in metadata
        assert "architecture" in metadata
        assert metadata["model_type"] == "DualStreamFusion"
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_processing_time(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test that processing time is recorded correctly.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "processing_time_seconds" in data
        assert "inference_time_ms" in data
        assert data["processing_time_seconds"] >= 0
        assert data["inference_time_ms"] == pytest.approx(mock_detection_result.processing_time * 1000.0)
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_multiple_sequential(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test multiple sequential image detections.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        record_ids = []
        
        for i in range(3):
            img = Image.new('RGB', (100, 100), color=(i * 80, 0, 0))
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            response = client.post(
                "/api/v1/predict/image",
                files={"file": (f"test_{i}.png", img_bytes, "image/png")}
            )
            
            assert response.status_code == 200
            data = response.json()
            record_ids.append(data["record_id"])
        
        assert len(record_ids) == 3
        assert len(set(record_ids)) == 3
        
        stats_response = client.get("/api/v1/telemetry/stats")
        assert stats_response.status_code == 200
        stats_data = stats_response.json()
        assert stats_data["total_scans"] == 3
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_response_schema_compliance(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test that image response complies with ImagePredictionResponse schema.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "prediction", "is_fake", "confidence",
            "probabilities", "processing_time_seconds", "inference_time_ms",
            "metadata", "record_id"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert isinstance(data["probabilities"], dict)
        assert "authentic" in data["probabilities"]
        assert "deepfake" in data["probabilities"]
        
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["processing_time_seconds"], (int, float))
        assert isinstance(data["inference_time_ms"], (int, float))
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_probabilities_sum_to_one(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test that image detection probabilities sum to approximately 1.0.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        probabilities = data["probabilities"]
        prob_sum = probabilities["authentic"] + probabilities["deepfake"]
        
        assert abs(prob_sum - 1.0) < 0.01
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_very_small_image(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test image detection with very small image (edge case).
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        img = Image.new('RGB', (32, 32), color='yellow')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("tiny.png", img_bytes, "image/png")}
        )
        
        assert response.status_code in [200, 400]
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_square_vs_rectangular(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test image detection with square and rectangular images.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        test_dimensions = [
            (100, 100),
            (200, 100),
            (100, 200),
            (400, 300)
        ]
        
        for width, height in test_dimensions:
            img = Image.new('RGB', (width, height), color='purple')
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            response = client.post(
                "/api/v1/predict/image",
                files={"file": (f"test_{width}x{height}.png", img_bytes, "image/png")}
            )
            
            assert response.status_code == 200
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_confidence_matches_probabilities(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file
    ):
        """
        Test that confidence matches the maximum probability.
        """
        from app.models.base import DetectionResult
        
        test_result = DetectionResult(
            prediction="authentic",
            confidence=0.75,
            probabilities={"authentic": 0.75, "deepfake": 0.25},
            processing_time=1.0,
            metadata={"model_type": "DualStreamFusion", "device": "cpu"}
        )
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = test_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        max_prob = max(data["probabilities"].values())
        assert data["confidence"] == max_prob
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_empty_metadata(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file
    ):
        """
        Test image detection when detector returns minimal metadata.
        """
        from app.models.base import DetectionResult
        
        minimal_result = DetectionResult(
            prediction="deepfake",
            confidence=0.80,
            probabilities={"authentic": 0.20, "deepfake": 0.80},
            processing_time=1.0,
            metadata={}
        )
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = minimal_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "metadata" in data
        assert isinstance(data["metadata"], dict)


# ============================================================================
# Database Integration Tests
# ============================================================================

class TestImageDetectionEdgeCases:
    """Test edge cases and security for image detection."""
    
    def test_predict_image_malicious_filename(self, client: TestClient):
        """
        Test image detection with potentially malicious filename.
        """
        from io import BytesIO
        from PIL import Image
        
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        malicious_names = [
            "../../../etc/passwd.png",
            "..\\..\\..\\windows\\system32\\config.png",
            "test<script>alert('xss')</script>.png",
            "test'; DROP TABLE users; --.png"
        ]
        
        for malicious_name in malicious_names:
            img_bytes.seek(0)
            response = client.post(
                "/api/v1/predict/image",
                files={"file": (malicious_name, img_bytes, "image/png")}
            )
            
            assert response.status_code in [200, 400]
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_unicode_filename(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test image detection with Unicode characters in filename.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        img = Image.new('RGB', (100, 100), color='cyan')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("测试图片_日本語_한글.png", img_bytes, "image/png")}
        )
        
        assert response.status_code == 200
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_long_filename(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test image detection with very long filename.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        img = Image.new('RGB', (100, 100), color='orange')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        long_name = "a" * 200 + ".png"
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": (long_name, img_bytes, "image/png")}
        )
        
        assert response.status_code in [200, 400]
    
    def test_predict_image_empty_file(self, client: TestClient):
        """
        Test image detection with empty file.
        """
        from io import BytesIO
        
        empty_file = BytesIO(b"")
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("empty.png", empty_file, "image/png")}
        )
        
        assert response.status_code == 400
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_special_characters_in_filename(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test image detection with special characters in filename.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        img = Image.new('RGB', (100, 100), color='magenta')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        special_names = [
            "test image with spaces.png",
            "test-image-with-dashes.png",
            "test_image_with_underscores.png",
            "test.multiple.dots.png"
        ]
        
        for special_name in special_names:
            img_bytes.seek(0)
            response = client.post(
                "/api/v1/predict/image",
                files={"file": (special_name, img_bytes, "image/png")}
            )
            
            assert response.status_code == 200


class TestDatabaseIntegration:
    """Test database operations and persistence."""
    
    @patch('app.models.image_detector.ImageDetector')
    def test_detection_persists_to_database(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test that detection results are properly logged to the database.
        """
        # Configure mock detector
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        # Perform detection
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify record was created
        record_id = data["record_id"]
        assert record_id is not None
        
        # Query database directly
        record = crud.get_record_by_id(test_db, record_id)
        
        # Verify record fields
        assert record is not None
        assert record.file_name == "test_image.png"
        assert record.file_type == "image"
        assert record.classification == "Fake"
        assert record.detection_score == 0.92
        assert record.processing_duration >= 0  # Can be 0 for mocked detectors
    
    def test_history_returns_recent_records(self, client: TestClient, test_db):
        """
        Test that history endpoint returns records in correct order.
        """
        # Create multiple records with different timestamps
        import time
        
        record_ids = []
        for i in range(3):
            record = crud.create_detection_record(test_db, {
                "file_name": f"test{i}.jpg",
                "file_type": "image",
                "file_size": 1024,
                "detection_score": 0.8,
                "classification": "Real",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.0
            })
            record_ids.append(record.id)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Query history
        response = client.get("/api/v1/telemetry/history")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all records are returned
        assert data["total"] == 3
        assert len(data["records"]) == 3
        
        # Verify records are ordered by timestamp (newest first)
        # The last created record should be first in the response
        assert data["records"][0]["id"] == record_ids[-1]


# ============================================================================
# Mock Detector Tests
# ============================================================================

class TestImageDetectionAdvanced:
    """Advanced tests for image detection including performance and concurrency."""
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_with_different_confidence_levels(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db
    ):
        """
        Test image detection with various confidence levels.
        """
        from app.models.base import DetectionResult
        
        confidence_levels = [0.51, 0.65, 0.75, 0.85, 0.95, 0.99]
        
        for confidence in confidence_levels:
            test_result = DetectionResult(
                prediction="deepfake" if confidence > 0.5 else "authentic",
                confidence=confidence,
                probabilities={
                    "authentic": 1.0 - confidence,
                    "deepfake": confidence
                },
                processing_time=1.0,
                metadata={"model_type": "DualStreamFusion", "device": "cpu"}
            )
            
            mock_detector_instance = Mock()
            mock_detector_instance.detect.return_value = test_result
            
            mock_factory = Mock()
            mock_factory.get_detector.return_value = mock_detector_instance
            client.app.state.detector_factory = mock_factory
            
            img = Image.new('RGB', (100, 100), color='red')
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            response = client.post(
                "/api/v1/predict/image",
                files={"file": (f"test_{confidence}.png", img_bytes, "image/png")}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["confidence"] == confidence
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_batch_simulation(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test multiple image detections simulating batch processing.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        num_images = 5
        results = []
        
        for i in range(num_images):
            img = Image.new('RGB', (100, 100), color=(i * 50, 0, 0))
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            response = client.post(
                "/api/v1/predict/image",
                files={"file": (f"batch_{i}.png", img_bytes, "image/png")}
            )
            
            assert response.status_code == 200
            results.append(response.json())
        
        assert len(results) == num_images
        
        stats_response = client.get("/api/v1/telemetry/stats")
        stats_data = stats_response.json()
        assert stats_data["total_scans"] == num_images
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_case_insensitive_extension(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test that file extensions are case-insensitive.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        extensions = [".PNG", ".Png", ".pNg", ".JPG", ".JpEg"]
        
        for ext in extensions:
            img = Image.new('RGB', (100, 100), color='white')
            img_bytes = BytesIO()
            save_format = 'JPEG' if 'jpg' in ext.lower() or 'jpeg' in ext.lower() else 'PNG'
            img.save(img_bytes, format=save_format)
            img_bytes.seek(0)
            
            content_type = "image/jpeg" if 'jpg' in ext.lower() or 'jpeg' in ext.lower() else "image/png"
            
            response = client.post(
                "/api/v1/predict/image",
                files={"file": (f"test{ext}", img_bytes, content_type)}
            )
            
            assert response.status_code == 200
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_detector_called_with_correct_input(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test that detector is called with correct input type.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        
        mock_detector_instance.detect.assert_called_once()
        call_args = mock_detector_instance.detect.call_args[0]
        assert len(call_args) > 0


class TestMockDetectorIntegration:
    """Test that mocking works correctly for different scenarios."""
    
    def test_mock_detector_factory_called(
        self,
        client: TestClient,
        sample_image_file,
        mock_detector_factory
    ):
        """
        Test that the mock detector factory is properly invoked.
        """
        # Override app state with our mock factory
        client.app.state.detector_factory = mock_detector_factory
        
        # Make request
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        # Verify mock was called
        mock_detector_factory.get_detector.assert_called_with("image")
    
    @patch('app.models.factory.DetectorFactory')
    def test_detector_factory_initialization(self, mock_factory_class):
        """
        Test that DetectorFactory can be mocked at the class level.
        """
        # Create mock factory instance
        mock_factory_instance = Mock()
        mock_factory_class.return_value = mock_factory_instance
        
        # Import and use factory
        from app.models.factory import DetectorFactory
        
        factory = DetectorFactory()
        
        # Verify mock was used
        assert factory == mock_factory_instance


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_validation_error_format(self, client: TestClient):
        """
        Test that validation errors return proper format.
        """
        # Send invalid query parameter
        response = client.get("/api/v1/telemetry/history?limit=invalid")
        
        # Should return 422 Unprocessable Entity
        assert response.status_code == 422
        data = response.json()
        
        # Verify error response structure (custom handler format)
        assert "error" in data
        assert "message" in data
        assert "details" in data  # Our custom handler uses "details" (plural)
    
    def test_invalid_endpoint(self, client: TestClient):
        """
        Test that invalid endpoints return 404.
        """
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
    
    @patch('app.models.audio_detector.AudioDetector')
    def test_audio_detection_error_handling(
        self,
        mock_audio_detector_class,
        client: TestClient,
        sample_audio_file
    ):
        """
        Test error handling when audio detector fails.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.side_effect = RuntimeError("Model inference failed")
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    @patch('app.models.video_detector.VideoDetector')
    def test_video_detection_error_handling(
        self,
        mock_video_detector_class,
        client: TestClient,
        sample_video_file
    ):
        """
        Test error handling when video detector fails.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.side_effect = RuntimeError("Video processing failed")
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/video",
            files={"file": sample_video_file}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


# ============================================================================
# Integration Tests with Full Flow
# ============================================================================

class TestFullDetectionFlow:
    """End-to-end integration tests for complete detection flow."""
    
    @patch('app.models.image_detector.ImageDetector')
    def test_full_image_detection_flow(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db
    ):
        """
        Test complete flow: upload image -> detect -> save to DB -> query stats.
        """
        # Step 1: Create mock detector
        mock_result = DetectionResult(
            prediction="deepfake",
            confidence=0.95,
            probabilities={"authentic": 0.05, "deepfake": 0.95},
            processing_time=1.234,
            metadata={"model_type": "DualStreamFusion", "device": "cpu"}
        )
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        # Step 2: Upload and detect image
        img = Image.new('RGB', (224, 224), color='blue')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        assert response.status_code == 200
        detection_data = response.json()
        
        # Verify detection response
        assert detection_data["prediction"] == "deepfake"
        assert detection_data["confidence"] == 0.95
        record_id = detection_data["record_id"]
        
        # Step 3: Query stats to verify record was saved
        stats_response = client.get("/api/v1/telemetry/stats")
        assert stats_response.status_code == 200
        stats_data = stats_response.json()
        
        # Verify stats updated
        assert stats_data["total_scans"] == 1
        assert stats_data["deepfakes_detected"] == 1
        
        # Step 4: Query specific record
        record_response = client.get(f"/api/v1/telemetry/results/{record_id}")
        assert record_response.status_code == 200
        record_data = record_response.json()
        
        # Verify record details
        assert record_data["id"] == record_id
        assert record_data["classification"] == "Fake"
        assert record_data["file_type"] == "image"
    
    @patch('app.models.audio_detector.AudioDetector')
    def test_full_audio_detection_flow(
        self,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        sample_audio_file
    ):
        """
        Test complete flow: upload audio -> detect -> save to DB -> query stats.
        """
        mock_result = DetectionResult(
            prediction="deepfake",
            confidence=0.88,
            probabilities={"authentic": 0.12, "deepfake": 0.88},
            processing_time=2.567,
            metadata={
                "model_type": "DualFeature_CNN_GRU",
                "num_segments_analyzed": 4,
                "device": "cpu"
            }
        )
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file}
        )
        
        assert response.status_code == 200
        detection_data = response.json()
        
        assert detection_data["prediction"] == "deepfake"
        assert detection_data["confidence"] == 0.88
        record_id = detection_data["record_id"]
        
        stats_response = client.get("/api/v1/telemetry/stats")
        assert stats_response.status_code == 200
        stats_data = stats_response.json()
        
        assert stats_data["total_scans"] == 1
        assert stats_data["deepfakes_detected"] == 1
        assert stats_data["scans_by_type"]["audio"] == 1
        
        record_response = client.get(f"/api/v1/telemetry/results/{record_id}")
        assert record_response.status_code == 200
        record_data = record_response.json()
        
        assert record_data["id"] == record_id
        assert record_data["classification"] == "Fake"
        assert record_data["file_type"] == "audio"
    
    @patch('app.models.video_detector.VideoDetector')
    def test_full_video_detection_flow(
        self,
        mock_video_detector_class,
        client: TestClient,
        test_db,
        sample_video_file
    ):
        """
        Test complete flow: upload video -> detect -> save to DB -> query stats.
        """
        mock_result = DetectionResult(
            prediction="authentic",
            confidence=0.94,
            probabilities={"authentic": 0.94, "deepfake": 0.06},
            processing_time=9.123,
            metadata={
                "model_type": "TriStreamMultimodalNet",
                "num_frames": 20,
                "device": "cpu"
            }
        )
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/video",
            files={"file": sample_video_file}
        )
        
        assert response.status_code == 200
        detection_data = response.json()
        
        assert detection_data["prediction"] == "authentic"
        assert detection_data["confidence"] == 0.94
        assert detection_data["is_fake"] is False
        record_id = detection_data["record_id"]
        
        stats_response = client.get("/api/v1/telemetry/stats")
        assert stats_response.status_code == 200
        stats_data = stats_response.json()
        
        assert stats_data["total_scans"] == 1
        assert stats_data["real_media_detected"] == 1
        assert stats_data["scans_by_type"]["video"] == 1
        
        record_response = client.get(f"/api/v1/telemetry/results/{record_id}")
        assert record_response.status_code == 200
        record_data = record_response.json()
        
        assert record_data["id"] == record_id
        assert record_data["classification"] == "Real"
        assert record_data["file_type"] == "video"


# ============================================================================
# Performance Tests
# ============================================================================

class TestImageDetectionBoundaryConditions:
    """Test boundary conditions and extreme cases for image detection."""
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_minimum_size(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test image detection with minimum acceptable size (1x1 pixel).
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        img = Image.new('RGB', (1, 1), color='black')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("tiny_1x1.png", img_bytes, "image/png")}
        )
        
        assert response.status_code in [200, 400]
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_exact_threshold_confidence(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file
    ):
        """
        Test image detection with confidence exactly at 0.5 threshold.
        """
        from app.models.base import DetectionResult
        
        threshold_result = DetectionResult(
            prediction="authentic",
            confidence=0.50,
            probabilities={"authentic": 0.50, "deepfake": 0.50},
            processing_time=1.0,
            metadata={"model_type": "DualStreamFusion", "device": "cpu"}
        )
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = threshold_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["confidence"] == 0.50
        assert data["prediction"] in ["authentic", "deepfake"]
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_multiple_formats_same_session(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test detecting images in different formats in the same session.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        formats = [
            ('PNG', 'image/png', '.png'),
            ('JPEG', 'image/jpeg', '.jpg'),
            ('BMP', 'image/bmp', '.bmp')
        ]
        
        for save_format, content_type, extension in formats:
            img = Image.new('RGB', (100, 100), color='red')
            img_bytes = BytesIO()
            img.save(img_bytes, format=save_format)
            img_bytes.seek(0)
            
            response = client.post(
                "/api/v1/predict/image",
                files={"file": (f"test{extension}", img_bytes, content_type)}
            )
            
            assert response.status_code == 200
        
        stats_response = client.get("/api/v1/telemetry/stats")
        stats_data = stats_response.json()
        assert stats_data["total_scans"] == len(formats)
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_with_exif_data(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test image detection with EXIF metadata.
        """
        from PIL.ExifTags import TAGS
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        img = Image.new('RGB', (100, 100), color='yellow')
        
        exif_dict = {
            271: "Test Camera",
            272: "Test Model",
            306: "2026:03:25 10:30:00"
        }
        
        from PIL import Image as PILImage
        exif_bytes = PILImage.Exif()
        for tag_id, value in exif_dict.items():
            exif_bytes[tag_id] = value
        
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG', exif=exif_bytes)
        img_bytes.seek(0)
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("exif_image.jpg", img_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
    
    @patch('app.models.image_detector.ImageDetector')
    def test_predict_image_content_type_mismatch(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        mock_detection_result
    ):
        """
        Test image detection when content-type doesn't match file extension.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        img = Image.new('RGB', (100, 100), color='pink')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("test.png", img_bytes, "image/jpeg")}
        )
        
        assert response.status_code in [200, 400]


class TestMockDetectorIntegration:
    """Test that mocking works correctly for different scenarios."""
    
    def test_mock_detector_factory_called(
        self,
        client: TestClient,
        sample_image_file,
        mock_detector_factory
    ):
        """
        Test that the mock detector factory is properly invoked.
        """
        # Override app state with our mock factory
        client.app.state.detector_factory = mock_detector_factory
        
        # Make request
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        # Verify mock was called
        mock_detector_factory.get_detector.assert_called_with("image")
    
    @patch('app.models.factory.DetectorFactory')
    def test_detector_factory_initialization(self, mock_factory_class):
        """
        Test that DetectorFactory can be mocked at the class level.
        """
        # Create mock factory instance
        mock_factory_instance = Mock()
        mock_factory_class.return_value = mock_factory_instance
        
        # Import and use factory
        from app.models.factory import DetectorFactory
        
        factory = DetectorFactory()
        
        # Verify mock was used
        assert factory == mock_factory_instance


class TestPerformance:
    """Test performance-related features."""
    
    def test_response_includes_process_time_header(self, client: TestClient):
        """
        Test that responses include X-Process-Time header.
        """
        response = client.get("/")
        
        # Verify header is present
        assert "X-Process-Time" in response.headers
        
        # Verify it's a valid float
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0
    
    @patch('app.models.image_detector.ImageDetector')
    def test_image_detection_performance_tracking(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test that image detection tracks performance metrics.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "processing_time_seconds" in data
        assert "inference_time_ms" in data
        assert data["processing_time_seconds"] >= 0
        assert data["inference_time_ms"] >= 0
        
        record_id = data["record_id"]
        record = crud.get_record_by_id(test_db, record_id)
        assert record.processing_duration >= 0


# ============================================================================
# Parametrized Tests
# ============================================================================

class TestParametrizedScenarios:
    """Test multiple scenarios using pytest parametrize."""
    
    @pytest.mark.parametrize("limit,expected_max", [
        (5, 5),
        (10, 10),
        (100, 100),
        (1, 1),
    ])
    def test_history_limit_variations(
        self,
        client: TestClient,
        test_db,
        limit: int,
        expected_max: int
    ):
        """
        Test history endpoint with various limit values.
        """
        # Create 15 sample records
        for i in range(15):
            crud.create_detection_record(test_db, {
                "file_name": f"test{i}.jpg",
                "file_type": "image",
                "file_size": 1024,
                "detection_score": 0.8,
                "classification": "Real",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.0
            })
        
        # Query with specific limit
        response = client.get(f"/api/v1/telemetry/history?limit={limit}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify limit is respected
        assert len(data["records"]) <= expected_max
        assert data["total"] <= expected_max
    
    @pytest.mark.parametrize("file_type,classification,expected_count", [
        ("image", "Fake", 1),
        ("video", "Real", 1),
        ("audio", "Fake", 1),
    ])
    def test_stats_breakdown_by_type(
        self,
        client: TestClient,
        test_db,
        file_type: str,
        classification: str,
        expected_count: int
    ):
        """
        Test statistics breakdown by file type and classification.
        """
        # Create a record with specific type and classification
        crud.create_detection_record(test_db, {
            "file_name": f"test.{file_type}",
            "file_type": file_type,
            "file_size": 1024,
            "detection_score": 0.9,
            "classification": classification,
            "model_version": "TestModel-v1.0",
            "processing_duration": 1.0
        })
        
        # Query stats
        response = client.get("/api/v1/telemetry/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify breakdown
        assert data["scans_by_type"][file_type] == expected_count
        assert data["classification_breakdown"][classification] == expected_count


class TestMultimodalStatistics:
    """Test statistics aggregation across all media types."""
    
    def test_stats_with_mixed_media_types(self, client: TestClient, test_db):
        """
        Test statistics with image, audio, and video records.
        """
        sample_records = [
            {
                "file_name": "image1.jpg",
                "file_type": "image",
                "file_size": 2048,
                "detection_score": 0.92,
                "classification": "Fake",
                "model_version": "ImageDetector-v1.0",
                "processing_duration": 1.2
            },
            {
                "file_name": "audio1.wav",
                "file_type": "audio",
                "file_size": 5120,
                "detection_score": 0.87,
                "classification": "Fake",
                "model_version": "AudioDetector-v1.0",
                "processing_duration": 2.5
            },
            {
                "file_name": "video1.mp4",
                "file_type": "video",
                "file_size": 10240,
                "detection_score": 0.78,
                "classification": "Real",
                "model_version": "VideoDetector-v1.0",
                "processing_duration": 8.3
            },
            {
                "file_name": "audio2.mp3",
                "file_type": "audio",
                "file_size": 4096,
                "detection_score": 0.91,
                "classification": "Real",
                "model_version": "AudioDetector-v1.0",
                "processing_duration": 2.1
            }
        ]
        
        for record_data in sample_records:
            crud.create_detection_record(test_db, record_data)
        
        response = client.get("/api/v1/telemetry/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_scans"] == 4
        assert data["deepfakes_detected"] == 2
        assert data["real_media_detected"] == 2
        
        assert data["scans_by_type"]["image"] == 1
        assert data["scans_by_type"]["audio"] == 2
        assert data["scans_by_type"]["video"] == 1
        
        assert data["classification_breakdown"]["Fake"] == 2
        assert data["classification_breakdown"]["Real"] == 2
        
        expected_avg = (1.2 + 2.5 + 8.3 + 2.1) / 4
        assert abs(data["avg_processing_duration"] - expected_avg) < 0.01
    
    @patch('app.models.image_detector.ImageDetector')
    @patch('app.models.audio_detector.AudioDetector')
    @patch('app.models.video_detector.VideoDetector')
    def test_all_detection_types_in_sequence(
        self,
        mock_video_detector_class,
        mock_audio_detector_class,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        sample_audio_file,
        sample_video_file
    ):
        """
        Test sequential detection of all three media types.
        """
        from app.models.base import DetectionResult
        
        mock_image_result = DetectionResult(
            prediction="deepfake",
            confidence=0.95,
            probabilities={"authentic": 0.05, "deepfake": 0.95},
            processing_time=1.2,
            metadata={"model_type": "ImageDetector", "device": "cpu"}
        )
        
        mock_audio_result = DetectionResult(
            prediction="authentic",
            confidence=0.88,
            probabilities={"authentic": 0.88, "deepfake": 0.12},
            processing_time=2.3,
            metadata={"model_type": "AudioDetector", "device": "cpu", "num_segments_analyzed": 3}
        )
        
        mock_video_result = DetectionResult(
            prediction="deepfake",
            confidence=0.91,
            probabilities={"authentic": 0.09, "deepfake": 0.91},
            processing_time=8.5,
            metadata={"model_type": "VideoDetector", "device": "cpu", "num_frames": 15}
        )
        
        mock_image_detector = Mock()
        mock_image_detector.detect.return_value = mock_image_result
        
        mock_audio_detector = Mock()
        mock_audio_detector.detect.return_value = mock_audio_result
        
        mock_video_detector = Mock()
        mock_video_detector.detect.return_value = mock_video_result
        
        mock_factory = Mock()
        def get_detector_side_effect(media_type):
            if media_type == "image":
                return mock_image_detector
            elif media_type == "audio":
                return mock_audio_detector
            elif media_type == "video":
                return mock_video_detector
        
        mock_factory.get_detector.side_effect = get_detector_side_effect
        client.app.state.detector_factory = mock_factory
        
        image_resp = client.post("/api/v1/predict/image", files={"file": sample_image_file})
        assert image_resp.status_code == 200
        assert image_resp.json()["prediction"] == "deepfake"
        
        audio_resp = client.post("/api/v1/predict/audio", files={"file": sample_audio_file})
        assert audio_resp.status_code == 200
        assert audio_resp.json()["prediction"] == "authentic"
        
        video_resp = client.post("/api/v1/predict/video", files={"file": sample_video_file})
        assert video_resp.status_code == 200
        assert video_resp.json()["prediction"] == "deepfake"
        
        stats_resp = client.get("/api/v1/telemetry/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.json()
        
        assert stats_data["total_scans"] == 3
        assert stats_data["deepfakes_detected"] == 2
        assert stats_data["real_media_detected"] == 1
        
        assert stats_data["scans_by_type"]["image"] == 1
        assert stats_data["scans_by_type"]["audio"] == 1
        assert stats_data["scans_by_type"]["video"] == 1


class TestAudioVideoIntegration:
    """Test audio and video detection with database integration."""
    
    @patch('app.models.audio_detector.AudioDetector')
    def test_audio_detection_persists_to_database(
        self,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        sample_audio_file,
        mock_audio_detection_result
    ):
        """
        Test that audio detection results are properly logged to the database.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_audio_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        record_id = data["record_id"]
        assert record_id is not None
        
        record = crud.get_record_by_id(test_db, record_id)
        
        assert record is not None
        assert record.file_name == "test_audio.wav"
        assert record.file_type == "audio"
        assert record.classification == "Fake"
        assert record.detection_score == 0.89
        assert record.processing_duration >= 0
    
    @patch('app.models.video_detector.VideoDetector')
    def test_video_detection_persists_to_database(
        self,
        mock_video_detector_class,
        client: TestClient,
        test_db,
        sample_video_file,
        mock_video_detection_result
    ):
        """
        Test that video detection results are properly logged to the database.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_video_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/video",
            files={"file": sample_video_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        record_id = data["record_id"]
        assert record_id is not None
        
        record = crud.get_record_by_id(test_db, record_id)
        
        assert record is not None
        assert record.file_name == "test_video.mp4"
        assert record.file_type == "video"
        assert record.classification == "Fake"
        assert record.detection_score == 0.91
        assert record.processing_duration >= 0
    
    @patch('app.models.audio_detector.AudioDetector')
    @patch('app.models.video_detector.VideoDetector')
    def test_mixed_media_history(
        self,
        mock_video_detector_class,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        sample_audio_file,
        sample_video_file,
        mock_audio_detection_result,
        mock_video_detection_result
    ):
        """
        Test that history endpoint returns records from all media types.
        """
        mock_audio_detector = Mock()
        mock_audio_detector.detect.return_value = mock_audio_detection_result
        
        mock_video_detector = Mock()
        mock_video_detector.detect.return_value = mock_video_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.side_effect = lambda media_type: (
            mock_audio_detector if media_type == "audio" else mock_video_detector
        )
        client.app.state.detector_factory = mock_factory
        
        audio_response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file}
        )
        assert audio_response.status_code == 200
        
        video_response = client.post(
            "/api/v1/predict/video",
            files={"file": sample_video_file}
        )
        assert video_response.status_code == 200
        
        history_response = client.get("/api/v1/telemetry/history")
        assert history_response.status_code == 200
        history_data = history_response.json()
        
        assert history_data["total"] == 2
        assert len(history_data["records"]) == 2
        
        file_types = {record["file_type"] for record in history_data["records"]}
        assert "audio" in file_types
        assert "video" in file_types


class TestResponseSchemaValidation:
    """Test response schema validation for all media types."""
    
    @patch('app.models.image_detector.ImageDetector')
    def test_image_response_schema_compliance(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test that image detection response complies with ImagePredictionResponse schema.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "prediction", "is_fake", "confidence",
            "probabilities", "processing_time_seconds", "inference_time_ms",
            "metadata", "record_id"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert isinstance(data["probabilities"], dict)
        assert "authentic" in data["probabilities"]
        assert "deepfake" in data["probabilities"]
        
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["processing_time_seconds"], (int, float))
        assert isinstance(data["inference_time_ms"], (int, float))
    
    @patch('app.models.audio_detector.AudioDetector')
    def test_audio_response_schema_compliance(
        self,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        sample_audio_file,
        mock_audio_detection_result
    ):
        """
        Test that audio detection response complies with AudioPredictionResponse schema.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_audio_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "prediction", "is_fake", "confidence",
            "probabilities", "processing_time_seconds", "inference_time_ms",
            "metadata", "record_id"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert isinstance(data["probabilities"], dict)
        assert "authentic" in data["probabilities"]
        assert "deepfake" in data["probabilities"]
        
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["processing_time_seconds"], (int, float))
        assert isinstance(data["inference_time_ms"], (int, float))
    
    @patch('app.models.video_detector.VideoDetector')
    def test_video_response_schema_compliance(
        self,
        mock_video_detector_class,
        client: TestClient,
        test_db,
        sample_video_file,
        mock_video_detection_result
    ):
        """
        Test that video detection response complies with VideoPredictionResponse schema.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_video_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/video",
            files={"file": sample_video_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "prediction", "is_fake", "confidence",
            "probabilities", "processing_time_seconds", "inference_time_ms",
            "metadata", "record_id"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert isinstance(data["probabilities"], dict)
        assert "authentic" in data["probabilities"]
        assert "deepfake" in data["probabilities"]
        
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["processing_time_seconds"], (int, float))
        assert isinstance(data["inference_time_ms"], (int, float))
    
    @patch('app.models.image_detector.ImageDetector')
    def test_image_probabilities_sum_to_one(
        self,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test that image detection probabilities sum to approximately 1.0.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        probabilities = data["probabilities"]
        prob_sum = probabilities["authentic"] + probabilities["deepfake"]
        
        assert abs(prob_sum - 1.0) < 0.01
    
    @patch('app.models.audio_detector.AudioDetector')
    def test_audio_probabilities_sum_to_one(
        self,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        sample_audio_file,
        mock_audio_detection_result
    ):
        """
        Test that audio detection probabilities sum to approximately 1.0.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_audio_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        probabilities = data["probabilities"]
        prob_sum = probabilities["authentic"] + probabilities["deepfake"]
        
        assert abs(prob_sum - 1.0) < 0.01
    
    @patch('app.models.video_detector.VideoDetector')
    def test_video_probabilities_sum_to_one(
        self,
        mock_video_detector_class,
        client: TestClient,
        test_db,
        sample_video_file,
        mock_video_detection_result
    ):
        """
        Test that video detection probabilities sum to approximately 1.0.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_video_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/video",
            files={"file": sample_video_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        probabilities = data["probabilities"]
        prob_sum = probabilities["authentic"] + probabilities["deepfake"]
        
        assert abs(prob_sum - 1.0) < 0.01
    
    @patch('app.models.image_detector.ImageDetector')
    @patch('app.models.audio_detector.AudioDetector')
    @patch('app.models.video_detector.VideoDetector')
    def test_all_media_types_consistent_schema(
        self,
        mock_video_detector_class,
        mock_audio_detector_class,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        sample_audio_file,
        sample_video_file,
        mock_detection_result,
        mock_audio_detection_result,
        mock_video_detection_result
    ):
        """
        Test that all media types return consistent base schema.
        """
        mock_image_detector = Mock()
        mock_image_detector.detect.return_value = mock_detection_result
        
        mock_audio_detector = Mock()
        mock_audio_detector.detect.return_value = mock_audio_detection_result
        
        mock_video_detector = Mock()
        mock_video_detector.detect.return_value = mock_video_detection_result
        
        mock_factory = Mock()
        def get_detector_side_effect(media_type):
            if media_type == "image":
                return mock_image_detector
            elif media_type == "audio":
                return mock_audio_detector
            elif media_type == "video":
                return mock_video_detector
        
        mock_factory.get_detector.side_effect = get_detector_side_effect
        client.app.state.detector_factory = mock_factory
        
        image_resp = client.post("/api/v1/predict/image", files={"file": sample_image_file})
        audio_resp = client.post("/api/v1/predict/audio", files={"file": sample_audio_file})
        video_resp = client.post("/api/v1/predict/video", files={"file": sample_video_file})
        
        assert image_resp.status_code == 200
        assert audio_resp.status_code == 200
        assert video_resp.status_code == 200
        
        base_fields = [
            "prediction",
            "is_fake",
            "confidence",
            "probabilities",
            "processing_time_seconds",
            "inference_time_ms",
        ]
        
        for response in [image_resp, audio_resp, video_resp]:
            data = response.json()
            for field in base_fields:
                assert field in data


# ============================================================================
# Fixture Tests (Verify Test Setup)
# ============================================================================

class TestMultimodalStatistics:
    """Test statistics aggregation across all media types."""
    
    def test_stats_with_mixed_media_types(self, client: TestClient, test_db):
        """
        Test statistics with image, audio, and video records.
        """
        sample_records = [
            {
                "file_name": "image1.jpg",
                "file_type": "image",
                "file_size": 2048,
                "detection_score": 0.92,
                "classification": "Fake",
                "model_version": "ImageDetector-v1.0",
                "processing_duration": 1.2
            },
            {
                "file_name": "audio1.wav",
                "file_type": "audio",
                "file_size": 5120,
                "detection_score": 0.87,
                "classification": "Fake",
                "model_version": "AudioDetector-v1.0",
                "processing_duration": 2.5
            },
            {
                "file_name": "video1.mp4",
                "file_type": "video",
                "file_size": 10240,
                "detection_score": 0.78,
                "classification": "Real",
                "model_version": "VideoDetector-v1.0",
                "processing_duration": 8.3
            },
            {
                "file_name": "audio2.mp3",
                "file_type": "audio",
                "file_size": 4096,
                "detection_score": 0.91,
                "classification": "Real",
                "model_version": "AudioDetector-v1.0",
                "processing_duration": 2.1
            }
        ]
        
        for record_data in sample_records:
            crud.create_detection_record(test_db, record_data)
        
        response = client.get("/api/v1/telemetry/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_scans"] == 4
        assert data["deepfakes_detected"] == 2
        assert data["real_media_detected"] == 2
        
        assert data["scans_by_type"]["image"] == 1
        assert data["scans_by_type"]["audio"] == 2
        assert data["scans_by_type"]["video"] == 1
        
        assert data["classification_breakdown"]["Fake"] == 2
        assert data["classification_breakdown"]["Real"] == 2
        
        expected_avg = (1.2 + 2.5 + 8.3 + 2.1) / 4
        assert abs(data["avg_processing_duration"] - expected_avg) < 0.01
    
    @patch('app.models.image_detector.ImageDetector')
    @patch('app.models.audio_detector.AudioDetector')
    @patch('app.models.video_detector.VideoDetector')
    def test_all_detection_types_in_sequence(
        self,
        mock_video_detector_class,
        mock_audio_detector_class,
        mock_image_detector_class,
        client: TestClient,
        test_db,
        sample_image_file,
        sample_audio_file,
        sample_video_file
    ):
        """
        Test sequential detection of all three media types.
        """
        from app.models.base import DetectionResult
        
        mock_image_result = DetectionResult(
            prediction="deepfake",
            confidence=0.95,
            probabilities={"authentic": 0.05, "deepfake": 0.95},
            processing_time=1.2,
            metadata={"model_type": "ImageDetector", "device": "cpu"}
        )
        
        mock_audio_result = DetectionResult(
            prediction="authentic",
            confidence=0.88,
            probabilities={"authentic": 0.88, "deepfake": 0.12},
            processing_time=2.3,
            metadata={"model_type": "AudioDetector", "device": "cpu", "num_segments_analyzed": 3}
        )
        
        mock_video_result = DetectionResult(
            prediction="deepfake",
            confidence=0.91,
            probabilities={"authentic": 0.09, "deepfake": 0.91},
            processing_time=8.5,
            metadata={"model_type": "VideoDetector", "device": "cpu", "num_frames": 15}
        )
        
        mock_image_detector = Mock()
        mock_image_detector.detect.return_value = mock_image_result
        
        mock_audio_detector = Mock()
        mock_audio_detector.detect.return_value = mock_audio_result
        
        mock_video_detector = Mock()
        mock_video_detector.detect.return_value = mock_video_result
        
        mock_factory = Mock()
        def get_detector_side_effect(media_type):
            if media_type == "image":
                return mock_image_detector
            elif media_type == "audio":
                return mock_audio_detector
            elif media_type == "video":
                return mock_video_detector
        
        mock_factory.get_detector.side_effect = get_detector_side_effect
        client.app.state.detector_factory = mock_factory
        
        image_resp = client.post("/api/v1/predict/image", files={"file": sample_image_file})
        assert image_resp.status_code == 200
        assert image_resp.json()["prediction"] == "deepfake"
        
        audio_resp = client.post("/api/v1/predict/audio", files={"file": sample_audio_file})
        assert audio_resp.status_code == 200
        assert audio_resp.json()["prediction"] == "authentic"
        
        video_resp = client.post("/api/v1/predict/video", files={"file": sample_video_file})
        assert video_resp.status_code == 200
        assert video_resp.json()["prediction"] == "deepfake"
        
        stats_resp = client.get("/api/v1/telemetry/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.json()
        
        assert stats_data["total_scans"] == 3
        assert stats_data["deepfakes_detected"] == 2
        assert stats_data["real_media_detected"] == 1
        
        assert stats_data["scans_by_type"]["image"] == 1
        assert stats_data["scans_by_type"]["audio"] == 1
        assert stats_data["scans_by_type"]["video"] == 1


class TestAudioDetection:
    """Test suite for audio deepfake detection endpoint."""
    
    @patch('app.models.audio_detector.AudioDetector')
    def test_predict_audio_success(
        self,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        sample_audio_file,
        mock_audio_detection_result
    ):
        """
        Test POST /api/v1/predict/audio - Successful audio detection.
        
        This test mocks the AudioDetector to avoid loading heavy PyTorch models.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_audio_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "processing_time_seconds" in data
        assert "inference_time_ms" in data
        assert "metadata" in data
        
        assert data["prediction"] == "deepfake"
        assert data["confidence"] == 0.89
        assert data["is_fake"] is True
        
        assert "authentic" in data["probabilities"]
        assert "deepfake" in data["probabilities"]
        assert data["probabilities"]["deepfake"] == 0.89
        
        assert "record_id" in data
        assert data["record_id"] is not None
        
        mock_factory.get_detector.assert_called_once_with("audio")
        mock_detector_instance.detect.assert_called_once()
    
    def test_predict_audio_invalid_extension(self, client: TestClient):
        """
        Test POST /api/v1/predict/audio with invalid file extension.
        """
        invalid_file = BytesIO(b"fake content")
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": ("test.txt", invalid_file, "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_predict_audio_corrupted_file(self, client: TestClient, sample_invalid_audio_file):
        """
        Test POST /api/v1/predict/audio with corrupted audio file.
        """
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_invalid_audio_file}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    @patch('app.models.audio_detector.AudioDetector')
    def test_predict_audio_authentic(
        self,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        sample_audio_file,
        mock_audio_authentic_result
    ):
        """
        Test POST /api/v1/predict/audio - Detection of authentic audio.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_audio_authentic_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["prediction"] == "authentic"
        assert data["is_fake"] is False
        assert data["confidence"] == 0.86
        
        record_id = data["record_id"]
        record = crud.get_record_by_id(test_db, record_id)
        assert record is not None
        assert record.classification == "Real"
        assert record.file_type == "audio"
    
    def test_predict_audio_no_file(self, client: TestClient):
        """
        Test POST /api/v1/predict/audio without file parameter.
        """
        response = client.post("/api/v1/predict/audio")
        
        assert response.status_code == 422
    
    @patch('app.models.audio_detector.AudioDetector')
    def test_predict_audio_metadata_structure(
        self,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        sample_audio_file,
        mock_audio_detection_result
    ):
        """
        Test that audio detection returns correct metadata structure.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_audio_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        metadata = data["metadata"]
        assert metadata.get("num_segments_analyzed") == 3
        assert "model_type" in metadata
        assert metadata["model_type"] == "DualFeature_CNN_GRU"
        assert "architecture" in metadata
    
    @pytest.mark.parametrize("audio_format,content_type", [
        (".mp3", "audio/mp3"),
        (".wav", "audio/wav"),
        (".flac", "audio/flac"),
        (".ogg", "audio/ogg"),
        (".m4a", "audio/m4a"),
    ])
    @patch('app.models.audio_detector.AudioDetector')
    def test_predict_audio_various_formats(
        self,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        mock_audio_detection_result,
        audio_format: str,
        content_type: str
    ):
        """
        Test audio detection with various audio formats.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_audio_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        audio_bytes = BytesIO(b"fake audio content")
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": (f"test{audio_format}", audio_bytes, content_type)}
        )
        
        assert response.status_code in [200, 400]


class TestAudioDetection:
    """Test suite for audio deepfake detection endpoint."""
    
    @patch('app.models.audio_detector.AudioDetector')
    def test_predict_audio_success(
        self,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        sample_audio_file,
        mock_audio_detection_result
    ):
        """
        Test POST /api/v1/predict/audio - Successful audio detection.
        
        This test mocks the AudioDetector to avoid loading heavy PyTorch models.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_audio_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "processing_time_seconds" in data
        assert "inference_time_ms" in data
        assert "metadata" in data
        
        assert data["prediction"] == "deepfake"
        assert data["confidence"] == 0.89
        assert data["is_fake"] is True
        
        assert "authentic" in data["probabilities"]
        assert "deepfake" in data["probabilities"]
        assert data["probabilities"]["deepfake"] == 0.89
        
        assert "record_id" in data
        assert data["record_id"] is not None
        
        mock_factory.get_detector.assert_called_once_with("audio")
        mock_detector_instance.detect.assert_called_once()
    
    def test_predict_audio_invalid_extension(self, client: TestClient):
        """
        Test POST /api/v1/predict/audio with invalid file extension.
        """
        invalid_file = BytesIO(b"fake content")
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": ("test.txt", invalid_file, "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_predict_audio_corrupted_file(self, client: TestClient, sample_invalid_audio_file):
        """
        Test POST /api/v1/predict/audio with corrupted audio file.
        """
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_invalid_audio_file}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    @patch('app.models.audio_detector.AudioDetector')
    def test_predict_audio_authentic(
        self,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        sample_audio_file,
        mock_audio_authentic_result
    ):
        """
        Test POST /api/v1/predict/audio - Detection of authentic audio.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_audio_authentic_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["prediction"] == "authentic"
        assert data["is_fake"] is False
        assert data["confidence"] == 0.86
        
        record_id = data["record_id"]
        record = crud.get_record_by_id(test_db, record_id)
        assert record is not None
        assert record.classification == "Real"
        assert record.file_type == "audio"
    
    def test_predict_audio_no_file(self, client: TestClient):
        """
        Test POST /api/v1/predict/audio without file parameter.
        """
        response = client.post("/api/v1/predict/audio")
        
        assert response.status_code == 422
    
    @patch('app.models.audio_detector.AudioDetector')
    def test_predict_audio_metadata_structure(
        self,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        sample_audio_file,
        mock_audio_detection_result
    ):
        """
        Test that audio detection returns correct metadata structure.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_audio_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        metadata = data["metadata"]
        assert metadata.get("num_segments_analyzed") == 3
        assert "model_type" in metadata
        assert metadata["model_type"] == "DualFeature_CNN_GRU"
        assert "architecture" in metadata
    
    @pytest.mark.parametrize("audio_format,content_type", [
        (".mp3", "audio/mp3"),
        (".wav", "audio/wav"),
        (".flac", "audio/flac"),
        (".ogg", "audio/ogg"),
        (".m4a", "audio/m4a"),
    ])
    @patch('app.models.audio_detector.AudioDetector')
    def test_predict_audio_various_formats(
        self,
        mock_audio_detector_class,
        client: TestClient,
        test_db,
        mock_audio_detection_result,
        audio_format: str,
        content_type: str
    ):
        """
        Test audio detection with various audio formats.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_audio_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        audio_bytes = BytesIO(b"fake audio content")
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": (f"test{audio_format}", audio_bytes, content_type)}
        )
        
        assert response.status_code in [200, 400]


class TestVideoDetection:
    """Test suite for video deepfake detection endpoint."""
    
    @patch('app.models.video_detector.VideoDetector')
    def test_predict_video_success(
        self,
        mock_video_detector_class,
        client: TestClient,
        test_db,
        sample_video_file,
        mock_video_detection_result
    ):
        """
        Test POST /api/v1/predict/video - Successful video detection.
        
        This test mocks the VideoDetector to avoid loading heavy PyTorch models.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_video_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/video",
            files={"file": sample_video_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "processing_time_seconds" in data
        assert "inference_time_ms" in data
        assert "metadata" in data
        
        assert data["prediction"] == "deepfake"
        assert data["confidence"] == 0.91
        assert data["is_fake"] is True
        
        assert "authentic" in data["probabilities"]
        assert "deepfake" in data["probabilities"]
        assert data["probabilities"]["deepfake"] == 0.91
        
        assert "record_id" in data
        assert data["record_id"] is not None
        
        mock_factory.get_detector.assert_called_once_with("video")
        mock_detector_instance.detect.assert_called_once()
    
    def test_predict_video_invalid_extension(self, client: TestClient):
        """
        Test POST /api/v1/predict/video with invalid file extension.
        """
        invalid_file = BytesIO(b"fake content")
        
        response = client.post(
            "/api/v1/predict/video",
            files={"file": ("test.txt", invalid_file, "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_predict_video_corrupted_file(self, client: TestClient, sample_invalid_video_file):
        """
        Test POST /api/v1/predict/video with corrupted video file.
        """
        response = client.post(
            "/api/v1/predict/video",
            files={"file": sample_invalid_video_file}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    @patch('app.models.video_detector.VideoDetector')
    def test_predict_video_authentic(
        self,
        mock_video_detector_class,
        client: TestClient,
        test_db,
        sample_video_file,
        mock_video_authentic_result
    ):
        """
        Test POST /api/v1/predict/video - Detection of authentic video.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_video_authentic_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/video",
            files={"file": sample_video_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["prediction"] == "authentic"
        assert data["is_fake"] is False
        assert data["confidence"] == 0.93
        
        record_id = data["record_id"]
        record = crud.get_record_by_id(test_db, record_id)
        assert record is not None
        assert record.classification == "Real"
        assert record.file_type == "video"
    
    def test_predict_video_no_file(self, client: TestClient):
        """
        Test POST /api/v1/predict/video without file parameter.
        """
        response = client.post("/api/v1/predict/video")
        
        assert response.status_code == 422
    
    @patch('app.models.video_detector.VideoDetector')
    def test_predict_video_metadata_structure(
        self,
        mock_video_detector_class,
        client: TestClient,
        test_db,
        sample_video_file,
        mock_video_detection_result
    ):
        """
        Test that video detection returns correct metadata structure.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_video_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/video",
            files={"file": sample_video_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        metadata = data["metadata"]
        assert metadata.get("num_frames") == 15
        assert "model_type" in metadata
        assert metadata["model_type"] == "TriStreamMultimodalNet"
        assert "architecture" in metadata
    
    @pytest.mark.parametrize("video_format,content_type", [
        (".mp4", "video/mp4"),
        (".avi", "video/avi"),
        (".mov", "video/mov"),
        (".mkv", "video/mkv"),
        (".webm", "video/webm"),
    ])
    @patch('app.models.video_detector.VideoDetector')
    def test_predict_video_various_formats(
        self,
        mock_video_detector_class,
        client: TestClient,
        test_db,
        mock_video_detection_result,
        video_format: str,
        content_type: str
    ):
        """
        Test video detection with various video formats.
        """
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_video_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        video_bytes = BytesIO(b"fake video content")
        
        response = client.post(
            "/api/v1/predict/video",
            files={"file": (f"test{video_format}", video_bytes, content_type)}
        )
        
        assert response.status_code in [200, 400]
    
    @patch('app.models.video_detector.VideoDetector')
    def test_predict_video_response_fields(
        self,
        mock_video_detector_class,
        client: TestClient,
        test_db,
        sample_video_file
    ):
        """
        Test that video response includes all required fields.
        """
        from app.models.base import DetectionResult
        
        mock_result = DetectionResult(
            prediction="deepfake",
            confidence=0.87,
            probabilities={"authentic": 0.13, "deepfake": 0.87},
            processing_time=10.5,
            metadata={
                "model_type": "TriStreamMultimodalNet",
                "num_frames": 20,
                "device": "cpu",
                "frame_analysis": {
                    "min_deepfake_prob": 0.75,
                    "max_deepfake_prob": 0.95,
                    "mean_deepfake_prob": 0.87
                },
                "temporal_consistency": {
                    "variance": 0.02,
                    "is_consistent": True
                }
            }
        )
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/video",
            files={"file": sample_video_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["prediction"] == "deepfake"
        assert data["confidence"] == 0.87
        assert data["is_fake"] is True
        assert data["processing_time_seconds"] >= 0
        assert data["inference_time_ms"] == pytest.approx(10500.0)
        
        meta = data["metadata"]
        assert "frame_analysis" in meta
        assert "temporal_consistency" in meta


class TestFixtures:
    """Test that fixtures are working correctly."""
    
    def test_test_database_is_isolated(self, test_db):
        """
        Test that test database is properly isolated.
        """
        # Create a record in test database
        record = crud.create_detection_record(test_db, {
            "file_name": "test.jpg",
            "file_type": "image",
            "file_size": 1024,
            "detection_score": 0.9,
            "classification": "Real",
            "model_version": "TestModel-v1.0",
            "processing_duration": 1.0
        })
        
        # Verify record was created
        assert record.id is not None
        
        # Query it back
        fetched = crud.get_record_by_id(test_db, record.id)
        assert fetched is not None
        assert fetched.id == record.id
    
    def test_sample_image_file_is_valid(self, sample_image_file):
        """
        Test that sample_image_file fixture creates valid image.
        """
        filename, file_obj, content_type = sample_image_file
        
        # Verify structure
        assert filename == "test_image.png"
        assert content_type == "image/png"
        
        # Verify it's a valid image
        file_obj.seek(0)
        img = Image.open(file_obj)
        assert img.size == (100, 100)
        assert img.mode == "RGB"
    
    def test_sample_audio_file_is_valid(self, sample_audio_file):
        """
        Test that sample_audio_file fixture creates valid audio.
        """
        filename, file_obj, content_type = sample_audio_file
        
        assert filename == "test_audio.wav"
        assert content_type == "audio/wav"
        
        file_obj.seek(0)
        assert len(file_obj.read()) > 0
    
    def test_sample_video_file_is_valid(self, sample_video_file):
        """
        Test that sample_video_file fixture creates valid video.
        """
        filename, file_obj, content_type = sample_video_file
        
        assert filename == "test_video.mp4"
        assert content_type == "video/mp4"
        
        file_obj.seek(0)
        assert len(file_obj.read()) > 0
    
    def test_mock_detection_result_structure(self, mock_detection_result):
        """
        Test that mock_detection_result has correct structure.
        """
        assert mock_detection_result.prediction == "deepfake"
        assert mock_detection_result.confidence == 0.92
        assert "authentic" in mock_detection_result.probabilities
        assert "deepfake" in mock_detection_result.probabilities
        assert mock_detection_result.processing_time > 0
    
    def test_mock_audio_detection_result_structure(self, mock_audio_detection_result):
        """
        Test that mock_audio_detection_result has correct structure.
        """
        assert mock_audio_detection_result.prediction == "deepfake"
        assert mock_audio_detection_result.confidence == 0.89
        assert "num_segments_analyzed" in mock_audio_detection_result.metadata
        assert mock_audio_detection_result.metadata["num_segments_analyzed"] == 3
    
    def test_mock_video_detection_result_structure(self, mock_video_detection_result):
        """
        Test that mock_video_detection_result has correct structure.
        """
        assert mock_video_detection_result.prediction == "deepfake"
        assert mock_video_detection_result.confidence == 0.91
        assert "num_frames" in mock_video_detection_result.metadata
        assert mock_video_detection_result.metadata["num_frames"] == 15


# ============================================================================
# Session History Tests - Anonymous User Tracking
# ============================================================================

class TestSessionHistory:
    """Test suite for anonymous session history with persistent media."""
    
    def test_image_detection_with_session_id(
        self,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test image detection endpoint accepts session_id and saves media.
        """
        from unittest.mock import Mock
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        session_id = "test-session-123"
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file},
            data={"session_id": session_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "record_id" in data
        record_id = data["record_id"]
        
        # Verify record was saved with session_id
        from app.db import crud
        record = crud.get_record_by_id(test_db, record_id)
        assert record is not None
        assert record.session_id == session_id
        assert record.media_path is not None
        assert record.media_path.startswith("/media/")
    
    def test_video_detection_with_session_id(
        self,
        client: TestClient,
        test_db,
        sample_video_file,
        mock_video_detection_result
    ):
        """
        Test video detection endpoint accepts session_id and saves media.
        """
        from unittest.mock import Mock
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_video_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        session_id = "test-session-456"
        
        response = client.post(
            "/api/v1/predict/video",
            files={"file": sample_video_file},
            data={"session_id": session_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "record_id" in data
        record_id = data["record_id"]
        
        # Verify record was saved with session_id
        from app.db import crud
        record = crud.get_record_by_id(test_db, record_id)
        assert record is not None
        assert record.session_id == session_id
        assert record.media_path is not None
        assert record.media_path.startswith("/media/")
    
    def test_audio_detection_with_session_id(
        self,
        client: TestClient,
        test_db,
        sample_audio_file,
        mock_audio_detection_result
    ):
        """
        Test audio detection endpoint accepts session_id and saves media.
        """
        from unittest.mock import Mock
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_audio_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        session_id = "test-session-789"
        
        response = client.post(
            "/api/v1/predict/audio",
            files={"file": sample_audio_file},
            data={"session_id": session_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "record_id" in data
        record_id = data["record_id"]
        
        # Verify record was saved with session_id
        from app.db import crud
        record = crud.get_record_by_id(test_db, record_id)
        assert record is not None
        assert record.session_id == session_id
        assert record.media_path is not None
        assert record.media_path.startswith("/media/")
    
    def test_detection_without_session_id(
        self,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test detection works without session_id (backward compatibility).
        """
        from unittest.mock import Mock
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "record_id" in data
        record_id = data["record_id"]
        
        # Verify record was saved without session_id
        from app.db import crud
        record = crud.get_record_by_id(test_db, record_id)
        assert record is not None
        assert record.session_id is None
        assert record.media_path is not None
    
    def test_history_filtered_by_session_id(
        self,
        client: TestClient,
        test_db
    ):
        """
        Test history endpoint filters by session_id correctly.
        """
        from app.db import crud
        
        # Create records for different sessions
        session_1_records = [
            {
                "file_name": "user1_image1.jpg",
                "file_type": "image",
                "file_size": 1024,
                "detection_score": 0.9,
                "classification": "Fake",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.0,
                "session_id": "session-user-1",
                "media_path": "/media/abc123_user1_image1.jpg"
            },
            {
                "file_name": "user1_image2.jpg",
                "file_type": "image",
                "file_size": 2048,
                "detection_score": 0.8,
                "classification": "Real",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.2,
                "session_id": "session-user-1",
                "media_path": "/media/def456_user1_image2.jpg"
            }
        ]
        
        session_2_records = [
            {
                "file_name": "user2_video.mp4",
                "file_type": "video",
                "file_size": 5120,
                "detection_score": 0.7,
                "classification": "Fake",
                "model_version": "TestModel-v1.0",
                "processing_duration": 2.0,
                "session_id": "session-user-2",
                "media_path": "/media/ghi789_user2_video.mp4"
            }
        ]
        
        # Insert records
        for record_data in session_1_records + session_2_records:
            crud.create_detection_record(test_db, record_data)
        
        # Test: Get history for session-user-1
        response = client.get(
            "/api/v1/telemetry/history",
            params={"session_id": "session-user-1"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total"] == 2
        assert len(data["records"]) == 2
        
        # Verify all records belong to session-user-1
        for record in data["records"]:
            assert record["session_id"] == "session-user-1"
            assert record["media_path"] is not None
        
        # Test: Get history for session-user-2
        response = client.get(
            "/api/v1/telemetry/history",
            params={"session_id": "session-user-2"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total"] == 1
        assert len(data["records"]) == 1
        assert data["records"][0]["session_id"] == "session-user-2"
        assert data["records"][0]["file_name"] == "user2_video.mp4"
    
    def test_history_without_session_id_returns_all(
        self,
        client: TestClient,
        test_db
    ):
        """
        Test history endpoint without session_id returns all records.
        """
        from app.db import crud
        
        # Create records for different sessions
        records = [
            {
                "file_name": "test1.jpg",
                "file_type": "image",
                "file_size": 1024,
                "detection_score": 0.9,
                "classification": "Fake",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.0,
                "session_id": "session-1",
                "media_path": "/media/file1.jpg"
            },
            {
                "file_name": "test2.jpg",
                "file_type": "image",
                "file_size": 2048,
                "detection_score": 0.8,
                "classification": "Real",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.2,
                "session_id": "session-2",
                "media_path": "/media/file2.jpg"
            },
            {
                "file_name": "test3.jpg",
                "file_type": "image",
                "file_size": 3072,
                "detection_score": 0.7,
                "classification": "Fake",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.5,
                "session_id": None,
                "media_path": "/media/file3.jpg"
            }
        ]
        
        for record_data in records:
            crud.create_detection_record(test_db, record_data)
        
        # Query without session_id filter
        response = client.get("/api/v1/telemetry/history")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total"] == 3
        assert len(data["records"]) == 3
    
    def test_media_path_stored_correctly(
        self,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test that media_path is stored correctly in database.
        """
        from unittest.mock import Mock
        from app.db import crud
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        session_id = "test-session-media-path"
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file},
            data={"session_id": session_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        record_id = data["record_id"]
        
        # Verify media_path in database
        record = crud.get_record_by_id(test_db, record_id)
        assert record.media_path is not None
        assert record.media_path.startswith("/media/")
        assert record.media_path.endswith(".png")
        
        # Verify media_path contains UUID prefix
        filename = record.media_path.split("/")[-1]
        assert "_" in filename
        uuid_part = filename.split("_")[0]
        assert len(uuid_part) == 8
    
    def test_persistent_media_file_saved(
        self,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test that media file is actually saved to persistent_media directory.
        """
        from unittest.mock import Mock
        from app.db import crud
        import os
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        session_id = "test-session-file-save"
        
        response = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file},
            data={"session_id": session_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        record_id = data["record_id"]
        
        # Get media_path from database
        record = crud.get_record_by_id(test_db, record_id)
        media_path = record.media_path
        
        # Verify file exists in persistent_media directory
        filename = media_path.split("/")[-1]
        persistent_file_path = os.path.join(os.getcwd(), "persistent_media", filename)
        
        assert os.path.exists(persistent_file_path), f"File should exist at {persistent_file_path}"
        
        # Verify file is not empty
        file_size = os.path.getsize(persistent_file_path)
        assert file_size > 0
        
        # Cleanup test file
        try:
            os.remove(persistent_file_path)
        except Exception:
            pass
    
    def test_multiple_users_separate_history(
        self,
        client: TestClient,
        test_db
    ):
        """
        Test that multiple users have separate history based on session_id.
        """
        from app.db import crud
        
        # Create records for three different users
        user1_records = [
            {
                "file_name": "user1_file1.jpg",
                "file_type": "image",
                "file_size": 1024,
                "detection_score": 0.9,
                "classification": "Fake",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.0,
                "session_id": "user-session-1",
                "media_path": "/media/a1_user1_file1.jpg"
            },
            {
                "file_name": "user1_file2.jpg",
                "file_type": "image",
                "file_size": 2048,
                "detection_score": 0.8,
                "classification": "Real",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.2,
                "session_id": "user-session-1",
                "media_path": "/media/a2_user1_file2.jpg"
            }
        ]
        
        user2_records = [
            {
                "file_name": "user2_file1.mp4",
                "file_type": "video",
                "file_size": 5120,
                "detection_score": 0.7,
                "classification": "Fake",
                "model_version": "TestModel-v1.0",
                "processing_duration": 2.0,
                "session_id": "user-session-2",
                "media_path": "/media/b1_user2_file1.mp4"
            }
        ]
        
        user3_records = [
            {
                "file_name": "user3_file1.wav",
                "file_type": "audio",
                "file_size": 3072,
                "detection_score": 0.85,
                "classification": "Real",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.5,
                "session_id": "user-session-3",
                "media_path": "/media/c1_user3_file1.wav"
            },
            {
                "file_name": "user3_file2.wav",
                "file_type": "audio",
                "file_size": 4096,
                "detection_score": 0.95,
                "classification": "Fake",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.8,
                "session_id": "user-session-3",
                "media_path": "/media/c2_user3_file2.wav"
            },
            {
                "file_name": "user3_file3.jpg",
                "file_type": "image",
                "file_size": 2560,
                "detection_score": 0.75,
                "classification": "Real",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.3,
                "session_id": "user-session-3",
                "media_path": "/media/c3_user3_file3.jpg"
            }
        ]
        
        # Insert all records
        for record_data in user1_records + user2_records + user3_records:
            crud.create_detection_record(test_db, record_data)
        
        # Test: User 1 should see only their 2 records
        response = client.get(
            "/api/v1/telemetry/history",
            params={"session_id": "user-session-1"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        for record in data["records"]:
            assert record["session_id"] == "user-session-1"
        
        # Test: User 2 should see only their 1 record
        response = client.get(
            "/api/v1/telemetry/history",
            params={"session_id": "user-session-2"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["records"][0]["session_id"] == "user-session-2"
        
        # Test: User 3 should see their 3 records
        response = client.get(
            "/api/v1/telemetry/history",
            params={"session_id": "user-session-3"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        for record in data["records"]:
            assert record["session_id"] == "user-session-3"
        
        # Test: Without session_id, should see all 6 records
        response = client.get("/api/v1/telemetry/history")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 6
    
    def test_history_limit_with_session_id(
        self,
        client: TestClient,
        test_db
    ):
        """
        Test that limit parameter works correctly with session_id filter.
        """
        from app.db import crud
        
        # Create 5 records for the same session
        session_id = "test-session-limit"
        for i in range(5):
            record_data = {
                "file_name": f"test_file_{i}.jpg",
                "file_type": "image",
                "file_size": 1024 * (i + 1),
                "detection_score": 0.8 + (i * 0.02),
                "classification": "Fake" if i % 2 == 0 else "Real",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.0 + (i * 0.1),
                "session_id": session_id,
                "media_path": f"/media/file_{i}.jpg"
            }
            crud.create_detection_record(test_db, record_data)
        
        # Test: Request only 3 records
        response = client.get(
            "/api/v1/telemetry/history",
            params={"session_id": session_id, "limit": 3}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total"] == 3
        assert len(data["records"]) == 3
        
        # Verify all records belong to the correct session
        for record in data["records"]:
            assert record["session_id"] == session_id
    
    def test_media_path_in_history_response(
        self,
        client: TestClient,
        test_db
    ):
        """
        Test that media_path is included in history response.
        """
        from app.db import crud
        
        # Create a record with media_path
        record_data = {
            "file_name": "test_with_media.jpg",
            "file_type": "image",
            "file_size": 1024,
            "detection_score": 0.9,
            "classification": "Fake",
            "model_version": "TestModel-v1.0",
            "processing_duration": 1.0,
            "session_id": "test-session",
            "media_path": "/media/xyz123_test_with_media.jpg"
        }
        
        crud.create_detection_record(test_db, record_data)
        
        # Get history
        response = client.get(
            "/api/v1/telemetry/history",
            params={"session_id": "test-session"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total"] == 1
        assert "media_path" in data["records"][0]
        assert data["records"][0]["media_path"] == "/media/xyz123_test_with_media.jpg"
    
    def test_unique_filenames_prevent_collisions(
        self,
        client: TestClient,
        test_db,
        sample_image_file,
        mock_detection_result
    ):
        """
        Test that uploading same filename twice creates unique media paths.
        """
        from unittest.mock import Mock
        from app.db import crud
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        
        mock_factory = Mock()
        mock_factory.get_detector.return_value = mock_detector_instance
        client.app.state.detector_factory = mock_factory
        
        session_id = "test-session-collision"
        
        # Upload same file twice
        response1 = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file},
            data={"session_id": session_id}
        )
        
        # Reset file pointer
        sample_image_file[1].seek(0)
        
        response2 = client.post(
            "/api/v1/predict/image",
            files={"file": sample_image_file},
            data={"session_id": session_id}
        )
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        record1 = crud.get_record_by_id(test_db, response1.json()["record_id"])
        record2 = crud.get_record_by_id(test_db, response2.json()["record_id"])
        
        # Verify different media paths (due to UUID prefix)
        assert record1.media_path != record2.media_path
        assert record1.media_path.startswith("/media/")
        assert record2.media_path.startswith("/media/")
        
        # Cleanup test files
        import os
        for record in [record1, record2]:
            if record.media_path:
                filename = record.media_path.split("/")[-1]
                file_path = os.path.join(os.getcwd(), "persistent_media", filename)
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception:
                    pass
    
    def test_crud_get_recent_history_with_session_filter(self, test_db):
        """
        Test CRUD function get_recent_history with session_id parameter.
        """
        from app.db import crud
        
        # Create records for different sessions
        records_data = [
            {
                "file_name": "session1_file1.jpg",
                "file_type": "image",
                "file_size": 1024,
                "detection_score": 0.9,
                "classification": "Fake",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.0,
                "session_id": "session-alpha",
                "media_path": "/media/file1.jpg"
            },
            {
                "file_name": "session1_file2.jpg",
                "file_type": "image",
                "file_size": 2048,
                "detection_score": 0.8,
                "classification": "Real",
                "model_version": "TestModel-v1.0",
                "processing_duration": 1.2,
                "session_id": "session-alpha",
                "media_path": "/media/file2.jpg"
            },
            {
                "file_name": "session2_file1.mp4",
                "file_type": "video",
                "file_size": 5120,
                "detection_score": 0.7,
                "classification": "Fake",
                "model_version": "TestModel-v1.0",
                "processing_duration": 2.0,
                "session_id": "session-beta",
                "media_path": "/media/file3.mp4"
            }
        ]
        
        for record_data in records_data:
            crud.create_detection_record(test_db, record_data)
        
        # Test: Get history for session-alpha
        records = crud.get_recent_history(test_db, session_id="session-alpha", limit=20)
        assert len(records) == 2
        for record in records:
            assert record.session_id == "session-alpha"
        
        # Test: Get history for session-beta
        records = crud.get_recent_history(test_db, session_id="session-beta", limit=20)
        assert len(records) == 1
        assert records[0].session_id == "session-beta"
        
        # Test: Get all history without filter
        records = crud.get_recent_history(test_db, session_id=None, limit=20)
        assert len(records) == 3
    
    def test_detection_record_to_dict_includes_new_fields(self, test_db):
        """
        Test that DetectionRecord.to_dict() includes session_id and media_path.
        """
        from app.db import crud
        
        record_data = {
            "file_name": "test_dict.jpg",
            "file_type": "image",
            "file_size": 1024,
            "detection_score": 0.9,
            "classification": "Fake",
            "model_version": "TestModel-v1.0",
            "processing_duration": 1.0,
            "session_id": "test-session-dict",
            "media_path": "/media/test_dict.jpg"
        }
        
        record = crud.create_detection_record(test_db, record_data)
        record_dict = record.to_dict()
        
        # Verify new fields are in dictionary
        assert "session_id" in record_dict
        assert "media_path" in record_dict
        assert record_dict["session_id"] == "test-session-dict"
        assert record_dict["media_path"] == "/media/test_dict.jpg"
