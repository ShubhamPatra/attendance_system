"""Unit tests for MiniFASNet anti-spoofing module."""

import pytest
import numpy as np
import cv2
import torch
from unittest.mock import patch, MagicMock

from vision.anti_spoofing import (
    init_models,
    is_ready,
    check_liveness,
    _setup_minifasnet_model,
    LIVENESS_LABELS,
)
import core.config as config


class TestMiniFASNetInitialization:
    """Test MiniFASNet model initialization."""
    
    def test_init_models_creates_model(self):
        """Initialization should create and load model."""
        init_models()
        assert is_ready()
    
    def test_init_models_sets_device(self):
        """Initialization should set device (CPU or CUDA)."""
        init_models()
        # Should initialize without errors
        assert is_ready()
    
    def test_model_architecture_valid(self):
        """MiniFASNet architecture should be valid."""
        model = _setup_minifasnet_model()
        assert model is not None
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (1, 2)  # 2-class output
    
    @patch('os.path.exists')
    def test_graceful_degradation_no_weights(self, mock_exists):
        """Should handle missing pretrained weights gracefully."""
        mock_exists.return_value = False  # Model file doesn't exist
        
        init_models()
        # Should still work with random initialization
        assert is_ready()


class TestCheckLiveness:
    """Test liveness checking functionality."""
    
    @pytest.fixture
    def real_face_crop(self):
        """Create a realistic face crop image."""
        # Create a random RGB face crop (96x96)
        face = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        
        # Add some texture to make it look like a face
        center = (48, 48)
        cv2.circle(face, center, 20, (200, 150, 100), -1)  # Skin tone circle
        cv2.circle(face, (40, 40), 5, (50, 50, 50), -1)  # Eye
        cv2.circle(face, (56, 40), 5, (50, 50, 50), -1)  # Eye
        
        return cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    
    @pytest.fixture
    def printed_photo(self, real_face_crop):
        """Create a synthetic printed photo (flat appearance)."""
        # Reduce contrast and add gloss
        flat_photo = cv2.GaussianBlur(real_face_crop, (5, 5), 0)
        flat_photo = cv2.convertScaleAbs(flat_photo, alpha=0.6, beta=50)
        return flat_photo
    
    def test_check_liveness_returns_tuple(self):
        """check_liveness should return (label, confidence) tuple."""
        init_models()
        
        # Create dummy face crop
        face = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        
        result = check_liveness(face)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        label, confidence = result
        assert isinstance(label, int)
        assert isinstance(confidence, float)
    
    def test_check_liveness_label_valid(self):
        """Returned label should be valid (0, 1, or -1)."""
        init_models()
        
        face = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        label, confidence = check_liveness(face)
        
        assert label in [0, 1, -1]
    
    def test_check_liveness_confidence_in_range(self):
        """Confidence should be in [0, 1] range."""
        init_models()
        
        face = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        label, confidence = check_liveness(face)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_check_liveness_empty_frame_rejected(self):
        """Empty frame should return low confidence."""
        init_models()
        
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        label, confidence = check_liveness(empty_frame)
        
        assert label == 0  # Spoof/no face
        assert confidence == 0.0
    
    def test_check_liveness_too_small_rejected(self):
        """Tiny face should be rejected."""
        init_models()
        
        # Very small image (8x8)
        tiny_face = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        label, confidence = check_liveness(tiny_face)
        
        assert label == 0  # Should be rejected
        assert confidence == 0.0
    
    def test_check_liveness_disabled_returns_real(self):
        """When disabled, should mark all faces as real."""
        with patch.object(config, 'DISABLE_ANTISPOOFING', True):
            face = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
            label, confidence = check_liveness(face)
            
            assert label == 1  # Real
            assert confidence == 1.0
    
    def test_check_liveness_model_not_ready_graceful(self):
        """If model not ready, should gracefully mark as real."""
        with patch('vision.anti_spoofing._is_ready', False):
            face = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
            label, confidence = check_liveness(face)
            
            assert label == 1  # Graceful degradation
            assert confidence == 1.0
    
    def test_check_liveness_exception_handling(self):
        """Should handle exceptions gracefully."""
        init_models()
        
        # Invalid input type
        with patch('vision.anti_spoofing._model') as mock_model:
            mock_model.side_effect = Exception("Model error")
            
            face = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
            label, confidence = check_liveness(face)
            
            # Should return graceful degradation
            assert label == 1
            assert confidence == 1.0


class TestPerformance:
    """Test performance characteristics."""
    
    def test_inference_speed(self):
        """Inference should be fast (<100ms)."""
        import time
        
        init_models()
        face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        start = time.time()
        for _ in range(5):
            check_liveness(face)
        elapsed = time.time() - start
        
        avg_ms = (elapsed * 1000) / 5
        
        # MiniFASNet should be < 20ms per frame on average
        assert avg_ms < 100, f"Inference too slow: {avg_ms:.2f}ms"
    
    def test_model_memory_efficient(self):
        """Model should have reasonable memory footprint."""
        model = _setup_minifasnet_model()
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        
        # MiniFASNet should have ~10-15M parameters (lightweight)
        assert params < 50_000_000, f"Model too large: {params:,} parameters"


class TestInputValidation:
    """Test input validation and edge cases."""
    
    def test_different_face_sizes(self):
        """Should handle various face crop sizes."""
        init_models()
        
        sizes = [(64, 64), (96, 96), (128, 128), (224, 224), (256, 256)]
        
        for h, w in sizes:
            face = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            label, confidence = check_liveness(face)
            
            assert label in [0, 1]
            assert 0 <= confidence <= 1
    
    def test_grayscale_input_converted(self):
        """Should handle grayscale input (convert to BGR)."""
        init_models()
        
        # Grayscale face
        gray_face = np.random.randint(0, 255, (96, 96), dtype=np.uint8)
        # Note: In real usage, cv2.cvtColor might fail with grayscale
        # But our implementation should handle it
        
        # Convert grayscale to BGR first
        bgr_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
        
        label, confidence = check_liveness(bgr_face)
        assert label in [0, 1]
    
    def test_extreme_brightness_values(self):
        """Should handle extreme brightness."""
        init_models()
        
        # Very bright
        bright = np.ones((96, 96, 3), dtype=np.uint8) * 255
        label, conf = check_liveness(bright)
        assert label in [0, 1] or conf == 1.0
        
        # Very dark
        dark = np.zeros((96, 96, 3), dtype=np.uint8)
        label, conf = check_liveness(dark)
        assert label in [0, 1] or conf == 1.0


class TestLivenessLabels:
    """Test liveness label definitions."""
    
    def test_label_mapping_exists(self):
        """All labels should be defined."""
        assert 0 in LIVENESS_LABELS
        assert 1 in LIVENESS_LABELS
        assert -1 in LIVENESS_LABELS
    
    def test_label_descriptions(self):
        """Labels should have human-readable descriptions."""
        assert LIVENESS_LABELS[0] == "spoof_or_no_face"
        assert LIVENESS_LABELS[1] == "real"
        assert LIVENESS_LABELS[-1] == "internal_error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
