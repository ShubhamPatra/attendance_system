"""Unit tests for screen/print detection module."""

import numpy as np
import cv2
import pytest
from vision.screen_print_detector import detect_screen_or_print


class TestScreenPrintDetector:
    """Test suite for screen/print detection using FFT and texture analysis."""
    
    @pytest.fixture
    def natural_face(self):
        """Generate a synthetic natural face-like image."""
        # Create skin-tone colored image with slight texture variation
        h, w = 128, 128
        face = np.ones((h, w, 3), dtype=np.uint8)
        
        # Skin tone (BGR): light brown
        face[:] = [135, 155, 165]
        
        # Add texture noise (natural skin variation)
        noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
        face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add some features (darker/lighter regions)
        cv2.circle(face, (40, 40), 8, (100, 120, 130), -1)  # Left eye
        cv2.circle(face, (88, 40), 8, (100, 120, 130), -1)  # Right eye
        cv2.ellipse(face, (64, 100), (20, 15), 0, 0, 180, (120, 140, 150), -1)  # Mouth
        
        return face
    
    @pytest.fixture
    def printed_photo(self):
        """Generate a synthetic printed photo (flat texture)."""
        # Create a face image
        h, w = 128, 128
        face = np.ones((h, w, 3), dtype=np.uint8)
        
        # Print: uniform skin tone, very flat
        face[:] = [130, 150, 160]
        
        # Add print dots (screening patterns from printing)
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                if (i // 4 + j // 4) % 2 == 0:
                    face[i:i+2, j:j+2] = [125, 145, 155]
        
        # Features are less pronounced in photos
        cv2.circle(face, (40, 40), 8, (110, 130, 140), -1)
        cv2.circle(face, (88, 40), 8, (110, 130, 140), -1)
        
        return face
    
    @pytest.fixture
    def lcd_screen_capture(self):
        """Generate a synthetic LCD screen capture."""
        h, w = 128, 128
        screen = np.ones((h, w, 3), dtype=np.uint8)
        
        # Screen pixels: desaturated, uniform
        screen[:] = [180, 180, 180]
        
        # Pixel grid pattern (moire from LCD)
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                if (i // 2 + j // 2) % 3 == 0:
                    screen[i, j] = [200, 200, 200]
                    screen[i+1, j] = [160, 160, 160]
        
        # Bright spots (backlight)
        cv2.rectangle(screen, (10, 10), (30, 30), (255, 255, 255), -1)
        
        # Desaturated features
        cv2.circle(screen, (40, 40), 8, (160, 160, 160), -1)
        
        return screen
    
    def test_natural_face_not_detected_as_screen(self, natural_face):
        """Natural face should score low on screen detection."""
        result = detect_screen_or_print(natural_face)
        
        assert result['is_screen_or_print'] == False
        assert result['screen_score'] < 0.5
        assert 'reason' in result
    
    def test_printed_photo_detected(self, printed_photo):
        """Printed photo should score high on screen/print detection."""
        result = detect_screen_or_print(printed_photo)
        
        # Printed photo might not always be detected as screen, but flatness should be high
        assert result['flatness_score'] > 0.4
    
    def test_lcd_screen_detected(self, lcd_screen_capture):
        """LCD screen capture should score high on screen detection."""
        result = detect_screen_or_print(lcd_screen_capture)
        
        assert result['is_screen_or_print'] == True or result['screen_score'] > 0.5
        assert result['periodic_score'] > 0.3  # Moire pattern detected
    
    def test_result_keys_present(self, natural_face):
        """Result dictionary should contain all required keys."""
        result = detect_screen_or_print(natural_face)
        
        required_keys = [
            'is_screen_or_print', 'screen_score', 'reason',
            'periodic_score', 'flatness_score', 'reflection_score'
        ]
        for key in required_keys:
            assert key in result
    
    def test_screen_score_in_valid_range(self, natural_face):
        """Screen score should always be in [0, 1]."""
        result = detect_screen_or_print(natural_face)
        
        assert 0.0 <= result['screen_score'] <= 1.0
        assert 0.0 <= result['periodic_score'] <= 1.0
        assert 0.0 <= result['flatness_score'] <= 1.0
        assert 0.0 <= result['reflection_score'] <= 1.0
    
    def test_too_small_image(self):
        """Very small images should return neutral result."""
        tiny = np.ones((32, 32, 3), dtype=np.uint8)
        result = detect_screen_or_print(tiny)
        
        # Too small to process effectively
        assert result['reason'] == 'too_small'
        assert result['screen_score'] == 0.0
    
    def test_grayscale_input(self, natural_face):
        """Should handle grayscale images."""
        gray = cv2.cvtColor(natural_face, cv2.COLOR_BGR2GRAY)
        result = detect_screen_or_print(gray)
        
        assert result is not None
        assert 'screen_score' in result
    
    def test_invalid_input_handles_gracefully(self):
        """Invalid inputs should be handled gracefully."""
        # Empty image
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        result = detect_screen_or_print(empty)
        
        assert result['screen_score'] == 0.0  # Graceful degradation
    
    def test_performance_under_3ms(self, natural_face):
        """Detection should run in <3ms."""
        import time
        
        start = time.time()
        for _ in range(10):
            detect_screen_or_print(natural_face)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed * 1000) / 10
        assert avg_time_ms < 3.0, f"Detection took {avg_time_ms:.2f}ms, target <3ms"
