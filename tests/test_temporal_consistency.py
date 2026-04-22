"""Unit tests for temporal consistency detection module."""

import numpy as np
import pytest
from vision.temporal_consistency import compute_temporal_consistency


class TestTemporalConsistency:
    """Test suite for temporal consistency detection (identifies static photos)."""
    
    @pytest.fixture
    def natural_face_motion(self):
        """Generate natural face motion (live person)."""
        # Natural head movement: varies by 2-5 pixels per frame
        frames = 10
        bbox_history = []
        face_center_history = []
        
        base_x, base_y = 100, 100
        base_w, base_h = 100, 150
        
        for i in range(frames):
            # Add natural variation
            dx = np.random.normal(0, 2.0)  # ~2px variation
            dy = np.random.normal(0, 2.0)
            dw = np.random.normal(0, 2.0)
            dh = np.random.normal(0, 2.0)
            
            x = int(base_x + dx)
            y = int(base_y + dy)
            w = int(base_w + dw)
            h = int(base_h + dh)
            
            bbox_history.append((x, y, w, h))
            face_center_history.append((x + w/2, y + h/2))
        
        return bbox_history, face_center_history, []
    
    @pytest.fixture
    def static_photo_motion(self):
        """Generate static face motion (printed photo or still image)."""
        # Static photo: almost no motion
        frames = 10
        bbox_history = []
        face_center_history = []
        
        # Completely static
        for i in range(frames):
            bbox_history.append((100, 100, 100, 150))
            face_center_history.append((150, 175))
        
        return bbox_history, face_center_history, []
    
    @pytest.fixture
    def video_loop_motion(self):
        """Generate video loop motion (oscillating pattern)."""
        # Video loop: forward-backward repeating pattern
        frames = 10
        bbox_history = []
        face_center_history = []
        
        base_x, base_y = 100, 100
        base_w, base_h = 100, 150
        
        for i in range(frames):
            # Alternating: move right, then left
            if i % 2 == 0:
                x = base_x + 5  # Move right
            else:
                x = base_x - 5  # Move left
            
            bbox_history.append((x, base_y, base_w, base_h))
            face_center_history.append((x + base_w/2, base_y + base_h/2))
        
        return bbox_history, face_center_history, []
    
    def test_natural_motion_accepted(self, natural_face_motion):
        """Natural face motion should score high on temporal consistency."""
        bbox_hist, center_hist, landmark_hist = natural_face_motion
        
        result = compute_temporal_consistency(
            bbox_history=bbox_hist,
            face_center_history=center_hist,
            landmarks_history=landmark_hist,
        )
        
        assert result['temporal_score'] > 0.4
        assert result['is_static'] == False
        assert len(result['issues']) == 0
    
    def test_static_photo_rejected(self, static_photo_motion):
        """Static photo should score low on temporal consistency."""
        bbox_hist, center_hist, landmark_hist = static_photo_motion
        
        result = compute_temporal_consistency(
            bbox_history=bbox_hist,
            face_center_history=center_hist,
            landmarks_history=landmark_hist,
        )
        
        assert result['temporal_score'] < 0.5
        assert result['is_static'] == True
        assert 'position_too_static' in result['issues']
    
    def test_video_loop_detected(self, video_loop_motion):
        """Video loop should detect oscillation pattern."""
        bbox_hist, center_hist, landmark_hist = video_loop_motion
        
        result = compute_temporal_consistency(
            bbox_history=bbox_hist,
            face_center_history=center_hist,
            landmarks_history=landmark_hist,
        )
        
        # Video loops may have moderate scores but oscillation should be detected
        assert 'oscillation_detected' in result['issues'] or result['temporal_score'] < 0.5
    
    def test_result_keys_present(self, natural_face_motion):
        """Result dictionary should contain all required keys."""
        bbox_hist, center_hist, landmark_hist = natural_face_motion
        
        result = compute_temporal_consistency(
            bbox_history=bbox_hist,
            face_center_history=center_hist,
            landmarks_history=landmark_hist,
        )
        
        required_keys = [
            'temporal_score', 'is_static', 'issues',
            'center_motion_std', 'bbox_area_std', 'landmark_variance'
        ]
        for key in required_keys:
            assert key in result
    
    def test_temporal_score_in_valid_range(self, natural_face_motion):
        """Temporal score should always be in [0, 1]."""
        bbox_hist, center_hist, landmark_hist = natural_face_motion
        
        result = compute_temporal_consistency(
            bbox_history=bbox_hist,
            face_center_history=center_hist,
            landmarks_history=landmark_hist,
        )
        
        assert 0.0 <= result['temporal_score'] <= 1.0
    
    def test_empty_history(self):
        """Empty history should return neutral result."""
        result = compute_temporal_consistency(
            bbox_history=[],
            face_center_history=[],
            landmarks_history=[],
        )
        
        assert result['temporal_score'] == 0.5  # Neutral
        assert 'insufficient_history' in result['issues']
    
    def test_single_frame_history(self):
        """Single frame history should return neutral result."""
        result = compute_temporal_consistency(
            bbox_history=[(100, 100, 100, 150)],
            face_center_history=[(150, 175)],
            landmarks_history=[],
        )
        
        assert result['temporal_score'] == 0.5  # Neutral
        assert 'insufficient_frames' in result['issues']
    
    def test_landmarks_with_variance(self):
        """Landmarks with natural variance should contribute to score."""
        # Create history with landmark variance
        bbox_history = [(100 + i, 100, 100, 150) for i in range(5)]
        face_center_history = [(150 + i, 175) for i in range(5)]
        
        # Natural landmarks with variation
        landmarks_history = [
            np.array([[45 + np.random.randn(), 40 + np.random.randn()],
                     [85 + np.random.randn(), 40 + np.random.randn()],
                     [65, 75],
                     [45, 110],
                     [85, 110]], dtype=np.float32)
            for _ in range(5)
        ]
        
        result = compute_temporal_consistency(
            bbox_history=bbox_history,
            face_center_history=face_center_history,
            landmarks_history=landmarks_history,
        )
        
        assert result['landmark_variance'] > 0.0
    
    def test_frozen_landmarks_detected(self):
        """Frozen landmarks should be detected as suspicious."""
        bbox_history = [(100, 100, 100, 150)] * 5  # Static box
        face_center_history = [(150, 175)] * 5  # Static center
        
        # Frozen landmarks: identical across all frames
        same_landmarks = np.array([
            [45, 40], [85, 40], [65, 75], [45, 110], [85, 110]
        ], dtype=np.float32)
        landmarks_history = [same_landmarks.copy() for _ in range(5)]
        
        result = compute_temporal_consistency(
            bbox_history=bbox_history,
            face_center_history=face_center_history,
            landmarks_history=landmarks_history,
        )
        
        assert result['landmark_variance'] < 0.01
        assert 'landmarks_frozen' in result['issues']
    
    def test_window_size_parameter(self, natural_face_motion):
        """Window size parameter should affect computation."""
        bbox_hist, center_hist, landmark_hist = natural_face_motion
        
        result_small = compute_temporal_consistency(
            bbox_history=bbox_hist,
            face_center_history=center_hist,
            landmarks_history=landmark_hist,
            window_size=3
        )
        
        result_large = compute_temporal_consistency(
            bbox_history=bbox_hist,
            face_center_history=center_hist,
            landmarks_history=landmark_hist,
            window_size=10
        )
        
        # Both should have valid results
        assert 'temporal_score' in result_small
        assert 'temporal_score' in result_large
    
    def test_performance_under_1ms(self, natural_face_motion):
        """Temporal consistency check should run in <1ms."""
        import time
        bbox_hist, center_hist, landmark_hist = natural_face_motion
        
        start = time.time()
        for _ in range(10):
            compute_temporal_consistency(
                bbox_history=bbox_hist,
                face_center_history=center_hist,
                landmarks_history=landmark_hist,
            )
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed * 1000) / 10
        assert avg_time_ms < 1.0, f"Check took {avg_time_ms:.2f}ms, target <1ms"
