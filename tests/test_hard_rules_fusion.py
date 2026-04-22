"""Unit tests for hard rules and updated fusion logic."""

import pytest
import numpy as np
from vision.anti_spoofing import apply_hard_rules, fuse_liveness_signals
import core.config as config


class TestHardRules:
    """Test suite for hard rejection rules."""
    
    def test_no_blinks_rejected(self):
        """Face with no blinks should be rejected."""
        passes, reason = apply_hard_rules(
            blink_count=0,
            motion_score=0.5,
            screen_score=0.2,
            temporal_score=0.7,
        )
        
        assert passes == False
        assert reason == "no_blinks_detected"
    
    def test_low_motion_rejected(self):
        """Face with low motion should be rejected."""
        passes, reason = apply_hard_rules(
            blink_count=1,
            motion_score=0.05,  # Below min threshold of 0.15
            screen_score=0.2,
            temporal_score=0.7,
        )
        
        assert passes == False
        assert reason == "insufficient_motion"
    
    def test_screen_detected_rejected(self):
        """Face with high screen score should be rejected."""
        passes, reason = apply_hard_rules(
            blink_count=1,
            motion_score=0.5,
            screen_score=0.75,  # Above rejection threshold of 0.60
            temporal_score=0.7,
        )
        
        assert passes == False
        assert reason == "screen_or_print_detected"
    
    def test_unstable_face_rejected(self):
        """Face with low temporal consistency should be rejected."""
        passes, reason = apply_hard_rules(
            blink_count=1,
            motion_score=0.5,
            screen_score=0.2,
            temporal_score=0.1,  # Below rejection threshold of 0.30
        )
        
        assert passes == False
        assert reason == "unstable_face"
    
    def test_all_rules_passed(self):
        """All passing rules should accept."""
        passes, reason = apply_hard_rules(
            blink_count=2,
            motion_score=0.5,
            screen_score=0.2,
            temporal_score=0.7,
        )
        
        assert passes == True
        assert reason == "passed_all_hard_rules"
    
    def test_disabled_hard_rules(self, monkeypatch):
        """Disabling hard rules should accept everything."""
        monkeypatch.setattr(config, 'LIVENESS_HARD_RULES_ENABLED', False)
        
        passes, reason = apply_hard_rules(
            blink_count=0,  # Normally would fail
            motion_score=0.05,  # Normally would fail
            screen_score=0.75,  # Normally would fail
            temporal_score=0.1,  # Normally would fail
        )
        
        assert passes == True
        assert reason == "hard_rules_disabled"
    
    def test_multiple_failures_reports_first(self):
        """Multiple failures should report first failing rule."""
        passes, reason = apply_hard_rules(
            blink_count=0,  # Fails
            motion_score=0.05,  # Also fails
            screen_score=0.2,
            temporal_score=0.7,
        )
        
        assert passes == False
        assert reason == "no_blinks_detected"  # First check


class TestFusionLogic:
    """Test suite for updated fusion logic with screen + temporal scores."""
    
    def test_basic_fusion(self):
        """Basic fusion should combine all components."""
        score = fuse_liveness_signals(
            cnn_score=0.8,
            blink_score=1.0,
            motion_score=0.6,
            texture_score=0.7,
            screen_score=0.2,
            temporal_score=0.8,
            blink_count=1,
        )
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should pass with good inputs
    
    def test_hard_rule_rejection_returns_zero(self):
        """Hard rule failures should return 0.0 score."""
        score = fuse_liveness_signals(
            cnn_score=0.99,  # Very high CNN confidence
            blink_score=1.0,
            motion_score=0.99,
            texture_score=0.99,
            screen_score=0.99,  # But high screen score → reject
            temporal_score=0.8,
            blink_count=1,
        )
        
        assert score == 0.0
    
    def test_screen_score_inverted(self):
        """High screen score should reduce final score."""
        # Same inputs except screen_score
        score_clean = fuse_liveness_signals(
            cnn_score=0.8,
            blink_score=1.0,
            motion_score=0.6,
            texture_score=0.7,
            screen_score=0.0,  # No screen
            temporal_score=0.8,
            blink_count=1,
        )
        
        score_screen = fuse_liveness_signals(
            cnn_score=0.8,
            blink_score=1.0,
            motion_score=0.6,
            texture_score=0.7,
            screen_score=0.5,  # Some screen
            temporal_score=0.8,
            blink_count=1,
        )
        
        assert score_clean > score_screen
    
    def test_temporal_score_contributes(self):
        """Temporal score should affect final fusion."""
        score_good_temporal = fuse_liveness_signals(
            cnn_score=0.8,
            blink_score=1.0,
            motion_score=0.6,
            texture_score=0.7,
            screen_score=0.2,
            temporal_score=0.9,  # Good temporal
            blink_count=1,
        )
        
        score_bad_temporal = fuse_liveness_signals(
            cnn_score=0.8,
            blink_score=1.0,
            motion_score=0.6,
            texture_score=0.7,
            screen_score=0.2,
            temporal_score=0.31,  # Just above threshold
            blink_count=1,
        )
        
        # Both pass hard rules but temporal affects final score
        assert score_good_temporal > score_bad_temporal
    
    def test_output_normalized(self):
        """Output should always be in [0, 1]."""
        test_cases = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1),
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1),
            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1),
        ]
        
        for cnn, blink, motion, texture, screen, temporal, bc in test_cases:
            score = fuse_liveness_signals(
                cnn_score=cnn,
                blink_score=blink,
                motion_score=motion,
                texture_score=texture,
                screen_score=screen,
                temporal_score=temporal,
                blink_count=bc,
            )
            assert 0.0 <= score <= 1.0
    
    def test_no_blink_rejection(self):
        """Blink count 0 should fail hard rules and return 0.0."""
        score = fuse_liveness_signals(
            cnn_score=0.99,
            blink_score=0.0,
            motion_score=0.99,
            texture_score=0.99,
            screen_score=0.1,
            temporal_score=0.99,
            blink_count=0,  # No blinks
        )
        
        assert score == 0.0
    
    def test_weights_configuration(self, monkeypatch):
        """Should use configurable weights."""
        # Override weights to test they're being used
        monkeypatch.setattr(config, 'LIVENESS_FUSION_WEIGHT_CNN_NEW', 1.0)
        monkeypatch.setattr(config, 'LIVENESS_FUSION_WEIGHT_BLINK_NEW', 0.0)
        monkeypatch.setattr(config, 'LIVENESS_FUSION_WEIGHT_MOTION_NEW', 0.0)
        monkeypatch.setattr(config, 'LIVENESS_FUSION_WEIGHT_TEXTURE_NEW', 0.0)
        monkeypatch.setattr(config, 'LIVENESS_FUSION_WEIGHT_SCREEN_NEW', 0.0)
        monkeypatch.setattr(config, 'LIVENESS_FUSION_WEIGHT_TEMPORAL_NEW', 0.0)
        
        # With 100% weight on CNN, score should equal CNN score
        score = fuse_liveness_signals(
            cnn_score=0.6,
            blink_score=0.0,
            motion_score=0.0,
            texture_score=0.0,
            screen_score=0.0,
            temporal_score=0.5,
            blink_count=1,
        )
        
        assert abs(score - 0.6) < 0.01
    
    def test_handles_extreme_values(self):
        """Should handle extreme input values gracefully."""
        score = fuse_liveness_signals(
            cnn_score=2.0,  # Out of range high
            blink_score=-1.0,  # Out of range low
            motion_score=0.5,
            texture_score=0.5,
            screen_score=0.5,
            temporal_score=0.5,
            blink_count=1,
        )
        
        # Should clamp to valid range
        assert 0.0 <= score <= 1.0
    
    def test_fusion_decision_threshold(self):
        """Score >= 0.5 should indicate LIVE decision."""
        # High quality face: should pass threshold
        high_quality_score = fuse_liveness_signals(
            cnn_score=0.85,
            blink_score=1.0,
            motion_score=0.7,
            texture_score=0.8,
            screen_score=0.1,
            temporal_score=0.85,
            blink_count=2,
        )
        
        assert high_quality_score >= 0.5
        
        # Low quality face: should fail threshold
        low_quality_score = fuse_liveness_signals(
            cnn_score=0.3,
            blink_score=0.0,
            motion_score=0.2,
            texture_score=0.3,
            screen_score=0.6,
            temporal_score=0.2,
            blink_count=0,
        )
        
        assert low_quality_score < 0.5 or low_quality_score == 0.0


class TestIntegration:
    """Integration tests for hard rules + fusion pipeline."""
    
    def test_pipeline_photo_attack(self):
        """Photo attack should be rejected."""
        # Simulating a printed photo: no blinks, low motion
        score = fuse_liveness_signals(
            cnn_score=0.95,  # Photo might fool CNN
            blink_score=0.0,  # But no blinks
            motion_score=0.0,  # No motion
            texture_score=0.6,
            screen_score=0.7,  # Detected as screen/print
            temporal_score=0.1,  # Static
            blink_count=0,  # Critical: no blinks
        )
        
        # Hard rule failure due to no blinks
        assert score == 0.0
    
    def test_pipeline_live_face(self):
        """Live face should be accepted."""
        # Natural person: blinks, moves, low screen score
        score = fuse_liveness_signals(
            cnn_score=0.82,
            blink_score=1.0,
            motion_score=0.7,
            texture_score=0.75,
            screen_score=0.15,
            temporal_score=0.8,
            blink_count=2,
        )
        
        # Should pass all hard rules and have high fusion score
        assert score > 0.6
    
    def test_pipeline_screen_attack(self):
        """LCD screen attack should be rejected."""
        # High screen detection score should trigger hard rule rejection
        score = fuse_liveness_signals(
            cnn_score=0.88,
            blink_score=1.0,
            motion_score=0.5,
            texture_score=0.6,
            screen_score=0.75,  # Screen detected
            temporal_score=0.4,
            blink_count=1,
        )
        
        # High screen score fails hard rule
        assert score == 0.0
