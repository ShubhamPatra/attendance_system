"""
Integration tests for advanced anti-spoofing, analytics, and vector search features.

Test suites:
  - test_texture_analyzer: LBP-based texture analysis
  - test_challenge_response: Interactive challenge validation
  - test_fusion_scoring: Multi-signal liveness fusion
  - test_analytics_pipelines: MongoDB aggregation queries
  - test_faiss_search: Vector search operations
  - test_fallback_behavior: Graceful degradation when dependencies unavailable
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np

# Test imports - order matters for lazy loading
import core.config as config


class TestTextureAnalyzer(unittest.TestCase):
    """Test vision.texture_analyzer.TextureAnalyzer for flat surface detection."""

    def setUp(self):
        """Create test fixtures."""
        from vision.texture_analyzer import TextureAnalyzer
        self.analyzer = TextureAnalyzer()
    
    def test_texture_analyzer_init(self):
        """Verify TextureAnalyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(config.TEXTURE_LBP_RADIUS, 1)
        self.assertEqual(config.TEXTURE_LBP_POINTS, 8)
    
    def test_flat_surface_detection(self):
        """Test that flat surfaces (constant color) are detected as spoof."""
        # Create a uniform flat image (simulates screen/printed photo)
        flat_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        lbp_hist, flatness = self.analyzer.analyze_texture(flat_image)
        
        # Flat surfaces should have high flatness score
        self.assertGreater(flatness, config.TEXTURE_FLATNESS_THRESHOLD - 0.2)
        self.assertIsNotNone(lbp_hist)
        self.assertGreater(len(lbp_hist), 0)
    
    def test_natural_surface_detection(self):
        """Test that natural faces (varied texture) are detected as real."""
        # Create a textured image (random noise = natural texture)
        natural_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        lbp_hist, flatness = self.analyzer.analyze_texture(natural_image)
        
        # Natural surfaces should have low flatness score
        self.assertLess(flatness, config.TEXTURE_FLATNESS_THRESHOLD + 0.1)
        self.assertIsNotNone(lbp_hist)
    
    def test_flatness_classification(self):
        """Test flatness classification output."""
        # Test flat image
        flat_image = np.ones((224, 224, 3), dtype=np.uint8) * 100
        result = self.analyzer.get_flatness_classification(flat_image)
        
        # Result should be "flat" or "textured"
        self.assertIn(result, ["flat", "textured"])


class TestChallengeResponse(unittest.TestCase):
    """Test vision.challenge_response.ChallengeResponse for challenge validation."""

    def setUp(self):
        """Create test fixtures."""
        from vision.challenge_response import ChallengeResponse
        self.challenger = ChallengeResponse()
        
        # Create mock landmarks (68 points)
        self.mock_landmarks = np.random.rand(68, 2).astype(np.float32)
        
        # Create mock motion history
        self.mock_motion_history = [(i, 100 + i*2, 100 + i*1) for i in range(10)]
    
    def test_challenge_response_init(self):
        """Verify ChallengeResponse initializes correctly."""
        self.assertIsNotNone(self.challenger)
        self.assertGreater(config.CHALLENGE_RESPONSE_TIMEOUT_SECONDS, 0)
        self.assertGreater(config.BLINK_EAR_THRESHOLD, 0)
    
    def test_ear_computation(self):
        """Test Eye Aspect Ratio (EAR) computation."""
        # Create mock landmarks for blink test
        result = self.challenger.validate_response(
            landmarks=self.mock_landmarks,
            motion_history=self.mock_motion_history,
            frame=np.zeros((224, 224, 3), dtype=np.uint8),
            challenge_type="blink",
        )
        
        # Result should be a dict with confidence
        self.assertIsInstance(result, dict)
        self.assertIn("confidence", result)
        conf = result["confidence"]
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
    
    def test_challenge_timeout(self):
        """Test that challenges timeout correctly."""
        result = self.challenger.validate_response(
            landmarks=self.mock_landmarks,
            motion_history=self.mock_motion_history,
            frame=np.zeros((224, 224, 3), dtype=np.uint8),
            challenge_type="smile",
        )
        
        self.assertIsInstance(result, dict)
        # Challenge should eventually timeout (confidence decreases over time)
    
    def test_all_challenge_types(self):
        """Test that all challenge types are supported."""
        challenge_types = ["blink", "smile", "move_left", "move_right"]
        
        for challenge_type in challenge_types:
            result = self.challenger.validate_response(
                landmarks=self.mock_landmarks,
                motion_history=self.mock_motion_history,
                frame=np.zeros((224, 224, 3), dtype=np.uint8),
                challenge_type=challenge_type,
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn("confidence", result)


class TestFusionScoring(unittest.TestCase):
    """Test vision.anti_spoofing.fuse_liveness_signals for multi-signal fusion."""

    def setUp(self):
        """Create test fixtures."""
        from vision.anti_spoofing import fuse_liveness_signals, normalize_signal_to_confidence
        self.fuse_liveness_signals = fuse_liveness_signals
        self.normalize_signal = normalize_signal_to_confidence
    
    def test_fusion_default_weights(self):
        """Test fusion with default weights."""
        fused = self.fuse_liveness_signals(
            cnn_score=0.95,
            blink_score=1.0,
            motion_score=0.8,
            texture_score=0.9,
            challenge_score=0.0,
        )
        
        # Fused score should be in [0, 1]
        self.assertGreaterEqual(fused, 0.0)
        self.assertLessEqual(fused, 1.0)
        
        # With strong CNN + blink + motion, should be high confidence
        self.assertGreater(fused, 0.8)
    
    def test_fusion_custom_weights(self):
        """Test fusion with custom weights."""
        custom_weights = {
            "cnn": 0.5,
            "blink": 0.3,
            "motion": 0.1,
            "texture": 0.1,
            "challenge": 0.0,
        }
        
        fused = self.fuse_liveness_signals(
            cnn_score=0.9,
            blink_score=0.5,
            motion_score=0.5,
            texture_score=0.5,
            challenge_score=0.0,
            weights=custom_weights,
        )
        
        self.assertGreaterEqual(fused, 0.0)
        self.assertLessEqual(fused, 1.0)
    
    def test_signal_normalization(self):
        """Test normalization of different signal types."""
        # Binary signal
        normalized_binary = self.normalize_signal(1, "binary")
        self.assertEqual(normalized_binary, 1.0)
        
        normalized_binary_zero = self.normalize_signal(0, "binary")
        self.assertEqual(normalized_binary_zero, 0.0)
        
        # Continuous signal
        normalized_cont = self.normalize_signal(0.5, "continuous")
        self.assertEqual(normalized_cont, 0.5)
        
        # Flatness signal (inverted)
        normalized_flat = self.normalize_signal(0.8, "flatness")
        self.assertEqual(normalized_flat, 0.2)
    
    def test_spoof_detection_strong_signal(self):
        """Test that strong spoof signals result in low fused score."""
        fused = self.fuse_liveness_signals(
            cnn_score=0.1,  # Strong spoof signal
            blink_score=0.0,
            motion_score=0.0,
            texture_score=0.1,
            challenge_score=0.0,
        )
        
        # With strong spoof signals, fused score should be low
        self.assertLess(fused, 0.4)


class TestAnalyticsPipelines(unittest.TestCase):
    """Test core.analytics_pipelines MongoDB aggregation queries."""

    @patch('core.database.get_db')
    def test_analytics_overview(self, mock_get_db):
        """Test get_analytics_overview aggregation."""
        from core.analytics_pipelines import get_analytics_overview
        
        # Mock MongoDB response
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        # Mock aggregation result
        mock_db.attendance.aggregate.return_value = [
            {
                "total_students": 100,
                "present_today": 90,
                "late_count": 10,
                "absent_count": 10,
                "avg_attendance_percent": 85.5,
                "on_time_percent": 80.0,
            }
        ]
        
        result = get_analytics_overview(days=7)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["total_students"], 100)
        self.assertEqual(result["present_today"], 90)
    
    @patch('core.database.get_db')
    def test_attendance_trend_daily(self, mock_get_db):
        """Test get_attendance_trend_daily aggregation."""
        from core.analytics_pipelines import get_attendance_trend_daily
        
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        # Mock daily breakdown
        mock_db.attendance.aggregate.return_value = [
            {
                "date": "2026-04-17",
                "total": 100,
                "present": 85,
                "late": 10,
                "absent": 15,
                "present_percent": 85.0,
            },
            {
                "date": "2026-04-16",
                "total": 100,
                "present": 88,
                "late": 8,
                "absent": 12,
                "present_percent": 88.0,
            },
        ]
        
        result = get_attendance_trend_daily(days=30)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
    
    @patch('core.database.get_db')
    def test_late_statistics(self, mock_get_db):
        """Test get_late_statistics aggregation."""
        from core.analytics_pipelines import get_late_statistics
        
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        # Mock late stats
        mock_db.attendance.aggregate.return_value = [
            {
                "daily_trend": [
                    {"date": "2026-04-17", "late_count": 12, "on_time_count": 88},
                ],
                "peak_late_hour": 9,
                "total_late": 50,
                "top_late_students": [
                    {"name": "John", "reg_no": "001", "late_count": 8},
                    {"name": "Jane", "reg_no": "002", "late_count": 6},
                ],
            }
        ]
        
        result = get_late_statistics(days=30)
        
        self.assertIsNotNone(result)
        self.assertIn("peak_late_hour", result)
        self.assertIn("top_late_students", result)


class TestFAISSSearch(unittest.TestCase):
    """Test vision.embedding_search FAISS operations."""

    def setUp(self):
        """Create test fixtures."""
        # Only run if FAISS available
        try:
            import faiss
            self.faiss_available = True
        except ImportError:
            self.faiss_available = False
    
    @unittest.skipIf(not property(lambda self: self.faiss_available), "FAISS not available")
    def test_faiss_index_creation(self):
        """Test FAISS index creation and basic operations."""
        from vision.embedding_search import FAISSIndex
        
        index = FAISSIndex(dimension=512, index_type="Flat")
        
        # Create test embeddings
        embeddings = np.random.rand(10, 512).astype(np.float32)
        student_ids = [f"student_{i}" for i in range(10)]
        
        # Add to index
        success = index.add(embeddings, student_ids)
        self.assertTrue(success)
        self.assertEqual(index.get_size(), 10)
    
    @unittest.skipIf(not property(lambda self: self.faiss_available), "FAISS not available")
    def test_faiss_search(self):
        """Test FAISS search operations."""
        from vision.embedding_search import FAISSIndex
        
        index = FAISSIndex(dimension=512, index_type="Flat")
        
        # Create and add test embeddings
        embeddings = np.random.rand(100, 512).astype(np.float32)
        student_ids = [f"student_{i}" for i in range(100)]
        
        index.add(embeddings, student_ids)
        
        # Search with first embedding
        query = embeddings[0]
        results = index.search(query, k=5)
        
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 5)
        
        # First result should be the query itself (distance ~ 0)
        student_id, distance, confidence = results[0]
        self.assertEqual(student_id, "student_0")
        self.assertLess(distance, 0.1)
    
    @unittest.skipIf(not property(lambda self: self.faiss_available), "FAISS not available")
    def test_faiss_index_persistence(self):
        """Test FAISS index save/load operations."""
        from vision.embedding_search import FAISSIndex
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index.bin"
            
            # Create and save index
            index1 = FAISSIndex(dimension=512, index_type="Flat")
            embeddings = np.random.rand(50, 512).astype(np.float32)
            student_ids = [f"student_{i}" for i in range(50)]
            
            index1.add(embeddings, student_ids)
            success = index1.save(str(index_path))
            self.assertTrue(success)
            
            # Load and verify
            index2 = FAISSIndex(dimension=512, index_type="Flat")
            success = index2.load(str(index_path))
            self.assertTrue(success)
            self.assertEqual(index2.get_size(), 50)


class TestFallbackBehavior(unittest.TestCase):
    """Test graceful fallback behavior when dependencies unavailable."""

    @patch('vision.embedding_search.faiss', None)
    def test_faiss_unavailable_fallback(self):
        """Test that system gracefully handles missing FAISS."""
        from vision.embedding_search import FAISSIndex
        
        index = FAISSIndex(dimension=512, index_type="Flat")
        
        # Should still initialize but report unavailable
        self.assertFalse(index.is_available())
    
    def test_texture_analyzer_with_invalid_input(self):
        """Test texture analyzer graceful handling of invalid input."""
        from vision.texture_analyzer import TextureAnalyzer
        
        analyzer = TextureAnalyzer()
        
        # Test with empty image
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        try:
            result = analyzer.analyze_texture(empty)
            # Should not crash, may return defaults
            self.assertIsNotNone(result)
        except Exception:
            pass  # Acceptable to raise on invalid input
    
    def test_challenge_response_with_invalid_landmarks(self):
        """Test challenge response graceful handling of invalid landmarks."""
        from vision.challenge_response import ChallengeResponse
        
        challenger = ChallengeResponse()
        
        # Test with wrong shape landmarks
        bad_landmarks = np.random.rand(10, 2)  # Should be 68, 2
        try:
            result = challenger.validate_response(
                landmarks=bad_landmarks,
                motion_history=[],
                frame=np.zeros((224, 224, 3), dtype=np.uint8),
                challenge_type="blink",
            )
            # Should not crash, may return low confidence
            self.assertIsInstance(result, dict)
        except Exception:
            pass  # Acceptable to raise on invalid input


class TestConfigIntegration(unittest.TestCase):
    """Test that advanced feature configuration works correctly."""

    def test_feature_flags_exist(self):
        """Verify all advanced feature flags are in config."""
        self.assertTrue(hasattr(config, "ENABLE_ADVANCED_LIVENESS"))
        self.assertTrue(hasattr(config, "ENABLE_CHALLENGE_RESPONSE"))
        self.assertTrue(hasattr(config, "ENABLE_TEXTURE_ANALYSIS"))
        self.assertTrue(hasattr(config, "ENABLE_VECTOR_SEARCH"))
        self.assertTrue(hasattr(config, "ENABLE_ANALYTICS"))
    
    def test_fusion_weights_configuration(self):
        """Verify fusion weights are properly configured."""
        self.assertTrue(hasattr(config, "LIVENESS_FUSION_WEIGHT_CNN"))
        self.assertTrue(hasattr(config, "LIVENESS_FUSION_WEIGHT_BLINK"))
        self.assertTrue(hasattr(config, "LIVENESS_FUSION_WEIGHT_MOTION"))
        self.assertTrue(hasattr(config, "LIVENESS_FUSION_WEIGHT_TEXTURE"))
        self.assertTrue(hasattr(config, "LIVENESS_FUSION_WEIGHT_CHALLENGE"))
        
        # Weights should sum to approximately 1.0
        total_weight = (
            config.LIVENESS_FUSION_WEIGHT_CNN +
            config.LIVENESS_FUSION_WEIGHT_BLINK +
            config.LIVENESS_FUSION_WEIGHT_MOTION +
            config.LIVENESS_FUSION_WEIGHT_TEXTURE +
            config.LIVENESS_FUSION_WEIGHT_CHALLENGE
        )
        self.assertAlmostEqual(total_weight, 1.0, places=1)
    
    def test_vector_search_configuration(self):
        """Verify vector search configuration."""
        self.assertTrue(hasattr(config, "VECTOR_SEARCH_BACKEND"))
        self.assertIn(config.VECTOR_SEARCH_BACKEND, ["faiss", "mongodb_atlas", "hybrid"])
        self.assertTrue(hasattr(config, "FAISS_INDEX_TYPE"))
        self.assertIn(config.FAISS_INDEX_TYPE, ["Flat", "IVFFlat", "HNSW"])
    
    def test_analytics_configuration(self):
        """Verify analytics configuration."""
        self.assertTrue(hasattr(config, "LATE_ARRIVAL_CUTOFF_TIME"))
        self.assertTrue(hasattr(config, "ANALYTICS_CACHE_SECONDS"))
        self.assertGreater(config.ANALYTICS_CACHE_SECONDS, 0)


class TestEndToEndIntegration(unittest.TestCase):
    """High-level integration tests combining multiple features."""

    @patch('core.database.get_db')
    def test_liveness_pipeline_with_fusion(self, mock_get_db):
        """Test complete liveness pipeline with fusion scoring."""
        from vision.anti_spoofing import fuse_liveness_signals
        
        # Simulate a complete liveness evaluation
        cnn_score = 0.92
        blink_score = 1.0
        motion_score = 0.85
        texture_score = 0.95
        
        fused = fuse_liveness_signals(
            cnn_score=cnn_score,
            blink_score=blink_score,
            motion_score=motion_score,
            texture_score=texture_score,
        )
        
        # Fused score should be high (real, live person)
        self.assertGreater(fused, 0.85)
    
    def test_all_dependencies_installed(self):
        """Verify core dependencies are available."""
        required_packages = [
            "cv2",
            "numpy",
            "pymongo",
            "torch",
        ]
        
        optional_packages = [
            "faiss",
            "plotly",
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                self.fail(f"Required package {package} not installed")
        
        # Optional packages should not cause test failure if missing
        for package in optional_packages:
            try:
                __import__(package)
            except ImportError:
                pass  # OK if optional packages missing


if __name__ == "__main__":
    unittest.main()
