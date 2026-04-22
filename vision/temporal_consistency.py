"""Temporal consistency detection for identifying static faces (photos).

Analyzes motion and landmark changes over a rolling window of frames to detect:
1. Unnatural positional stability (photo won't move naturally)
2. Frozen landmarks (eyes/mouth in identical position)
3. Oscillation patterns (video loop or screen refresh)

Runs in <1ms per frame using only NumPy.
"""

import numpy as np
from core.utils import setup_logging

logger = setup_logging()


def compute_temporal_consistency(
    bbox_history: list,
    face_center_history: list,
    landmarks_history: list,
    window_size: int = 5,
) -> dict:
    """Analyze temporal stability of detected face to identify static photos.
    
    Args:
        bbox_history: List of tuples (x, y, w, h), oldest first
        face_center_history: List of tuples (cx, cy)
        landmarks_history: List of numpy arrays (5, 2) with 5-point landmarks
        window_size: Number of recent frames to analyze (default 5)
        
    Returns:
        dict with keys:
            - temporal_score: float [0, 1], higher = more natural motion
            - is_static: bool, True if motion is suspiciously low
            - issues: list of detected problems
            - center_motion_std: float, standard deviation of center movement
            - bbox_area_std: float, standard deviation of bbox area changes
            - landmark_variance: float, mean variance of landmark positions
    """
    # Validate input
    if not bbox_history or not face_center_history:
        return {
            "temporal_score": 0.5,  # Neutral
            "is_static": False,
            "issues": ["insufficient_history"],
            "center_motion_std": 0.0,
            "bbox_area_std": 0.0,
            "landmark_variance": 0.0,
        }
    
    try:
        # Use most recent N frames
        history_len = min(window_size, len(bbox_history), len(face_center_history))
        
        if history_len < 2:
            return {
                "temporal_score": 0.5,
                "is_static": False,
                "issues": ["insufficient_frames"],
                "center_motion_std": 0.0,
                "bbox_area_std": 0.0,
                "landmark_variance": 0.0,
            }
        
        # Extract recent history
        bbox_window = bbox_history[-history_len:]
        center_window = face_center_history[-history_len:]
        landmark_window = landmarks_history[-history_len:] if landmarks_history else []
        
        # Compute motion metrics
        center_motion_std = _compute_center_motion_variance(center_window)
        bbox_area_std = _compute_bbox_area_variance(bbox_window)
        landmark_variance = _compute_landmark_variance(landmark_window)
        oscillation_score = _detect_oscillation_patterns(center_window)
        
        # Combine into final score
        # Natural face: should have measurable motion
        # Photo: will be completely static or have periodic motion
        
        issues = []
        
        # Check for too-static position
        if center_motion_std < 0.5:
            issues.append("position_too_static")
        
        # Check for frozen landmarks
        if landmark_variance < 0.001:
            issues.append("landmarks_frozen")
        
        # Check for suspicious oscillation (video loop)
        if oscillation_score > 0.7:
            issues.append("oscillation_detected")
        
        # Compute final temporal score
        # Natural motion: center_std ~2-5px, bbox_std ~0.02-0.05, no oscillation
        # Static photo: center_std ~0.1px, bbox_std ~0.001, or perfect oscillation
        
        # Normalize each component to [0, 1]
        # 0 = static/unnatural, 1 = natural motion
        
        # Center motion: expected natural variation ~2-5px per frame at 30fps
        # Scale: 0.5px std → 0.0, 3px std → 1.0
        center_motion_score = min(1.0, max(0.0, (center_motion_std - 0.3) / 3.0))
        
        # Bbox area variation: expected ~1-3% per frame for head movements
        # Scale: 0.001 std → 0.0, 0.03 std → 1.0
        bbox_area_score = min(1.0, max(0.0, (bbox_area_std - 0.001) / 0.03))
        
        # Landmark variance: should be >0.01 for natural eyes/mouth movement
        # Scale: 0.001 → 0.0, 0.05 → 1.0
        landmark_score = min(1.0, max(0.0, (landmark_variance - 0.001) / 0.05))
        
        # Oscillation: low score is good (no periodic pattern)
        oscillation_penalty = 1.0 - oscillation_score  # Invert
        
        # Combine with weights
        temporal_score = (
            0.4 * center_motion_score +
            0.3 * bbox_area_score +
            0.2 * landmark_score +
            0.1 * oscillation_penalty
        )
        
        # Hard thresholds for determining if static
        is_static = (
            center_motion_std < 0.5 and
            bbox_area_std < 0.005 and
            oscillation_score < 0.3
        )
        
        return {
            "temporal_score": float(temporal_score),
            "is_static": is_static,
            "issues": issues,
            "center_motion_std": float(center_motion_std),
            "bbox_area_std": float(bbox_area_std),
            "landmark_variance": float(landmark_variance),
        }
        
    except Exception as e:
        logger.debug(f"Temporal consistency error: {e}")
        return {
            "temporal_score": 0.5,
            "is_static": False,
            "issues": ["computation_error"],
            "center_motion_std": 0.0,
            "bbox_area_std": 0.0,
            "landmark_variance": 0.0,
        }


def _compute_center_motion_variance(center_history: list) -> float:
    """Compute variance in face center movement (pixel displacement).
    
    Args:
        center_history: List of (cx, cy) tuples
        
    Returns:
        float, standard deviation of Euclidean distances between consecutive frames
    """
    if len(center_history) < 2:
        return 0.0
    
    try:
        center_array = np.array(center_history, dtype=np.float32)
        
        # Compute Euclidean distances between consecutive frames
        displacements = np.linalg.norm(
            center_array[1:] - center_array[:-1], axis=1
        )
        
        # Return standard deviation of displacements
        return float(np.std(displacements))
        
    except Exception as e:
        logger.debug(f"Center motion variance error: {e}")
        return 0.0


def _compute_bbox_area_variance(bbox_history: list) -> float:
    """Compute variance in bounding box area changes.
    
    Args:
        bbox_history: List of (x, y, w, h) tuples
        
    Returns:
        float, standard deviation of relative area changes between frames
    """
    if len(bbox_history) < 2:
        return 0.0
    
    try:
        # Extract widths and heights
        areas = np.array([(w * h) for x, y, w, h in bbox_history], dtype=np.float32)
        
        # Compute relative changes: (a[i+1] - a[i]) / a[i]
        relative_changes = np.abs((areas[1:] - areas[:-1]) / (areas[:-1] + 1e-8))
        
        # Return standard deviation of relative changes
        return float(np.std(relative_changes))
        
    except Exception as e:
        logger.debug(f"Bbox area variance error: {e}")
        return 0.0


def _compute_landmark_variance(landmark_history: list) -> float:
    """Compute mean variance of landmark positions.
    
    Args:
        landmark_history: List of numpy arrays (5, 2) with 5-point landmarks
        
    Returns:
        float, mean standard deviation across all 5 landmarks
    """
    if not landmark_history or len(landmark_history) < 2:
        return 0.0
    
    try:
        # Stack all landmarks into (frames, 5, 2) array
        landmark_stack = np.array(landmark_history, dtype=np.float32)
        
        if landmark_stack.shape[1] < 5:
            return 0.0
        
        # For each of 5 landmarks, compute variance across frames
        variances = []
        for i in range(5):
            landmark_positions = landmark_stack[:, i, :]  # All frames for landmark i
            # Variance as sum of squared distances from mean position
            mean_pos = np.mean(landmark_positions, axis=0)
            distances = np.linalg.norm(landmark_positions - mean_pos, axis=1)
            variance = np.var(distances)
            variances.append(variance)
        
        # Return mean variance across all landmarks
        return float(np.mean(variances))
        
    except Exception as e:
        logger.debug(f"Landmark variance error: {e}")
        return 0.0


def _detect_oscillation_patterns(center_history: list) -> float:
    """Detect periodic oscillation (video loop or screen refresh).
    
    Args:
        center_history: List of (cx, cy) tuples
        
    Returns:
        float [0, 1], score indicating oscillation strength
    """
    if len(center_history) < 4:
        return 0.0
    
    try:
        center_array = np.array(center_history, dtype=np.float32)
        
        # Compute displacements
        displacements = center_array[1:] - center_array[:-1]
        
        # Check for alternating pattern (moves A → returns to A)
        # This is characteristic of video loops or screen refresh artifacts
        
        # Look for repeating 2-frame cycles
        if len(displacements) >= 3:
            # Check if displacement[i] ≈ -displacement[i+1] (forward-backward)
            reverse_patterns = 0
            for i in range(len(displacements) - 1):
                # Similarity: how much do they cancel out?
                reversal = np.dot(displacements[i], -displacements[i+1])
                reversal /= (np.linalg.norm(displacements[i]) * np.linalg.norm(displacements[i+1]) + 1e-8)
                
                if reversal > 0.7:  # Strong reversal
                    reverse_patterns += 1
            
            oscillation_ratio = reverse_patterns / (len(displacements) - 1)
        else:
            oscillation_ratio = 0.0
        
        # Also check for periodicity in motion magnitude
        # Real natural motion has variation in speed; loops have constant speed
        magnitudes = np.linalg.norm(displacements, axis=1)
        magnitude_std = np.std(magnitudes)
        magnitude_mean = np.mean(magnitudes)
        
        # Low CV (coefficient of variation) indicates uniform periodic motion
        cv = magnitude_std / (magnitude_mean + 1e-8)
        uniform_motion_score = max(0.0, 1.0 - cv)
        
        # Combine both indicators
        oscillation_score = 0.6 * oscillation_ratio + 0.4 * uniform_motion_score
        
        return float(min(1.0, oscillation_score))
        
    except Exception as e:
        logger.debug(f"Oscillation detection error: {e}")
        return 0.0
