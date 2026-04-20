"""
Challenge-response module for interactive anti-spoofing.

Implements challenge prompts (blink, smile, move) and validates user responses
using facial landmarks and motion history.
"""

import random
import time
import numpy as np
from typing import Optional, Tuple

from core.utils import setup_logging

logger = setup_logging()


class ChallengeResponse:
    """Manages interactive anti-spoofing challenges and validation."""
    
    CHALLENGE_TYPES = ["blink", "smile", "move_left", "move_right"]
    
    def __init__(self):
        """Initialize challenge-response system."""
        self.current_challenge = None
        self.challenge_issued_at = None
        self.validation_timeout_seconds = 10.0
    
    def generate_challenge(self) -> str:
        """
        Generate a random challenge prompt.
        
        Returns:
            Challenge type: "blink", "smile", "move_left", "move_right"
        """
        self.current_challenge = random.choice(self.CHALLENGE_TYPES)
        self.challenge_issued_at = time.monotonic()
        return self.current_challenge
    
    def _ear_from_landmarks(
        self,
        landmarks: np.ndarray,
        eye_indices: Optional[list] = None
    ) -> Tuple[float, float]:
        """
        Calculate Eye Aspect Ratio (EAR) from landmarks.
        
        Uses the formula from Soukupová & Terzopoulos:
        EAR = ||p2 - p6|| + ||p3 - p5|| / (2 * ||p1 - p4||)
        
        Args:
            landmarks: Facial landmarks (68 points x 2) or (5 points x 2)
            eye_indices: Optional custom indices for left/right eye
            
        Returns:
            (left_ear, right_ear) tuple
        """
        if landmarks is None or len(landmarks) < 68:
            return 0.0, 0.0
        
        try:
            # Dlib 68-point face landmarks
            # Left eye: indices 36-41
            # Right eye: indices 42-47
            
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            def compute_ear(eye_points):
                # Eye aspect ratio formula
                A = np.linalg.norm(eye_points[1] - eye_points[5])
                B = np.linalg.norm(eye_points[2] - eye_points[4])
                C = np.linalg.norm(eye_points[0] - eye_points[3])
                ear = (A + B) / (2.0 * C) if C > 0 else 0.0
                return ear
            
            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            
            return float(left_ear), float(right_ear)
            
        except Exception as exc:
            logger.debug(f"EAR computation failed: {exc}")
            return 0.0, 0.0
    
    def _mouth_opening_ratio(self, landmarks: np.ndarray) -> float:
        """
        Calculate mouth opening ratio from landmarks.
        
        Uses distance between upper and lower lips.
        
        Args:
            landmarks: Facial landmarks (68 points x 2)
            
        Returns:
            Mouth opening ratio [0.0-1.0]
        """
        if landmarks is None or len(landmarks) < 68:
            return 0.0
        
        try:
            # Mouth region: indices 48-67
            # Upper lip: 50-53
            # Lower lip: 56-59
            
            upper_lip_center = np.mean(landmarks[50:54], axis=0)
            lower_lip_center = np.mean(landmarks[56:60], axis=0)
            
            # Vertical distance (opening)
            mouth_open = np.linalg.norm(upper_lip_center - lower_lip_center)
            
            # Normalize by mouth width
            mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
            
            if mouth_width > 0:
                ratio = mouth_open / mouth_width
            else:
                ratio = 0.0
            
            return float(min(1.0, ratio))
            
        except Exception as exc:
            logger.debug(f"Mouth opening ratio computation failed: {exc}")
            return 0.0
    
    def validate_response(
        self,
        landmarks: np.ndarray,
        motion_history: list,
        challenge_type: Optional[str] = None,
        ear_threshold: float = 0.21,
        mouth_threshold: float = 0.3,
        motion_threshold: float = 8.0
    ) -> float:
        """
        Validate if user performed the required challenge action.
        
        Args:
            landmarks: Facial landmarks (68 points x 2)
            motion_history: List of motion magnitudes (pixels) for recent frames
            challenge_type: Type of challenge ("blink", "smile", "move_left", "move_right")
                           If None, uses current_challenge
            ear_threshold: EAR threshold for detecting closed eye
            mouth_threshold: Mouth opening ratio threshold for smile
            motion_threshold: Minimum motion magnitude (pixels) for move challenges
            
        Returns:
            Validation confidence [0.0-1.0]
            - 1.0: Challenge successfully validated
            - 0.0: Challenge not validated
            - 0.5: Challenge in progress or ambiguous
        """
        if challenge_type is None:
            challenge_type = self.current_challenge
        
        if challenge_type not in self.CHALLENGE_TYPES:
            logger.warning(f"Invalid challenge type: {challenge_type}")
            return 0.0
        
        # Check timeout
        if self.challenge_issued_at is not None:
            elapsed = time.monotonic() - self.challenge_issued_at
            if elapsed > self.validation_timeout_seconds:
                logger.debug(f"Challenge timeout after {elapsed:.1f}s")
                return 0.0
        
        try:
            if challenge_type == "blink":
                return self._validate_blink(landmarks, ear_threshold)
            
            elif challenge_type == "smile":
                return self._validate_smile(landmarks, mouth_threshold)
            
            elif challenge_type in ["move_left", "move_right"]:
                return self._validate_move(motion_history, motion_threshold)
            
            else:
                return 0.0
        
        except Exception as exc:
            logger.warning(f"Challenge validation failed: {exc}")
            return 0.0
    
    def _validate_blink(self, landmarks: np.ndarray, ear_threshold: float) -> float:
        """
        Validate blink challenge.
        
        Requires at least 2 consecutive frames with EAR < threshold.
        
        Args:
            landmarks: Facial landmarks
            ear_threshold: EAR threshold for closed eye
            
        Returns:
            Confidence [0.0-1.0]
        """
        left_ear, right_ear = self._ear_from_landmarks(landmarks)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Blink detected if both eyes have low EAR
        blink_detected = avg_ear < ear_threshold
        
        if blink_detected:
            return 0.8  # Good confidence for detected blink
        else:
            return 0.0
    
    def _validate_smile(self, landmarks: np.ndarray, mouth_threshold: float) -> float:
        """
        Validate smile challenge.
        
        Requires mouth opening above threshold.
        
        Args:
            landmarks: Facial landmarks
            mouth_threshold: Mouth opening ratio threshold
            
        Returns:
            Confidence [0.0-1.0]
        """
        mouth_open = self._mouth_opening_ratio(landmarks)
        
        # Smile detected if mouth opening exceeds threshold
        smile_detected = mouth_open > mouth_threshold
        
        if smile_detected:
            # Scale confidence by how much mouth is open
            return min(1.0, mouth_open / (mouth_threshold * 2.0))
        else:
            return 0.0
    
    def _validate_move(
        self,
        motion_history: list,
        motion_threshold: float
    ) -> float:
        """
        Validate move challenge.
        
        Requires horizontal motion magnitude above threshold.
        
        Args:
            motion_history: List of motion magnitudes
            motion_threshold: Minimum motion magnitude
            
        Returns:
            Confidence [0.0-1.0]
        """
        if not motion_history or len(motion_history) < 3:
            return 0.0
        
        # Average recent motion
        recent_motion = np.mean(motion_history[-5:])
        
        # Check if motion exceeds threshold
        move_detected = recent_motion > motion_threshold
        
        if move_detected:
            # Scale confidence by how much motion occurred
            return min(1.0, recent_motion / (motion_threshold * 2.0))
        else:
            return 0.0
    
    def reset_challenge(self):
        """Reset current challenge."""
        self.current_challenge = None
        self.challenge_issued_at = None
