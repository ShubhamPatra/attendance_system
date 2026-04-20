"""
Enrollment image validation and quality control.

PHASE 4: Validates enrollment images for:
- Blur detection (Laplacian variance threshold)
- Face size validation (min 80x80 pixels in frame)
- Lighting/brightness checks (40-250 range)
- Face angle detection (yaw/pitch/roll < 30°)
- Multi-angle enforcement (3-5 images with >30° yaw spread)
- Duplicate face detection (cosine similarity > 0.92)

Used during student registration to ensure high-quality, diverse facial data.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
from bson import ObjectId

import core.config as config
from core.utils import setup_logging
import core.database as database
from vision.face_engine import get_arcface_backend, generate_encoding

logger = setup_logging()


class EnrollmentValidationError(Exception):
    """Custom exception for enrollment validation failures."""
    pass


class EnrollmentValidator:
    """Validates student enrollment images during registration.
    
    PHASE 4: Multi-angle enrollment requirement with image quality control.
    """

    # Image quality thresholds
    MIN_BLUR_SHARPNESS = 100.0  # Laplacian variance
    MIN_FACE_SIZE_PIXELS = 80  # Min 80x80 face in frame
    MIN_BRIGHTNESS = 40.0
    MAX_BRIGHTNESS = 250.0
    MAX_FACE_ANGLE_DEGREES = 30.0

    # Multi-angle requirements
    MIN_ENROLLMENT_IMAGES = 3
    MAX_ENROLLMENT_IMAGES = 5
    MIN_YAW_SPREAD_DEGREES = 30.0  # Require diversity in angles

    @staticmethod
    def validate_image_quality(image_bgr: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Validate basic image quality (not face-specific).
        
        Args:
            image_bgr: BGR image array
        
        Returns:
            (is_valid, error_reason)
        """
        if image_bgr is None or image_bgr.size == 0:
            return False, "Image is empty or invalid"

        if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
            return False, "Image must be 3-channel BGR"

        h, w = image_bgr.shape[:2]
        if h < 240 or w < 320:
            return False, f"Image too small ({w}x{h}, min 320x240)"

        return True, None

    @staticmethod
    def detect_face_in_image(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect a face in the image using ArcFace.
        
        Returns (x, y, w, h) bounding box in XYWH format, or None if no face found.
        """
        try:
            af = get_arcface_backend()
            faces = af.get_faces(image_bgr)
            if not faces:
                return None

            # Get largest face
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = face.bbox
            w = int(x2 - x1)
            h = int(y2 - y1)
            x = int(x1)
            y = int(y1)
            return (x, y, w, h)
        except Exception as e:
            logger.warning("Failed to detect face using ArcFace: %s", e)
            return None

    @staticmethod
    def assess_face_quality(image_bgr: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> Tuple[bool, Optional[str]]:
        """Assess quality of detected face.
        
        Checks:
        - Face size (min 80x80)
        - Blur (Laplacian variance > 100)
        - Brightness (40-250)
        - Face area ratio in frame
        
        Returns:
            (is_valid, error_reason)
        """
        x, y, w, h = bbox_xywh
        fh, fw = image_bgr.shape[:2]

        # Face size check
        if w < EnrollmentValidator.MIN_FACE_SIZE_PIXELS or h < EnrollmentValidator.MIN_FACE_SIZE_PIXELS:
            return False, f"Face too small ({w}x{h}px, min {EnrollmentValidator.MIN_FACE_SIZE_PIXELS}px)"

        # Face area ratio check
        face_area = w * h
        frame_area = fw * fh
        face_area_ratio = face_area / frame_area
        if face_area_ratio < 0.02:  # Face should be at least 2% of frame
            return False, f"Face too small in frame (area ratio {face_area_ratio:.1%}, min 2%)"

        # Extract ROI for quality analysis
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)
        roi = image_bgr[y1:y2, x1:x2]

        if roi.size == 0:
            return False, "Invalid face ROI"

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Blur check (Laplacian variance)
        blur_variance = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        if blur_variance < EnrollmentValidator.MIN_BLUR_SHARPNESS:
            return False, f"Face too blurry (sharpness {blur_variance:.0f}, min {EnrollmentValidator.MIN_BLUR_SHARPNESS:.0f})"

        # Brightness check
        brightness = float(np.mean(gray_roi))
        if brightness < EnrollmentValidator.MIN_BRIGHTNESS:
            return False, f"Face too dark (brightness {brightness:.0f}, min {EnrollmentValidator.MIN_BRIGHTNESS:.0f})"
        if brightness > EnrollmentValidator.MAX_BRIGHTNESS:
            return False, f"Face too bright (brightness {brightness:.0f}, max {EnrollmentValidator.MAX_BRIGHTNESS:.0f})"

        return True, None

    @staticmethod
    def extract_face_angles(image_bgr: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> Optional[Dict[str, float]]:
        """Extract yaw, pitch, roll angles from face using landmarks.
        
        Returns dict with 'yaw', 'pitch', 'roll' in degrees, or None on error.
        """
        try:
            af = get_arcface_backend()
            faces = af.get_faces(image_bgr)
            if not faces:
                return None

            # Get largest face
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            # Extract angles if available
            if hasattr(face, 'pose') and face.pose is not None:
                return {
                    "yaw": float(face.pose[0]),
                    "pitch": float(face.pose[1]),
                    "roll": float(face.pose[2]),
                }
            return None
        except Exception as e:
            logger.debug("Failed to extract face angles: %s", e)
            return None

    @staticmethod
    def check_face_angle_validity(angles: Optional[Dict[str, float]]) -> Tuple[bool, Optional[str]]:
        """Check if face angles are acceptable for enrollment.
        
        Rejects faces with yaw/pitch/roll > 30°.
        """
        if angles is None:
            return True, None  # If we can't extract angles, allow it

        max_angle = max(abs(angles.get("yaw", 0)), abs(angles.get("pitch", 0)), abs(angles.get("roll", 0)))
        if max_angle > EnrollmentValidator.MAX_FACE_ANGLE_DEGREES:
            return False, f"Face angle too extreme (max {max_angle:.0f}°, max {EnrollmentValidator.MAX_FACE_ANGLE_DEGREES}°)"

        return True, None

    @staticmethod
    def validate_enrollment_image(image_bgr: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Full validation of a single enrollment image.
        
        Returns:
            (is_valid, error_reason)
        """
        # 1. Basic image quality
        valid, reason = EnrollmentValidator.validate_image_quality(image_bgr)
        if not valid:
            return False, reason

        # 2. Detect face
        bbox = EnrollmentValidator.detect_face_in_image(image_bgr)
        if bbox is None:
            return False, "No face detected in image"

        # 3. Face quality
        valid, reason = EnrollmentValidator.assess_face_quality(image_bgr, bbox)
        if not valid:
            return False, reason

        # 4. Face angle
        angles = EnrollmentValidator.extract_face_angles(image_bgr, bbox)
        valid, reason = EnrollmentValidator.check_face_angle_validity(angles)
        if not valid:
            return False, reason

        return True, None

    @staticmethod
    def check_duplicate_face(student_id: ObjectId, candidate_image_bgr: np.ndarray, threshold: float = 0.92) -> Tuple[bool, Optional[str]]:
        """Check if candidate face already exists (for a different student).
        
        Returns:
            (is_duplicate, error_reason)
        """
        try:
            # Generate embedding for candidate
            candidate_encoding = generate_encoding(candidate_image_bgr)
            if candidate_encoding is None:
                return False, None  # Can't check, allow it

            # Check for similar faces in database
            from vision.face_engine import encoding_cache
            flat_enc, flat_idx, ids, names = encoding_cache.get_flat()
            if flat_enc is None or len(flat_enc) == 0:
                return False, None

            # L2-normalize candidate
            query = candidate_encoding.astype(np.float32).flatten()
            q_norm = np.linalg.norm(query)
            if q_norm > 0:
                query = query / q_norm

            # Compute similarities
            similarities = flat_enc @ query

            # Per-student max similarity (excluding self)
            max_sims = {}
            for idx, student_idx in enumerate(flat_idx):
                sim = float(similarities[idx])
                student_id_from_cache = ids[student_idx]
                if student_id_from_cache == student_id:
                    continue  # Skip own encodings
                if student_id_from_cache not in max_sims:
                    max_sims[student_id_from_cache] = sim
                else:
                    max_sims[student_id_from_cache] = max(max_sims[student_id_from_cache], sim)

            # Check if any match is too high (duplicate)
            for other_student_id, sim in max_sims.items():
                if sim > threshold:
                    other_student = database.get_student_by_id(other_student_id)
                    other_name = other_student.get("name", "Unknown") if other_student else "Unknown"
                    return True, f"Face too similar to existing student '{other_name}' (similarity {sim:.3f} > {threshold:.3f})"

            return False, None
        except Exception as e:
            logger.warning("Failed to check for duplicate faces: %s", e)
            return False, None  # On error, allow enrollment

    @staticmethod
    def analyze_angle_diversity(images_bgr: List[np.ndarray]) -> Tuple[bool, Optional[str], List[Optional[Dict[str, float]]]]:
        """Analyze angle diversity across multiple enrollment images.
        
        Requires >30° yaw spread across images.
        
        Returns:
            (is_diverse, error_reason, angles_list)
        """
        angles_list = []
        yaw_angles = []

        for image_bgr in images_bgr:
            angles = EnrollmentValidator.extract_face_angles(image_bgr, (0, 0, 0, 0))
            angles_list.append(angles)
            if angles is not None:
                yaw_angles.append(angles.get("yaw", 0.0))

        if len(yaw_angles) < 2:
            # Can't assess diversity with < 2 angles
            return True, None, angles_list

        yaw_spread = max(yaw_angles) - min(yaw_angles)
        if yaw_spread < EnrollmentValidator.MIN_YAW_SPREAD_DEGREES:
            return False, f"Insufficient angle diversity (yaw spread {yaw_spread:.0f}°, min {EnrollmentValidator.MIN_YAW_SPREAD_DEGREES}°)", angles_list

        return True, None, angles_list

    @staticmethod
    def validate_multi_angle_enrollment(images_bgr: List[np.ndarray], student_id: Optional[ObjectId] = None) -> Dict[str, any]:
        """Validate a full enrollment with multiple images.
        
        Returns dict:
        {
            "valid": bool,
            "error": Optional[str],
            "image_results": [{"index": int, "valid": bool, "error": Optional[str]}],
            "angle_diversity": {"valid": bool, "error": Optional[str]},
            "duplicate_detected": Optional[str],
        }
        """
        result = {
            "valid": False,
            "error": None,
            "image_results": [],
            "angle_diversity": {"valid": False, "error": None},
            "duplicate_detected": None,
        }

        # Check image count
        if len(images_bgr) < EnrollmentValidator.MIN_ENROLLMENT_IMAGES:
            result["error"] = f"Too few images ({len(images_bgr)}, min {EnrollmentValidator.MIN_ENROLLMENT_IMAGES})"
            return result

        if len(images_bgr) > EnrollmentValidator.MAX_ENROLLMENT_IMAGES:
            result["error"] = f"Too many images ({len(images_bgr)}, max {EnrollmentValidator.MAX_ENROLLMENT_IMAGES})"
            return result

        # Validate each image
        valid_count = 0
        for i, image_bgr in enumerate(images_bgr):
            is_valid, error_reason = EnrollmentValidator.validate_enrollment_image(image_bgr)
            result["image_results"].append({
                "index": i,
                "valid": is_valid,
                "error": error_reason,
            })
            if is_valid:
                valid_count += 1

        # Check if at least MIN images are valid
        if valid_count < EnrollmentValidator.MIN_ENROLLMENT_IMAGES:
            result["error"] = f"Only {valid_count}/{len(images_bgr)} images passed quality check"
            return result

        # Check angle diversity
        diversity_valid, diversity_error, angles_list = EnrollmentValidator.analyze_angle_diversity(images_bgr)
        result["angle_diversity"]["valid"] = diversity_valid
        result["angle_diversity"]["error"] = diversity_error

        if not diversity_valid:
            result["error"] = f"Insufficient angle diversity: {diversity_error}"
            return result

        # Check for duplicates (if student_id provided)
        if student_id is not None:
            for i, image_bgr in enumerate(images_bgr):
                is_dup, dup_error = EnrollmentValidator.check_duplicate_face(student_id, image_bgr)
                if is_dup:
                    result["duplicate_detected"] = dup_error
                    result["error"] = f"Duplicate face detected: {dup_error}"
                    return result

        # All checks passed
        result["valid"] = True
        return result
