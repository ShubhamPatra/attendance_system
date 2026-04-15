"""Verification pipeline for secure student onboarding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

import core.config as config
import core.database as core_database
from core.utils import setup_logging
from vision import anti_spoofing, pipeline, recognition
from vision.preprocessing import preprocess_face


logger = setup_logging()


@dataclass
class SampleResult:
    """Per-image verification result."""

    path: str
    face_count: int
    liveness_label: int
    liveness_confidence: float
    quality_score: float
    encoding_ok: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationResult:
    """Aggregate onboarding verification result."""

    status: str
    score: float
    reasons: list[str]
    liveness_score: float
    consistency_score: float
    quality_score: float
    duplicate_score: float
    duplicate_found: bool
    duplicate_match: dict | None
    encodings: list[np.ndarray]
    samples: list[SampleResult]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "score": self.score,
            "reasons": self.reasons,
            "liveness_score": self.liveness_score,
            "consistency_score": self.consistency_score,
            "quality_score": self.quality_score,
            "duplicate_score": self.duplicate_score,
            "duplicate_found": self.duplicate_found,
            "duplicate_match": self.duplicate_match,
            "encodings_count": len(self.encodings),
            "samples": [sample.to_dict() for sample in self.samples],
        }


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _quality_score(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, list[str]]:
    x, y, w, h = bbox
    frame_h, frame_w = frame_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return 0.0, ["Invalid face crop."]

    roi = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    area_ratio = (w * h) / max(1.0, frame_w * frame_h)

    blur_score = _clamp((blur_var / max(config.BLUR_THRESHOLD, 1.0)) * 50.0)
    brightness_mid = (config.BRIGHTNESS_THRESHOLD + config.BRIGHTNESS_MAX) / 2.0
    brightness_score = _clamp(100.0 - abs(brightness - brightness_mid) * 1.5)
    size_score = _clamp(area_ratio * 1200.0)

    reasons: list[str] = []
    if blur_var < config.BLUR_THRESHOLD:
        reasons.append("Face is too blurry.")
    if brightness < config.BRIGHTNESS_THRESHOLD:
        reasons.append("Face is too dark.")
    if brightness > config.BRIGHTNESS_MAX:
        reasons.append("Face is too bright.")
    if area_ratio < 0.03:
        reasons.append("Face is too small in frame.")

    score = round((blur_score * 0.45) + (brightness_score * 0.35) + (size_score * 0.20), 2)
    return score, reasons


def _detect_duplicate(encoding: np.ndarray, registration_number: str | None = None) -> tuple[bool, dict | None]:
    """Detect whether *encoding* belongs to an already-registered student.

    Uses **cosine similarity** (dot product of L2-normalised vectors)
    which is consistent with the ArcFace recognition pipeline.
    """
    best_match: dict | None = None
    best_similarity = -1.0

    # L2-normalise the query
    query = encoding.astype(np.float32).flatten()
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    for student_id, name, encodings in core_database.get_student_encodings():
        student_doc = core_database.get_student_by_id(student_id)
        if not student_doc:
            continue
        if registration_number and student_doc.get("registration_number") == registration_number:
            continue
        for candidate in encodings:
            # L2-normalise the candidate
            cand = candidate.astype(np.float32).flatten()
            c_norm = np.linalg.norm(cand)
            if c_norm > 0:
                cand = cand / c_norm
            similarity = float(np.dot(cand, query))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    "student_id": str(student_id),
                    "name": name,
                    "registration_number": student_doc.get("registration_number"),
                    "similarity": round(similarity, 4),
                }

    if best_match is None:
        return False, None

    # Duplicate if similarity exceeds threshold
    is_dup = best_similarity >= config.RECOGNITION_THRESHOLD
    return is_dup, best_match if is_dup else None


def _consistency_score(encodings: list[np.ndarray]) -> float:
    """Score how consistent a set of face encodings are.

    Uses **cosine similarity** between all pairs.  Higher inter-pair
    similarity → higher consistency score.
    """
    if len(encodings) < 2:
        return 100.0

    similarities: list[float] = []
    # L2-normalise all
    normed = []
    for e in encodings:
        v = e.astype(np.float32).flatten()
        n = np.linalg.norm(v)
        normed.append(v / n if n > 0 else v)

    for idx, enc in enumerate(normed):
        for other in normed[idx + 1:]:
            similarities.append(float(np.dot(enc, other)))
    if not similarities:
        return 100.0

    mean_sim = float(np.mean(similarities))
    # Map cosine similarity [0, 1] → quality score [0, 100]
    # A mean similarity of 0.5 maps to ~50, 0.7 → ~85, 0.9 → ~98
    return _clamp(mean_sim * 110.0 - 10.0)


def evaluate_student_samples(
    sample_paths: list[str],
    registration_number: str | None = None,
) -> VerificationResult:
    """Evaluate a set of webcam captures for student onboarding."""
    reasons: list[str] = []
    sample_results: list[SampleResult] = []
    encodings: list[np.ndarray] = []
    liveness_scores: list[float] = []
    quality_scores: list[float] = []

    if len(sample_paths) < config.STUDENT_MIN_CAPTURE_IMAGES:
        raise ValueError(
            f"At least {config.STUDENT_MIN_CAPTURE_IMAGES} capture images are required."
        )
    if len(sample_paths) > config.STUDENT_MAX_CAPTURE_IMAGES:
        raise ValueError(
            f"No more than {config.STUDENT_MAX_CAPTURE_IMAGES} capture images are allowed."
        )

    for path in sample_paths:
        image = cv2.imread(path)
        if image is None:
            reasons.append(f"{Path(path).name}: unable to read image.")
            sample_results.append(SampleResult(path=path, face_count=0, liveness_label=-1, liveness_confidence=0.0, quality_score=0.0, encoding_ok=False, reason="Unreadable image."))
            continue

        faces = pipeline.detect_faces_yunet(image)
        if len(faces) != 1:
            reason = "No face detected." if not faces else "Multiple faces detected."
            reasons.append(f"{Path(path).name}: {reason}")
            sample_results.append(SampleResult(path=path, face_count=len(faces), liveness_label=-1, liveness_confidence=0.0, quality_score=0.0, encoding_ok=False, reason=reason))
            continue

        bbox = faces[0]
        quality_score, quality_reasons = _quality_score(image, bbox)
        if quality_reasons:
            reasons.extend(f"{Path(path).name}: {reason}" for reason in quality_reasons)

        x, y, w, h = bbox
        # Run liveness on full frame (not tight ROI) to avoid false spoof detection on side poses.
        label, confidence = anti_spoofing.check_liveness(image)
        
        # Handle liveness labels with appropriate scoring:
        # label 1  = real (accept fully)
        # label 0  = spoof/no-face (strict reject)
        # label -1 = model/internal error (strict reject)
        # label 2  = other_attack (soft penalty with confidence threshold)
        if label == 1:
            liveness_score = confidence * 100.0
        elif label == 0:
            liveness_score = 0.0
            reasons.append(f"{Path(path).name}: spoof or no-face detected by liveness model.")
        elif label == -1:
            liveness_score = 0.0
            reasons.append(
                f"{Path(path).name}: liveness model error; please recapture in better lighting."
            )
        else:  # label == 2 (other_attack): pose, lighting, or model uncertainty
            # Soft penalty: use 60% of confidence as baseline, minimum 20 to avoid total rejection.
            liveness_score = max(20.0, confidence * 100.0 * 0.6)
            reasons.append(f"{Path(path).name}: possible pose or lighting issue (detected as {label}; confidence {confidence:.2f}).")
        
        liveness_scores.append(liveness_score)

        # Try to extract 5-point landmarks for ArcFace alignment
        landmarks_5 = None
        if config.EMBEDDING_BACKEND == "arcface":
            try:
                from vision.face_engine import get_arcface_backend
                af = get_arcface_backend()
                af_faces = af.get_faces(preprocess_face(image))
                if af_faces and hasattr(af_faces[0], 'kps') and af_faces[0].kps is not None:
                    landmarks_5 = af_faces[0].kps
            except Exception as exc:
                logger.debug("Could not extract InsightFace landmarks for student capture: %s", exc)

        encoding, encoding_reason = recognition.encode_face_with_reason(image, bbox, landmarks_5)
        encoding_ok = encoding is not None
        if encoding_ok:
            encodings.append(encoding)
        else:
            reasons.append(f"{Path(path).name}: {encoding_reason}")

        quality_scores.append(quality_score)
        sample_results.append(
            SampleResult(
                path=path,
                face_count=len(faces),
                liveness_label=label,
                liveness_confidence=round(confidence, 4),
                quality_score=round(quality_score, 2),
                encoding_ok=encoding_ok,
                reason=encoding_reason or ("; ".join(quality_reasons) if quality_reasons else ""),
            )
        )

    if not encodings:
        return VerificationResult(
            status="rejected",
            score=0.0,
            reasons=reasons or ["No valid face encodings could be generated."],
            liveness_score=0.0,
            consistency_score=0.0,
            quality_score=0.0,
            duplicate_score=0.0,
            duplicate_found=False,
            duplicate_match=None,
            encodings=[],
            samples=sample_results,
        )

    liveness_score = round(float(np.mean(liveness_scores)) if liveness_scores else 0.0, 2)
    consistency_score = round(_consistency_score(encodings), 2)
    quality_score = round(float(np.mean(quality_scores)) if quality_scores else 0.0, 2)
    duplicate_found = False
    duplicate_match = None
    for encoding in encodings:
        duplicate_found, duplicate_match = _detect_duplicate(encoding, registration_number)
        if duplicate_found:
            break
    duplicate_score = 0.0 if duplicate_found else 100.0

    final_score = round(
        (liveness_score * 0.40)
        + (consistency_score * 0.25)
        + (quality_score * 0.20)
        + (duplicate_score * 0.15),
        2,
    )

    if duplicate_found:
        reasons.append("A matching approved face already exists in the database.")

    if final_score >= config.STUDENT_AUTO_APPROVE_SCORE and not duplicate_found and not any("spoof" in reason.lower() for reason in reasons):
        status = "approved"
    elif final_score >= config.STUDENT_PENDING_SCORE:
        status = "pending"
    else:
        status = "rejected"

    return VerificationResult(
        status=status,
        score=final_score,
        reasons=reasons,
        liveness_score=liveness_score,
        consistency_score=consistency_score,
        quality_score=quality_score,
        duplicate_score=duplicate_score,
        duplicate_found=duplicate_found,
        duplicate_match=duplicate_match,
        encodings=encodings,
        samples=sample_results,
    )
