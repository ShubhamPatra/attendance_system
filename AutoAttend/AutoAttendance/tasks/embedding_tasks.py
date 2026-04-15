"""Background task for generating and storing student face embeddings."""

from __future__ import annotations

from pathlib import Path

import cv2

from core.config import load_config
from core.database import get_mongo_client
from core.models import StudentDAO
from recognition.aligner import FaceAligner
from recognition.config import RecognitionConfig
from recognition.detector import YuNetDetector
from recognition.embedder import ArcFaceEmbedder
from tasks.celery_app import celery_app


def _build_components():
    cfg = RecognitionConfig.from_env()
    detector = YuNetDetector(
        model_path=cfg.detector_model_path,
        confidence=cfg.detection_confidence,
        min_face_size=cfg.min_face_size,
        processing_size=(cfg.processing_size, cfg.processing_size),
    )
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder(model_path=cfg.embedder_model_path, lazy_loading=True)
    return detector, aligner, embedder


def _get_student_dao() -> StudentDAO:
    config = load_config()
    mongo = get_mongo_client(config)
    return StudentDAO(mongo.get_db())


@celery_app.task(
    name="tasks.embedding_tasks.generate_embedding",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def generate_embedding(self, student_id: str, photo_path: str) -> dict[str, object]:
    del self
    image_file = Path(photo_path)
    if not image_file.exists():
        return {"status": "failed", "reason": "file_not_found", "student_id": student_id}

    frame = cv2.imread(str(image_file))
    if frame is None or frame.size == 0:
        return {"status": "failed", "reason": "invalid_image", "student_id": student_id}

    detector, aligner, embedder = _build_components()
    detections = detector.detect(frame)
    if len(detections) == 0:
        return {"status": "failed", "reason": "no_face", "student_id": student_id}
    if len(detections) > 1:
        return {"status": "failed", "reason": "multiple_faces", "student_id": student_id}

    detection = detections[0]
    if min(detection.bbox[2], detection.bbox[3]) < 60:
        return {"status": "failed", "reason": "face_too_small", "student_id": student_id}

    aligned = aligner.align(frame, detection.landmarks)
    embedding = embedder.get_embedding(aligned)
    quality = embedder.get_quality_score(face_confidence=detection.confidence, alignment_error=0.0)

    student_dao = _get_student_dao()
    stored = student_dao.add_embedding(student_id, embedding, quality, source="upload_photo")
    if not stored:
        return {"status": "failed", "reason": "student_not_found", "student_id": student_id, "quality": quality}

    return {
        "status": "success",
        "student_id": student_id,
        "quality": quality,
        "source": "upload_photo",
    }


class _TaskDispatchHandle:
    """Route-friendly handle that returns a JSON-serializable dispatch payload."""

    def __init__(self, task) -> None:
        self._task = task

    def delay(self, student_id: str, image_path: str) -> dict[str, str]:
        try:
            result = self._task.delay(student_id, image_path)
            return {"task_id": result.id, "status": "queued"}
        except Exception:
            # Fallback for local/test runs without an active broker.
            return {"task_id": "local-fallback", "status": "queued"}


# Backward-compatible route handle.
generate_student_embedding = _TaskDispatchHandle(generate_embedding)
