from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from recognition.aligner import FaceAligner
from recognition.config import RecognitionConfig
from recognition.detector import DetectionResult, YuNetDetector
from recognition.embedder import ArcFaceEmbedder
from recognition.matcher import FaceMatcher
from recognition.tracker import CSRTTracker
from core.profiling import PipelineProfiler, RollingFPSCounter


@dataclass(slots=True)
class SmoothedIdentity:
	student_id: str | None
	confidence: float
	count: int


class MultiFrameSmoother:
	def __init__(self, window_size: int = 5, majority_count: int = 3) -> None:
		self.window_size = max(1, int(window_size))
		self.majority_count = max(1, int(majority_count))
		self._buffer: deque[tuple[str | None, float]] = deque(maxlen=self.window_size)

	def add(self, student_id: str | None, confidence: float) -> SmoothedIdentity:
		self._buffer.append((student_id, float(confidence)))
		votes = [item for item in self._buffer if item[0] is not None]
		if not votes:
			return SmoothedIdentity(student_id=None, confidence=0.0, count=0)

		counts = Counter([sid for sid, _ in votes])
		winner, count = counts.most_common(1)[0]
		if count < self.majority_count:
			return SmoothedIdentity(student_id=None, confidence=0.0, count=count)

		winner_scores = [score for sid, score in votes if sid == winner]
		avg_conf = float(sum(winner_scores) / max(1, len(winner_scores)))
		return SmoothedIdentity(student_id=winner, confidence=avg_conf, count=count)

	def clear(self) -> None:
		self._buffer.clear()


class RecognitionPipeline:
	def __init__(
		self,
		config: RecognitionConfig,
		detector: YuNetDetector | None = None,
		tracker: CSRTTracker | None = None,
		aligner: FaceAligner | None = None,
		embedder: ArcFaceEmbedder | None = None,
		matcher: FaceMatcher | None = None,
		student_dao: Any | None = None,
		profiler: PipelineProfiler | None = None,
		adaptive_detection: bool = True,
	) -> None:
		self.config = config
		self.detector = detector or YuNetDetector(
			model_path=self.config.detector_model_path,
			confidence=self.config.detection_confidence,
			nms_threshold=self.config.nms_threshold,
			top_k=self.config.top_k,
			min_face_size=self.config.min_face_size,
			processing_size=(self.config.processing_size, self.config.processing_size),
		)
		self.tracker = tracker or CSRTTracker(reset_interval=30)
		self.aligner = aligner or FaceAligner()
		self.embedder = embedder or ArcFaceEmbedder(model_path=self.config.embedder_model_path, lazy_loading=True)
		self.matcher = matcher or FaceMatcher()
		self.student_dao = student_dao

		self.smoother = MultiFrameSmoother(
			window_size=self.config.smoother_window,
			majority_count=self.config.smoother_majority,
		)

		self.frame_count = 0
		self.fps = 0.0
		self._last_frame_ts: float | None = None
		self._last_landmarks: np.ndarray | None = None
		self._last_detection_bbox: tuple[int, int, int, int] | None = None
		self._active_track_id = "main"
		self.active_course_id: str | None = None
		self.profiler = profiler or PipelineProfiler()
		self._rolling_fps = RollingFPSCounter(window_size=30)
		self.adaptive_detection = bool(adaptive_detection)

	def load_gallery(self, course_id: str) -> int:
		if self.student_dao is None:
			raise RuntimeError("student_dao is required to load gallery")

		cache_key = f"course:{course_id}"

		def _loader() -> dict[str, np.ndarray]:
			rows = self.student_dao.get_roster_embeddings(course_id)
			gallery: dict[str, list[np.ndarray]] = {}
			for row in rows:
				sid = row.get("student_id")
				emb = row.get("embedding")
				if not sid or emb is None:
					continue
				gallery.setdefault(sid, []).append(np.asarray(emb, dtype=np.float32))
			return {sid: np.stack(vectors, axis=0) for sid, vectors in gallery.items() if vectors}

		gallery = self.matcher.cache.prewarm(cache_key, _loader)
		self.active_course_id = course_id
		return int(sum(v.shape[0] for v in gallery.values()))

	def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
		self.frame_count += 1
		now = perf_counter()
		if self._last_frame_ts is not None:
			delta = now - self._last_frame_ts
			if delta > 0:
				self._rolling_fps.add_frame_time(delta)
				self.fps = self._rolling_fps.value()
		self._last_frame_ts = now

		scaled_frame, x_scale, y_scale = self._scale_frame(frame)

		tracker_state = self._get_tracker_state()
		detect_interval = self._get_dynamic_interval(tracker_state) if self.adaptive_detection else max(1, self.config.detect_interval)
		use_detection = self.frame_count == 1 or (self.frame_count % max(1, detect_interval) == 0)

		current_bbox = tracker_state.get("bbox")
		if self.adaptive_detection and use_detection and isinstance(current_bbox, tuple) and self._last_detection_bbox is not None:
			movement = self._bbox_center_movement_px(self._last_detection_bbox, current_bbox)
			if movement < 3.0 and tracker_state.get("status") == "high":
				use_detection = False

		if use_detection:
			return self._process_detection_frame(frame, scaled_frame, x_scale, y_scale)
		return self._process_tracking_frame(frame)

	def _process_detection_frame(
		self,
		frame: np.ndarray,
		scaled_frame: np.ndarray,
		x_scale: float,
		y_scale: float,
	) -> dict[str, Any]:
		with self.profiler.stage("detect"):
			detections = self.detector.detect(scaled_frame)
		detections = self._remap_coords(detections, x_scale=x_scale, y_scale=y_scale)
		if not detections:
			self.smoother.add(None, 0.0)
			return self._result_payload(stage="detection", status="no_face")

		best = max(detections, key=lambda item: item.confidence)
		with self.profiler.stage("track_init"):
			self.tracker.init_track(frame, best.bbox, track_id=self._active_track_id)
		self._last_landmarks = np.asarray(best.landmarks, dtype=np.float32)
		self._last_detection_bbox = best.bbox
		return self._recognize(frame, stage="detection", detection=best)

	def _process_tracking_frame(self, frame: np.ndarray) -> dict[str, Any]:
		with self.profiler.stage("track"):
			ok, bbox = self.tracker.update(frame, track_id=self._active_track_id)
		if not ok or bbox is None:
			self.smoother.add(None, 0.0)
			return self._result_payload(stage="tracking", status="lost")

		if (self.frame_count % max(1, self.config.embedding_interval)) != 0:
			return self._result_payload(stage="tracking", status="tracking", bbox=bbox)

		landmarks_source = self._last_landmarks if self._last_landmarks is not None else np.zeros((5, 2), dtype=np.int32)
		detection_like = DetectionResult(
			bbox=bbox,
			landmarks=[tuple(map(int, item)) for item in landmarks_source],
			confidence=1.0,
		)
		return self._recognize(frame, stage="tracking", detection=detection_like)

	def _recognize(self, frame: np.ndarray, stage: str, detection: DetectionResult) -> dict[str, Any]:
		try:
			landmarks = np.asarray(detection.landmarks, dtype=np.float32)
			with self.profiler.stage("align"):
				aligned = self.aligner.align(frame, landmarks)
			with self.profiler.stage("embed"):
				embedding = self.embedder.get_embedding(aligned)
		except Exception as exc:
			self.smoother.add(None, 0.0)
			return self._result_payload(stage=stage, status="align_or_embed_error", error=str(exc), bbox=detection.bbox)

		match_student, match_similarity, stable = self._match_embedding(embedding)
		status = "matched" if stable.student_id else "candidate"
		return self._result_payload(
			stage=stage,
			status=status,
			bbox=detection.bbox,
			candidate_student_id=match_student,
			candidate_similarity=match_similarity,
			student_id=stable.student_id,
			confidence=stable.confidence,
			vote_count=stable.count,
			matched=stable.student_id is not None,
		)

	def _match_embedding(self, embedding: np.ndarray) -> tuple[str | None, float, SmoothedIdentity]:
		if not self.active_course_id:
			stable = self.smoother.add(None, 0.0)
			return None, 0.0, stable

		cache_key = f"course:{self.active_course_id}"
		gallery = self.matcher.cache.get(cache_key) or {}
		if not gallery:
			stable = self.smoother.add(None, 0.0)
			return None, 0.0, stable

		results = self.matcher.match_against_gallery(embedding, gallery, top_k=1)
		if not results:
			stable = self.smoother.add(None, 0.0)
			return None, 0.0, stable

		top = results[0]
		if not self.matcher.is_match(top.similarity, mode=self.config.match_mode):
			stable = self.smoother.add(None, 0.0)
			return top.student_id, top.similarity, stable

		stable = self.smoother.add(top.student_id, top.similarity)
		return top.student_id, top.similarity, stable

	def _result_payload(self, stage: str, status: str, **extras: Any) -> dict[str, Any]:
		payload: dict[str, Any] = {
			"frame_count": self.frame_count,
			"fps": float(self.fps),
			"stage": stage,
			"status": status,
		}
		payload.update(extras)
		return payload

	def _scale_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
		target_w = int(max(1, self.config.frame_scale_width))
		target_h = int(max(1, self.config.frame_scale_height))
		h, w = frame.shape[:2]
		if (w, h) == (target_w, target_h):
			return frame, 1.0, 1.0

		scaled = np.asarray(frame)
		if (w, h) != (target_w, target_h):
			import cv2

			scaled = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

		x_scale = w / float(target_w)
		y_scale = h / float(target_h)
		return scaled, x_scale, y_scale

	@staticmethod
	def _remap_coords(
		detections: list[DetectionResult],
		x_scale: float,
		y_scale: float,
	) -> list[DetectionResult]:
		if not detections:
			return []
		if abs(x_scale - 1.0) < 1e-9 and abs(y_scale - 1.0) < 1e-9:
			return detections

		remapped: list[DetectionResult] = []
		for det in detections:
			x, y, w, h = det.bbox
			bbox = (
				int(round(x * x_scale)),
				int(round(y * y_scale)),
				int(round(w * x_scale)),
				int(round(h * y_scale)),
			)
			landmarks = [
				(int(round(lx * x_scale)), int(round(ly * y_scale)))
				for lx, ly in det.landmarks
			]
			remapped.append(DetectionResult(bbox=bbox, landmarks=landmarks, confidence=det.confidence))
		return remapped

	def _get_tracker_state(self) -> dict[str, Any]:
		if hasattr(self.tracker, "get_state"):
			return dict(self.tracker.get_state(track_id=self._active_track_id))
		return {"status": "uncertain", "confidence": 0.5, "bbox": None, "failure_count": 0}

	@staticmethod
	def _get_dynamic_interval(tracker_state: dict[str, Any]) -> int:
		status = str(tracker_state.get("status", "uncertain")).lower()
		if status == "high":
			return 10
		if status == "lost":
			return 1
		return 3

	@staticmethod
	def _bbox_center_movement_px(bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]) -> float:
		x1, y1, w1, h1 = bbox1
		x2, y2, w2, h2 = bbox2
		cx1 = x1 + (w1 / 2.0)
		cy1 = y1 + (h1 / 2.0)
		cx2 = x2 + (w2 / 2.0)
		cy2 = y2 + (h2 / 2.0)
		return float(np.hypot(cx2 - cx1, cy2 - cy1))
