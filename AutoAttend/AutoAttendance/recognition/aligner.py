from __future__ import annotations

import cv2
import numpy as np


REFERENCE_LANDMARKS = np.array(
	[
		[38.2946, 51.6963],
		[73.5318, 51.5014],
		[56.0252, 71.7366],
		[41.5493, 92.3655],
		[70.7299, 92.2041],
	],
	dtype=np.float32,
)


class FaceAligner:
	def __init__(self, output_size: tuple[int, int] = (112, 112), max_alignment_error: float = 15.0) -> None:
		self.output_size = output_size
		self.max_alignment_error = float(max_alignment_error)
		self.reference_landmarks = REFERENCE_LANDMARKS.copy()

	def align(self, frame: np.ndarray, landmarks: np.ndarray | list[list[float]] | list[tuple[float, float]]) -> np.ndarray:
		if frame is None or frame.size == 0:
			raise ValueError("Input frame is empty")

		src = np.asarray(landmarks, dtype=np.float32)
		if src.shape != (5, 2):
			raise ValueError("Landmarks must be shape (5, 2)")

		matrix, _ = cv2.estimateAffinePartial2D(src, self.reference_landmarks, method=cv2.LMEDS)
		if matrix is None:
			raise ValueError("Failed to estimate similarity transform")

		transformed = cv2.transform(src.reshape(1, 5, 2), matrix).reshape(5, 2)
		error = float(np.mean(np.linalg.norm(transformed - self.reference_landmarks, axis=1)))
		if error > self.max_alignment_error:
			raise ValueError(f"Alignment error too high: {error:.3f}")

		aligned = cv2.warpAffine(
			frame,
			matrix,
			self.output_size,
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_REFLECT_101,
		)
		return aligned
