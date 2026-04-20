"""
Texture analysis module for advanced anti-spoofing.

Uses Local Binary Pattern (LBP) to detect flat surfaces and print attacks.
Flat surfaces (screens, printed photos) have high uniformity and low entropy,
whereas natural faces have diverse texture patterns.
"""

import numpy as np
import cv2
from typing import Tuple

from core.utils import setup_logging

logger = setup_logging()


class TextureAnalyzer:
    """Analyzes face texture for flatness/uniformity detection."""
    
    def __init__(self, radius: int = 1, points: int = 8):
        """
        Initialize texture analyzer.
        
        Args:
            radius: LBP radius for neighborhood analysis
            points: Number of sampling points for LBP
        """
        self.radius = radius
        self.points = points
    
    @staticmethod
    def _compute_lbp(image: np.ndarray, radius: int = 1, points: int = 8) -> np.ndarray:
        """
        Compute Local Binary Pattern.
        
        Args:
            image: Grayscale face ROI (H x W)
            radius: LBP neighborhood radius
            points: Number of sampling points
            
        Returns:
            LBP histogram (normalized) of length 256
        """
        h, w = image.shape
        
        # Pad image for boundary handling
        pad = radius
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        
        lbp_map = np.zeros((h, w), dtype=np.uint8)
        
        # Compute LBP for each pixel
        for y in range(h):
            for x in range(w):
                center = padded[y + pad, x + pad]
                lbp_code = 0
                
                # Sample points around the center pixel
                for k in range(points):
                    angle = 2.0 * np.pi * k / points
                    px = x + pad + radius * np.cos(angle)
                    py = y + pad + radius * np.sin(angle)
                    
                    # Bilinear interpolation
                    x1, y1 = int(np.floor(px)), int(np.floor(py))
                    x2, y2 = x1 + 1, y1 + 1
                    alpha = px - x1
                    beta = py - y1
                    
                    # Clamp to valid range
                    x1 = max(0, min(x1, padded.shape[1] - 1))
                    x2 = max(0, min(x2, padded.shape[1] - 1))
                    y1 = max(0, min(y1, padded.shape[0] - 1))
                    y2 = max(0, min(y2, padded.shape[0] - 1))
                    
                    # Bilinear interpolation
                    interp = (
                        (1 - alpha) * (1 - beta) * padded[y1, x1] +
                        alpha * (1 - beta) * padded[y1, x2] +
                        (1 - alpha) * beta * padded[y2, x1] +
                        alpha * beta * padded[y2, x2]
                    )
                    
                    if interp >= center:
                        lbp_code |= (1 << k)
                
                lbp_map[y, x] = lbp_code
        
        # Compute histogram
        hist, _ = np.histogram(lbp_map, bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        
        # Normalize
        if hist.sum() > 0:
            hist /= hist.sum()
        
        return hist
    
    @staticmethod
    def _compute_entropy(hist: np.ndarray) -> float:
        """
        Compute Shannon entropy of LBP histogram.
        
        High entropy = diverse texture (natural face)
        Low entropy = uniform texture (flat surface)
        
        Args:
            hist: Normalized histogram
            
        Returns:
            Entropy in [0, 1] (normalized by log2(256))
        """
        # Remove zero bins to avoid log(0)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        # Normalize by max entropy (log2(256))
        max_entropy = np.log2(256)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        return float(normalized_entropy)
    
    def analyze_texture(
        self, 
        face_roi: np.ndarray,
        roi_bbox: Tuple[int, int, int, int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Analyze texture flatness of face ROI.
        
        Args:
            face_roi: BGR image of face region (H x W x 3)
            roi_bbox: Bounding box (x1, y1, x2, y2) - unused, for consistency
            
        Returns:
            (lbp_histogram, flatness_score)
            - lbp_histogram: (256,) array of LBP histogram
            - flatness_score: [0.0-1.0] where:
              * 1.0 = very flat (likely screen/print)
              * 0.0 = very textured (likely natural face)
        """
        try:
            # Validate input
            if face_roi is None or face_roi.size == 0:
                return np.zeros(256, dtype=np.float32), 0.5  # Default middle score
            
            if len(face_roi.shape) != 3 or face_roi.shape[2] != 3:
                raise ValueError(f"Expected BGR image (H x W x 3), got shape {face_roi.shape}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Compute LBP histogram
            lbp_hist = self._compute_lbp(gray, self.radius, self.points)
            
            # Compute entropy
            entropy = self._compute_entropy(lbp_hist)
            
            # Flatness = 1 - entropy (inverse relationship)
            # Natural faces have high entropy (diverse patterns)
            # Flat surfaces have low entropy (uniform patterns)
            flatness_score = 1.0 - entropy
            
            # Also consider image uniformity using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = laplacian.var()
            
            # Normalize Laplacian variance to [0, 1]
            # High variance = detailed texture, low variance = flat
            laplacian_norm = min(1.0, laplacian_var / 100.0)  # 100 is empirical threshold
            texture_score = laplacian_norm
            
            # Combined flatness: average LBP flatness and inverse Laplacian
            combined_flatness = (flatness_score + (1.0 - texture_score)) / 2.0
            combined_flatness = float(combined_flatness)
            
            return lbp_hist, combined_flatness
            
        except Exception as exc:
            logger.warning(f"Texture analysis failed: {exc}")
            # Return neutral scores on failure
            return np.zeros(256, dtype=np.float32), 0.5
    
    def get_flatness_classification(self, flatness_score: float, threshold: float = 0.7) -> str:
        """
        Classify flatness level.
        
        Args:
            flatness_score: Score from analyze_texture()
            threshold: Classification threshold
            
        Returns:
            "flat" (likely spoof) or "textured" (likely natural)
        """
        return "flat" if flatness_score >= threshold else "textured"
