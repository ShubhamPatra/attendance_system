"""Screen and printed photo detection using FFT and texture analysis.

Detects LCD screen displays and printed photos by analyzing:
1. Periodic patterns (moire effect from LCD pixels)
2. Flatness (photos lack natural texture variation)
3. Reflections (LCD backlight gloss and bright spots)

This module runs in <3ms per frame using only OpenCV and NumPy.
"""

import numpy as np
import cv2
from core.utils import setup_logging

logger = setup_logging()


def detect_screen_or_print(face_roi: np.ndarray) -> dict:
    """Detect screen displays or printed photos using multi-channel analysis.
    
    Args:
        face_roi: Face region of interest (BGR or grayscale), minimum 48x48 pixels
        
    Returns:
        dict with keys:
            - is_screen_or_print: bool, True if likely screen/photo
            - screen_score: float [0, 1], confidence of detection
            - reason: str, detection reason (moire/flat/reflection/natural)
            - periodic_score: float [0, 1], moire pattern strength
            - flatness_score: float [0, 1], texture flatness (high = flat = suspicious)
            - reflection_score: float [0, 1], bright spot intensity
    """
    h, w = face_roi.shape[:2]
    
    # Validate minimum size
    if h < 48 or w < 48:
        return {
            "is_screen_or_print": False,
            "screen_score": 0.0,
            "reason": "too_small",
            "periodic_score": 0.0,
            "flatness_score": 0.0,
            "reflection_score": 0.0,
        }
    
    try:
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.astype(np.uint8) if face_roi.dtype != np.uint8 else face_roi
        
        # Run all detection methods
        periodic_score = _detect_periodic_patterns(gray)
        flatness_score = _detect_flatness(gray)
        reflection_score = _detect_reflection(face_roi)
        
        # Combine scores with weights
        # Periodic patterns (moire) = strongest indicator of LCD
        # Flatness = strong indicator of photo/screen
        # Reflection = weak indicator (some faces have shine)
        combined_score = (
            0.45 * periodic_score +      # Moire is distinctive for LCD
            0.40 * flatness_score +      # Flatness works for photos & screens
            0.15 * reflection_score      # Reflection is secondary
        )
        
        # Determine reason (for debugging)
        max_component = max(
            ("moire", periodic_score),
            ("flat", flatness_score),
            ("reflection", reflection_score),
            key=lambda x: x[1]
        )
        reason = max_component[0]
        
        # Decision threshold: 0.60
        is_screen_or_print = combined_score >= 0.60
        
        return {
            "is_screen_or_print": is_screen_or_print,
            "screen_score": float(combined_score),
            "reason": reason,
            "periodic_score": float(periodic_score),
            "flatness_score": float(flatness_score),
            "reflection_score": float(reflection_score),
        }
        
    except Exception as e:
        logger.debug(f"Screen detection error: {e}")
        # Graceful degradation: assume natural
        return {
            "is_screen_or_print": False,
            "screen_score": 0.0,
            "reason": "error",
            "periodic_score": 0.0,
            "flatness_score": 0.0,
            "reflection_score": 0.0,
        }


def _detect_periodic_patterns(gray: np.ndarray) -> float:
    """Detect periodic patterns (moire from LCD pixels) using FFT.
    
    LCD screens have regular pixel grids that create distinctive patterns
    in frequency domain (high energy at specific frequencies corresponding
    to pixel pitch, typically 0.2-0.5mm or ~50-200 pixels/inch).
    
    Returns:
        float [0, 1], score indicating presence of periodic patterns
    """
    try:
        # Resize to manageable size for FFT (64x64 = good speed/accuracy balance)
        h, w = gray.shape
        crop_size = min(64, h, w)
        y_start = (h - crop_size) // 2
        x_start = (w - crop_size) // 2
        crop = gray[y_start:y_start+crop_size, x_start:x_start+crop_size]
        
        # Apply Gaussian blur to reduce noise while preserving patterns
        blurred = cv2.GaussianBlur(crop, (3, 3), 0)
        
        # Compute 2D FFT
        f_transform = np.fft.fft2(blurred)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Normalize magnitude spectrum
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        # Analyze frequency bands
        # DC component (center) represents average brightness
        center = crop_size // 2
        
        # Define frequency bands (in pixels from center)
        # Band 1: Very low freq (0-5px) = overall brightness
        # Band 2: Low freq (5-15px) = large features (eyes, mouth)
        # Band 3: Mid freq (15-30px) = medium features (skin texture)
        # Band 4: High freq (30+px) = fine details and periodic noise
        
        def get_annular_mask(size, r_min, r_max):
            """Create annular (ring) mask in frequency domain."""
            y, x = np.ogrid[:size, :size]
            mask = ((x - size//2)**2 + (y - size//2)**2)**0.5
            return (mask >= r_min) & (mask < r_max)
        
        # Calculate energy in each band
        band_dc = get_annular_mask(crop_size, 0, 3)
        band_low = get_annular_mask(crop_size, 3, 10)
        band_mid = get_annular_mask(crop_size, 10, 20)
        band_high = get_annular_mask(crop_size, 20, crop_size//2)
        
        energy_dc = np.sum(magnitude[band_dc])
        energy_low = np.sum(magnitude[band_low])
        energy_mid = np.sum(magnitude[band_mid])
        energy_high = np.sum(magnitude[band_high])
        
        total_energy = energy_dc + energy_low + energy_mid + energy_high + 1e-8
        
        # For screens: high mid-frequency and high-frequency energy (periodic pixel pattern)
        # For natural faces: most energy in low frequencies (large features)
        ratio_high_to_total = (energy_mid + energy_high) / total_energy
        
        # Normalize to [0, 1]
        # Natural face: ratio ~0.1-0.2
        # Screen/photo: ratio ~0.3-0.5
        periodic_score = min(1.0, max(0.0, (ratio_high_to_total - 0.15) / 0.35))
        
        return float(periodic_score)
        
    except Exception as e:
        logger.debug(f"Periodic pattern detection error: {e}")
        return 0.0


def _detect_flatness(gray: np.ndarray) -> float:
    """Detect texture flatness using Laplacian variance and entropy.
    
    Printed photos lack the 3D texture of real skin - they're inherently flat.
    This is detectable via:
    1. Laplacian variance (edge detection): photos have lower variance
    2. Color histogram entropy: photos have more concentrated color distribution
    
    Returns:
        float [0, 1], score indicating flatness (high = flat = suspicious)
    """
    try:
        # Laplacian edge detection for texture richness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Natural skin: variance ~30-100
        # Photo/screen: variance ~5-30
        # Normalize to [0, 1] where high = flat = suspicious
        # Score 0.0 at variance=50, score 1.0 at variance=5
        flatness_from_laplacian = max(0.0, min(1.0, 1.0 - (laplacian_var / 50.0)))
        
        # Histogram entropy (color distribution diversity)
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        hist = hist[hist > 0]  # Remove zero bins
        
        # Entropy: -sum(p * log(p))
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Max entropy for 256 bins = 8.0
        # Natural skin: entropy ~5.0-7.0
        # Photo: entropy ~3.0-5.0
        flatness_from_entropy = max(0.0, min(1.0, 1.0 - (entropy / 6.0)))
        
        # Combine both measures
        flatness_score = 0.6 * flatness_from_laplacian + 0.4 * flatness_from_entropy
        
        return float(flatness_score)
        
    except Exception as e:
        logger.debug(f"Flatness detection error: {e}")
        return 0.0


def _detect_reflection(face_roi: np.ndarray) -> float:
    """Detect bright reflections from LCD backlight or glossy photo.
    
    LCD screens and glossy printed photos create bright reflections.
    We detect these as over-saturated bright regions.
    
    Returns:
        float [0, 1], score indicating reflection strength
    """
    try:
        # Convert to HSV for saturation and value detection
        if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        else:
            # If grayscale, convert to BGR first
            bgr = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        
        h, s, v = cv2.split(hsv)
        
        # Detect very bright pixels (value > 240)
        bright_mask = v > 240
        bright_ratio = np.mean(bright_mask.astype(np.float32))
        
        # Detect low saturation in bright areas (screen pixels are often desaturated)
        if np.sum(bright_mask) > 0:
            mean_saturation_in_bright = np.mean(s[bright_mask].astype(np.float32))
        else:
            mean_saturation_in_bright = 100.0
        
        # Reflection score combines:
        # 1. Ratio of very bright pixels (normal face ~5%, screen/glossy ~15-30%)
        # 2. Low saturation in bright areas (screens have desaturated bright spots)
        reflection_from_brightness = min(1.0, bright_ratio / 0.10)  # Saturate at 10%
        reflection_from_desaturation = max(0.0, 1.0 - (mean_saturation_in_bright / 100.0))
        
        reflection_score = 0.7 * reflection_from_brightness + 0.3 * reflection_from_desaturation
        
        return float(reflection_score)
        
    except Exception as e:
        logger.debug(f"Reflection detection error: {e}")
        return 0.0
