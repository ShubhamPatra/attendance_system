"""
System Configuration Guide - AutoAttendance System Upgrade

This document provides comprehensive reference for all configuration parameters
added in Phases 1-7 of the system upgrade.

All parameters are defined in core/config.py and can be overridden via
environment variables.

Last Updated: April 2026
Phase Coverage: 1-7
"""

# ============================================================================
# CONFIGURATION REFERENCE
# ============================================================================

## PHASE 1: RELIABILITY HARDENING
## Goal: Prevent frame stalls, handle high load, ensure graceful degradation

CAMERA_FRAME_TIMEOUT_SECONDS = 5.0
    Type: float
    Range: 1.0 - 30.0 seconds
    Default: 5.0
    Purpose: Maximum time to wait for new frame before force reconnect
    Impact: Too low = excessive reconnects; Too high = stalls on camera disconnect
    Recommendation: 5.0 for typical cameras, 10.0 for high-latency/network cameras
    Environment: CAMERA_FRAME_TIMEOUT_SECONDS

FRAME_QUEUE_MAX_DEPTH = 50
    Type: int
    Range: 10 - 500 frames
    Default: 50
    Purpose: Maximum event queue depth before frame dropping begins
    Impact: Too low = dropped frames during burst; Too high = memory overhead
    Recommendation: 50 for 1-2 cameras, 100+ for multi-camera setups
    Environment: FRAME_QUEUE_MAX_DEPTH

FRAME_DROP_ENABLED = True
    Type: bool
    Default: True
    Purpose: Enable/disable frame dropping under load
    Impact: When disabled, queue will fill and block pipeline
    Recommendation: Always True for production
    Environment: FRAME_DROP_ENABLED

GRACEFUL_DEGRADATION_ENABLED = True
    Type: bool
    Default: True
    Purpose: Enable/disable adaptive throttling under CPU/memory pressure
    Impact: Disabled = potentially sluggish UI under load; Enabled = reduced accuracy
    Recommendation: True for production, False for development/testing
    Environment: GRACEFUL_DEGRADATION_ENABLED

CPU_THRESHOLD = 80
    Type: int (percentage)
    Range: 50 - 95
    Default: 80
    Purpose: CPU usage % that triggers graceful degradation
    Impact: Too low = degradation during normal load; Too high = unresponsive system
    Recommendation: 80 for balanced experience, 75 for more aggressive tuning
    Environment: CPU_THRESHOLD

MEMORY_THRESHOLD = 85
    Type: int (percentage)
    Range: 60 - 95
    Default: 85
    Purpose: Memory usage % that triggers graceful degradation
    Impact: Similar to CPU threshold; affects memory-constrained environments
    Recommendation: 85 for most systems, lower on memory-limited devices
    Environment: MEMORY_THRESHOLD

GRACEFUL_DEGRADATION_DETECTION_INTERVAL_MAX = 15
    Type: int (seconds)
    Range: 5 - 60
    Default: 15
    Purpose: Maximum detection interval when under load
    Impact: Higher = faster processing but more CPU; Lower = smoother but less responsive
    Recommendation: 15 for balanced, 10 for performance-critical
    Environment: GRACEFUL_DEGRADATION_DETECTION_INTERVAL_MAX

GRACEFUL_DEGRADATION_DISABLE_ANTISPOOF = True
    Type: bool
    Default: True
    Purpose: Disable anti-spoof when under extreme CPU load
    Impact: Reduces security temporarily; saves ~30% CPU
    Recommendation: True for production (spoof checks resumed when load drops)
    Environment: GRACEFUL_DEGRADATION_DISABLE_ANTISPOOF

---

## PHASE 2: METRICS & OBSERVABILITY
## Goal: Provide real-time visibility into system performance

SLOW_FRAME_THRESHOLD_MS = 100.0
    Type: float (milliseconds)
    Range: 50 - 500 ms
    Default: 100.0
    Purpose: Threshold for logging slow frame processing
    Impact: Too low = excessive logging; Too high = misses bottlenecks
    Recommendation: 100 for 30 FPS (33ms per frame), 33 for 60 FPS targets
    Environment: SLOW_FRAME_THRESHOLD_MS

---

## PHASE 3: RECOGNITION CONFIDENCE & FALSE POSITIVE CONTROL
## Goal: Reduce false positives, improve match certainty

RECOGNITION_TOP2_SIMILARITY_MARGIN = 0.05
    Type: float (0-1 similarity score)
    Range: 0.01 - 0.20
    Default: 0.05
    Purpose: Minimum gap between top-2 candidates to accept match
    Impact: Higher = rejects more ambiguous matches (fewer FP); Lower = accepts more (more TP)
    Recommendation: 0.05 for balanced, 0.10 for high-security environments
    Environment: RECOGNITION_TOP2_SIMILARITY_MARGIN

COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT = 0.5
    Type: float (0-1)
    Range: 0.0 - 1.0
    Default: 0.5
    Purpose: Weight of recognition score in composed confidence
    Impact: Higher = recognition dominates decision
    Recommendation: 0.5 (balanced with liveness/consistency)
    Note: Must sum with other weights > 0
    Environment: COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT

COMPOSED_CONFIDENCE_LIVENESS_WEIGHT = 0.3
    Type: float (0-1)
    Range: 0.0 - 1.0
    Default: 0.3
    Purpose: Weight of liveness score in composed confidence
    Impact: Higher = liveness checks more important
    Recommendation: 0.3 (important for spoof detection)
    Note: Must sum with other weights > 0
    Environment: COMPOSED_CONFIDENCE_LIVENESS_WEIGHT

COMPOSED_CONFIDENCE_CONSISTENCY_WEIGHT = 0.2
    Type: float (0-1)
    Range: 0.0 - 1.0
    Default: 0.2
    Purpose: Weight of consistency score in composed confidence
    Impact: Higher = temporal stability more important
    Recommendation: 0.2 (frame consistency check)
    Note: Must sum with other weights > 0
    Environment: COMPOSED_CONFIDENCE_CONSISTENCY_WEIGHT

Formula:
    composed_confidence = (
        (recognition_weight * recog_score) +
        (liveness_weight * liveness_score) +
        (consistency_weight * consistency_score)
    ) / total_weight

---

## PHASE 4: ENROLLMENT QUALITY CONTROL
## Goal: Enforce high-quality, diverse facial data during enrollment

EnrollmentValidator.MIN_BLUR_SHARPNESS = 100.0
    Type: float (Laplacian variance)
    Range: 50.0 - 300.0
    Default: 100.0
    Purpose: Minimum image sharpness (Laplacian variance)
    Impact: Higher = rejects blurry images; Lower = allows slight blur
    Recommendation: 100 for controlled lighting, 80 for outdoor/variable lighting
    Adjustment: cv2.Laplacian(roi, cv2.CV_64F).var() > threshold

EnrollmentValidator.MIN_FACE_SIZE_PIXELS = 80
    Type: int (pixels)
    Range: 40 - 200
    Default: 80
    Purpose: Minimum face size in frame (width or height)
    Impact: Higher = requires closer faces; Lower = accepts distant faces
    Recommendation: 80 for good quality, 120 for high-accuracy requirements
    Note: Face should be at least 2% of frame area

EnrollmentValidator.MIN_BRIGHTNESS = 40
    Type: int (0-255)
    Range: 10 - 100
    Default: 40
    Purpose: Minimum average brightness in face ROI
    Impact: Higher = rejects dark images; Lower = allows low-light
    Recommendation: 40 for typical lighting, 60 for well-lit environments
    Adjustment: np.mean(gray_roi) > threshold

EnrollmentValidator.MAX_BRIGHTNESS = 250
    Type: int (0-255)
    Range: 200 - 255
    Default: 250
    Purpose: Maximum average brightness in face ROI
    Impact: Higher = allows overexposed; Lower = rejects bright images
    Recommendation: 250 for typical lighting, 220 for controlled exposure
    Adjustment: np.mean(gray_roi) < threshold

EnrollmentValidator.MAX_FACE_ANGLE_DEGREES = 30.0
    Type: float (degrees)
    Range: 15.0 - 60.0
    Default: 30.0
    Purpose: Maximum head rotation angle (yaw/pitch/roll)
    Impact: Higher = allows more extreme angles; Lower = enforces frontal
    Recommendation: 30 for balanced, 15 for frontal-only requirements
    Note: Extracted from face landmarks via InsightFace

EnrollmentValidator.MIN_ENROLLMENT_IMAGES = 3
    Type: int
    Range: 1 - 10
    Default: 3
    Purpose: Minimum images required for enrollment
    Impact: Higher = better coverage, slower enrollment; Lower = faster enrollment
    Recommendation: 3 for minimum, 5 for high-accuracy systems
    Environment: Via EnrollmentValidator.MIN_ENROLLMENT_IMAGES

EnrollmentValidator.MAX_ENROLLMENT_IMAGES = 5
    Type: int
    Range: 1 - 20
    Default: 5
    Purpose: Maximum images allowed for enrollment
    Impact: Prevents excessive uploads
    Recommendation: 5 for typical, 10 for high-accuracy systems
    Environment: Via EnrollmentValidator.MAX_ENROLLMENT_IMAGES

EnrollmentValidator.MIN_YAW_SPREAD_DEGREES = 30.0
    Type: float (degrees)
    Range: 15.0 - 90.0
    Default: 30.0
    Purpose: Minimum yaw angle spread across enrollment images
    Impact: Higher = requires more diverse angles; Lower = allows similar angles
    Recommendation: 30 for good coverage, 45 for extreme diversity
    Note: Computed as max(yaw) - min(yaw) across images

---

## PHASE 5: ANTI-CHEAT LOGGING & ANALYTICS
## Goal: Detect and log suspicious activities

Security Event Logging:
    - Event types: spoof_attempt, multi_identity, failed_match, liveness_uncertain,
                   repeated_spoof, abnormal_pattern, enrollment_fraud, duplicate_attendance
    - Severity levels: low, medium, high, critical
    - Retention: Determined by MongoDB TTL index (default: indefinite)
    - Buffer size: 100 events or 5 seconds (auto-flush)
    - Performance: <5ms latency (buffered)

Anomaly Detection Thresholds:

REPEATED_SPOOF_ATTEMPT_THRESHOLD = 3
    Type: int (number of attempts)
    Default: 3
    Purpose: Number of spoof attempts before "repeated_spoof" alert
    Impact: Higher = fewer alerts; Lower = more sensitive
    Recommendation: 3 for balanced detection

REPEATED_SPOOF_TIME_WINDOW_MINUTES = 10
    Type: int (minutes)
    Default: 10
    Purpose: Time window for counting repeated spoof attempts
    Impact: Longer = detects persistent attempts; Shorter = detects rapid attacks
    Recommendation: 10 for typical, 5 for high-security

DROPOUT_DETECTION_THRESHOLD_DAYS = 3
    Type: int (consecutive days)
    Default: 3
    Purpose: Days absent before marking dropout
    Impact: Higher = detects chronic absences; Lower = detects short absences
    Recommendation: 3 for typical (1 week = 5 days alert), 5 for lenient

IMPOSSIBLE_ATTENDANCE_TIME_WINDOW_MINUTES = 5
    Type: int (minutes)
    Default: 5
    Purpose: Time window to detect impossible transitions between cameras
    Impact: Longer = detects less obvious fraud; Shorter = detects only obvious
    Recommendation: 5 for typical multi-camera setups

---

## PHASE 7: KUBERNETES & DEPLOYMENT HARDENING
## Goal: Production-ready health checks and monitoring

Health Check Endpoints:

GET /api/health
    Response: {"status": "ok"} or {"status": "error"}
    HTTP Code: 200 or 503
    Purpose: Kubernetes liveness probe
    Recommendation: 5 second initial delay, 10 second timeout
    K8s Config: livenessProbe.initialDelaySeconds=5, timeoutSeconds=10

GET /api/health/detailed
    Response: {status, database, camera_status, models_loaded}
    HTTP Code: 200 or 503
    Purpose: Detailed diagnostic checks
    Recommendation: For manual monitoring and troubleshooting
    Checks:
        - Database connectivity (MongoDB ping)
        - Camera pipeline status (at least 1 camera ready)
        - Model loading state (YuNet, ArcFace, anti-spoof)

GET /api/readiness
    Response: {"ready": true/false, "reason": "..."}
    HTTP Code: 200 or 503
    Purpose: Kubernetes readiness probe
    Recommendation: 10 second initial delay, 5 second timeout
    K8s Config: readinessProbe.initialDelaySeconds=10, timeoutSeconds=5
    Ready Conditions:
        ✓ Database connection active
        ✓ At least 1 model loaded
        ✓ Camera pipeline initialized
        ✓ Metrics engine running

---

## CONFIGURATION BEST PRACTICES

1. ENVIRONMENT-SPECIFIC SETTINGS
   
   Development:
       GRACEFUL_DEGRADATION_ENABLED=False
       DEBUG_MODE=True
       CPU_THRESHOLD=95
   
   Staging:
       GRACEFUL_DEGRADATION_ENABLED=True
       CPU_THRESHOLD=85
       DEBUG_MODE=False
   
   Production:
       GRACEFUL_DEGRADATION_ENABLED=True
       CPU_THRESHOLD=80
       DEBUG_MODE=False
       LIVENESS_STRICT_THRESHOLD=0.95  # Stricter in production

2. PERFORMANCE TUNING
   
   For High-Throughput (1000+ students/hour):
       - Increase FRAME_QUEUE_MAX_DEPTH to 100+
       - Decrease SLOW_FRAME_THRESHOLD_MS to 50
       - Increase CPU_THRESHOLD to 90 (allow more processing)
       - Decrease RECOGNIZED_COOLDOWN to 2s
   
   For High-Security (minimize false positives):
       - Increase RECOGNITION_TOP2_SIMILARITY_MARGIN to 0.10
       - Increase COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT to 0.6
       - Decrease LIVENESS_THRESHOLD to 0.85
       - Increase MIN_ENROLLMENT_IMAGES to 5
   
   For Resource-Constrained (edge devices):
       - Enable GRACEFUL_DEGRADATION_ENABLED
       - Lower CPU_THRESHOLD to 70
       - Decrease FRAME_QUEUE_MAX_DEPTH to 20
       - Enable GRACEFUL_DEGRADATION_DISABLE_ANTISPOOF

3. MONITORING & ALERTING
   
   Recommended Alerts:
       - Frame drop rate > 5%
       - Average frame time > 200ms
       - Spoof detection rate > 10%
       - Dropout detection rate > 20%
       - Camera disconnects > 3 in 1 hour
       - API latency > 500ms

4. VALIDATION RULES
   
   ✓ All weights must sum > 0
   ✓ Thresholds must be logically ordered (min < max)
   ✓ Timeouts must be positive
   ✓ Percentages must be 0-100
   ✓ Array thresholds must be min ≤ value ≤ max

---

## ENVIRONMENT VARIABLES

All parameters can be overridden via environment variables using uppercase names:

Examples:
    export CAMERA_FRAME_TIMEOUT_SECONDS=10.0
    export CPU_THRESHOLD=75
    export COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT=0.6

Loading in Python:
    import os
    timeout = float(os.environ.get('CAMERA_FRAME_TIMEOUT_SECONDS', 5.0))

---

## CONFIGURATION VALIDATION

Run validation check:
    python scripts/verify_upgrade.py

Checks:
    ✓ All parameters present in core/config.py
    ✓ Parameter types are correct
    ✓ Ranges are within acceptable bounds
    ✓ Weights sum to valid values
    ✓ Database connection works
    ✓ All modules can be imported

---

## TROUBLESHOOTING

Problem: Excessive frame drops
Solution:
    - Increase FRAME_QUEUE_MAX_DEPTH
    - Check CPU usage (may need to optimize processing)
    - Reduce capture FPS

Problem: System sluggish under load
Solution:
    - Enable GRACEFUL_DEGRADATION_ENABLED
    - Lower CPU_THRESHOLD
    - Disable expensive features (anti-spoof, PPE detection)

Problem: Too many false positives
Solution:
    - Increase RECOGNITION_TOP2_SIMILARITY_MARGIN
    - Increase COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT
    - Decrease LIVENESS_THRESHOLD

Problem: Enrollment validator rejecting good images
Solution:
    - Lower MIN_BLUR_SHARPNESS
    - Lower MAX_FACE_ANGLE_DEGREES
    - Adjust MIN_BRIGHTNESS/MAX_BRIGHTNESS
    - Reduce MIN_YAW_SPREAD_DEGREES

Problem: Security logs not appearing
Solution:
    - Verify database connection
    - Check /api/health/detailed output
    - Look for exceptions in logs

Problem: Health endpoints return 503
Solution:
    - Check database connectivity
    - Verify models are loaded (check logs for errors)
    - Ensure camera pipeline is initialized

---

## RELATED DOCUMENTATION

- QUICKSTART_UPGRADE.md - Testing guide
- core/config.py - Source of truth for all parameters
- docs/DEPLOYMENT.md - Docker/K8s deployment configuration
- docs/ARCHITECTURE.md - System design and components
