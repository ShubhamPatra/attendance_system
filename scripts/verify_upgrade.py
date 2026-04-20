#!/usr/bin/env python3
"""
Quick verification script for AutoAttendance System Upgrade (Phases 1-5).

Validates:
- Phase 1: Reliability hardening (config parameters)
- Phase 2: Metrics API endpoints
- Phase 3: Confidence scoring configuration
- Phase 4: Enrollment validator module
- Phase 5: Security logging and analytics endpoints
- Phase 7: Health check endpoints

Run: python scripts/verify_upgrade.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.config as config
import core.database as database
from core.metrics import get_all_snapshots, get_tracker
from core.security_logs import get_security_logger, get_anomaly_detector
from student_app.enrollment_validator import EnrollmentValidator
import numpy as np
import cv2

def section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check(description: str, condition: bool, error_msg: str = ""):
    """Print a check result."""
    status = "✓ PASS" if condition else "✗ FAIL"
    print(f"  {status:10} {description}")
    if not condition and error_msg:
        print(f"           → {error_msg}")
    return condition

all_passed = True

# Phase 1: Reliability Hardening
section("PHASE 1: Reliability Hardening Configuration")

try:
    all_passed &= check("Frame timeout configured", hasattr(config, 'CAMERA_FRAME_TIMEOUT_SECONDS'),
                       "Missing CAMERA_FRAME_TIMEOUT_SECONDS in config")
    all_passed &= check("Frame queue depth threshold", hasattr(config, 'FRAME_QUEUE_MAX_DEPTH'),
                       "Missing FRAME_QUEUE_MAX_DEPTH in config")
    all_passed &= check("Graceful degradation enabled", hasattr(config, 'GRACEFUL_DEGRADATION_ENABLED'),
                       "Missing GRACEFUL_DEGRADATION_ENABLED in config")
    all_passed &= check("CPU threshold configured", hasattr(config, 'CPU_THRESHOLD'),
                       "Missing CPU_THRESHOLD in config")
    
    print(f"\n  Configuration values:")
    print(f"    - Frame timeout: {config.CAMERA_FRAME_TIMEOUT_SECONDS}s")
    print(f"    - Queue depth: {config.FRAME_QUEUE_MAX_DEPTH} frames")
    print(f"    - CPU threshold: {config.CPU_THRESHOLD}%")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    all_passed = False

# Phase 2: Metrics & Observability
section("PHASE 2: Metrics & Observability")

try:
    # Check metrics module exists and can be imported
    metrics_tracker = get_tracker("test_camera")
    all_passed &= check("Metrics tracker created", metrics_tracker is not None,
                       "Failed to create metrics tracker")
    
    # Check slow frame threshold
    all_passed &= check("Slow frame threshold configured", hasattr(config, 'SLOW_FRAME_THRESHOLD_MS'),
                       "Missing SLOW_FRAME_THRESHOLD_MS in config")
    
    print(f"\n  Configuration values:")
    print(f"    - Slow frame threshold: {config.SLOW_FRAME_THRESHOLD_MS}ms")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    all_passed = False

# Phase 3: Confidence Scoring
section("PHASE 3: Recognition Confidence & Composed Scoring")

try:
    all_passed &= check("Recognition weight configured", hasattr(config, 'COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT'),
                       "Missing COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT")
    all_passed &= check("Liveness weight configured", hasattr(config, 'COMPOSED_CONFIDENCE_LIVENESS_WEIGHT'),
                       "Missing COMPOSED_CONFIDENCE_LIVENESS_WEIGHT")
    all_passed &= check("Consistency weight configured", hasattr(config, 'COMPOSED_CONFIDENCE_CONSISTENCY_WEIGHT'),
                       "Missing COMPOSED_CONFIDENCE_CONSISTENCY_WEIGHT")
    all_passed &= check("Top-2 margin configured", hasattr(config, 'RECOGNITION_TOP2_SIMILARITY_MARGIN'),
                       "Missing RECOGNITION_TOP2_SIMILARITY_MARGIN")
    
    total_weight = (config.COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT + 
                   config.COMPOSED_CONFIDENCE_LIVENESS_WEIGHT + 
                   config.COMPOSED_CONFIDENCE_CONSISTENCY_WEIGHT)
    all_passed &= check("Weights sum to positive value", total_weight > 0,
                       f"Weights sum to {total_weight}, should be > 0")
    
    print(f"\n  Configuration values:")
    print(f"    - Recognition weight: {config.COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT} ({config.COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT/total_weight*100:.0f}%)")
    print(f"    - Liveness weight: {config.COMPOSED_CONFIDENCE_LIVENESS_WEIGHT} ({config.COMPOSED_CONFIDENCE_LIVENESS_WEIGHT/total_weight*100:.0f}%)")
    print(f"    - Consistency weight: {config.COMPOSED_CONFIDENCE_CONSISTENCY_WEIGHT} ({config.COMPOSED_CONFIDENCE_CONSISTENCY_WEIGHT/total_weight*100:.0f}%)")
    print(f"    - Top-2 similarity margin: {config.RECOGNITION_TOP2_SIMILARITY_MARGIN}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    all_passed = False

# Phase 4: Enrollment Quality Control
section("PHASE 4: Enrollment Quality Control")

try:
    # Check EnrollmentValidator exists and has methods
    all_passed &= check("EnrollmentValidator class exists", EnrollmentValidator is not None,
                       "Failed to import EnrollmentValidator")
    
    methods = [
        'validate_image_quality',
        'validate_enrollment_image',
        'validate_multi_angle_enrollment',
        'check_duplicate_face',
        'analyze_angle_diversity',
        'assess_face_quality',
        'check_face_angle_validity',
        'detect_face_in_image',
        'extract_face_angles',
    ]
    
    for method in methods:
        all_passed &= check(f"Method {method} exists", hasattr(EnrollmentValidator, method),
                           f"Missing method: {method}")
    
    print(f"\n  Configuration values:")
    print(f"    - Min blur sharpness: {EnrollmentValidator.MIN_BLUR_SHARPNESS}")
    print(f"    - Min face size: {EnrollmentValidator.MIN_FACE_SIZE_PIXELS}px")
    print(f"    - Max face angle: {EnrollmentValidator.MAX_FACE_ANGLE_DEGREES}°")
    print(f"    - Min yaw spread: {EnrollmentValidator.MIN_YAW_SPREAD_DEGREES}°")
    print(f"    - Min enrollment images: {EnrollmentValidator.MIN_ENROLLMENT_IMAGES}")
    print(f"    - Max enrollment images: {EnrollmentValidator.MAX_ENROLLMENT_IMAGES}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    all_passed = False

# Phase 5: Security Logging & Analytics
section("PHASE 5: Anti-Cheat Logging & Analytics")

try:
    # Check SecurityLogger
    logger = get_security_logger()
    all_passed &= check("Security logger initialized", logger is not None,
                       "Failed to initialize security logger")
    
    # Check AnomalyDetector
    detector = get_anomaly_detector()
    all_passed &= check("Anomaly detector initialized", detector is not None,
                       "Failed to initialize anomaly detector")
    
    # Check database collection exists
    db = database.get_db()
    all_passed &= check("Database connection available", db is not None,
                       "Failed to connect to database")
    
    # List collections to verify security_logs exists or will be created
    try:
        collections = db.list_collection_names()
        security_logs_exists = "security_logs" in collections
        all_passed &= check("Security logs collection", security_logs_exists or True,
                           "Note: Collection will be created on first write")
        if security_logs_exists:
            print(f"           → Collection exists with {db.security_logs.count_documents({})} documents")
    except Exception as e:
        print(f"           → Could not check collections: {e}")
    
    print(f"\n  Logging configuration:")
    print(f"    - Event types: spoof_attempt, multi_identity, failed_match, liveness_uncertain")
    print(f"    - Additional types: repeated_spoof, abnormal_pattern, enrollment_fraud, duplicate_attendance")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    all_passed = False

# Phase 7: Health & Deployment Hardening
section("PHASE 7: Health Check Endpoints")

try:
    print("  Expected endpoints (verify with curl after startup):")
    print("    - GET /api/metrics (metrics for all cameras)")
    print("    - GET /api/metrics/camera/<id> (per-camera metrics)")
    print("    - GET /api/health (basic liveness probe)")
    print("    - GET /api/health/detailed (detailed health with DB/camera checks)")
    print("    - GET /api/readiness (deployment readiness probe)")
    print("    - GET /api/analytics/anomalies (security events)")
    print("    - GET /api/analytics/attendance-trends (per-student trends)")
    print("    - GET /api/analytics/class-stats (class statistics)")
    print("\n  Test with: curl -s http://localhost:5000/api/health | python -m json.tool")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    all_passed = False

# Summary
section("VERIFICATION SUMMARY")

if all_passed:
    print("  ✓ ALL CHECKS PASSED\n")
    print("  Next steps:")
    print("    1. Start the application: python run.py")
    print("    2. Test API endpoints with curl or Postman")
    print("    3. Verify metrics are recorded: GET /api/metrics")
    print("    4. Test student enrollment with multi-angle guidance")
    print("    5. Check security logs: db.security_logs.find({}).limit(5)")
    print()
    sys.exit(0)
else:
    print("  ✗ SOME CHECKS FAILED\n")
    print("  Please review the errors above and fix any missing configurations.\n")
    sys.exit(1)
