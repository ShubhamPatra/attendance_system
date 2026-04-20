"""
Quick Start Guide for AutoAttendance System Upgrade

This guide helps you verify and test all Phase 1-5 implementations.
"""

# ============================================================================
# STEP 1: VERIFICATION
# ============================================================================

# Run the verification script to check all implementations
python scripts/verify_upgrade.py

# Expected output: "✓ ALL CHECKS PASSED"


# ============================================================================
# STEP 2: START APPLICATION
# ============================================================================

# Start the admin app (where routes are registered)
python run_admin.py

# Expected log lines:
#   - "Metrics routes registered"
#   - "Analytics routes registered"
#   - "Initializing camera pipelines..."


# ============================================================================
# STEP 3: TEST API ENDPOINTS (in separate terminal)
# ============================================================================

# Health checks
curl http://localhost:5000/api/health
# Expected: {"status": "ok"}

curl http://localhost:5000/api/health/detailed
# Expected: {"status": "ok", "database": "ok", "models": {...}}

curl http://localhost:5000/api/readiness
# Expected: {"ready": true} or {"ready": false, "reason": "..."}

# Metrics (Phase 2)
curl http://localhost:5000/api/metrics | python -m json.tool
# Expected: Aggregated metrics across all cameras

curl "http://localhost:5000/api/metrics/camera/0" | python -m json.tool
# Expected: Per-camera metrics (if camera 0 is active)

# Analytics (Phase 5)
curl "http://localhost:5000/api/analytics/anomalies?hours=24" | python -m json.tool
# Expected: Recent security events

curl "http://localhost:5000/api/analytics/attendance-trends?days=30" | python -m json.tool
# Expected: Per-student attendance and dropout indicators

curl "http://localhost:5000/api/analytics/class-stats?days=30" | python -m json.tool
# Expected: Class-wise statistics


# ============================================================================
# STEP 4: TEST STUDENT ENROLLMENT (Phase 4)
# ============================================================================

# 1. Start student app
python run_student.py

# 2. Navigate to http://localhost:5001/student/register
# 3. Create an account and proceed to capture

# 4. Click "Start Camera"
# 5. Observe angle guidance:
#    - Frame 1: "Face front"
#    - Frame 2: "Turn head ~30° to the left"
#    - Frame 3: "Turn head ~30° to the right"
#    - Frames 4-5: "Tilt head up or down"

# 6. Capture 3-5 frames with different angles
# 7. Submit and observe validation feedback:
#    - ✓ Frame 1: Quality status
#    - ✓ Frame 2: Quality status
#    - Angle Diversity: Sufficient or error
#    - Duplicate Detection: Checked automatically


# ============================================================================
# STEP 5: VERIFY DATABASE CHANGES (Phase 3 & 5)
# ============================================================================

# Connect to MongoDB and verify:

# Check attendance records have composed confidence scores
db.attendance.findOne({composed_confidence: {$exists: true}})
# Expected: Document with liveness_score, consistency_score, composed_confidence

# Check security logs collection exists
db.security_logs.count()
# Expected: Number of security events (>0 if spoofs detected)

# Verify indexes
db.security_logs.getIndexes()
# Expected: 5 indexes (timestamp, type_timestamp, student_timestamp, etc.)

# View sample security events
db.security_logs.find({}).limit(5).pretty()
# Expected: Events like {type: "spoof_attempt", severity: "high", ...}


# ============================================================================
# STEP 6: VERIFY METRICS RECORDING (Phase 2)
# ============================================================================

# While camera is running, check metrics are updating
while true; do
    curl -s http://localhost:5000/api/metrics | jq '.fps, .avg_frame_time_ms'
    sleep 1
done

# Expected: FPS >10, avg_frame_time <100ms (depending on system)


# ============================================================================
# STEP 7: TEST SPOOF DETECTION LOGGING (Phase 5)
# ============================================================================

# Attempt to enroll with a spoof (phone screen, photo)
# Check that security event was logged:

db.security_logs.find({type: "spoof_attempt"}).limit(1).pretty()
# Expected: Document with liveness_score: <0.5, severity: "high"


# ============================================================================
# STEP 8: TEST ANOMALY DETECTION (Phase 5)
# ============================================================================

# Check for students with repeated spoof attempts
curl "http://localhost:5000/api/analytics/anomalies?event_type=repeated_spoof" | python -m json.tool

# Check for students showing dropout pattern
curl "http://localhost:5000/api/analytics/attendance-trends?days=30" | jq '.trends[] | select(.dropout_detected == true)'


# ============================================================================
# COMMON ISSUES & TROUBLESHOOTING
# ============================================================================

# Issue: "Metrics routes not registered"
# Solution: Check admin_app/app.py has import and register call

# Issue: "/api/metrics returns 404"
# Solution: Restart admin app, verify routes are printed in logs

# Issue: "Security logs collection not found"
# Solution: This is normal - collection is auto-created on first write

# Issue: "Enrollment validator not found"
# Solution: Verify student_app/enrollment_validator.py exists and paths are correct

# Issue: "Angle guidance not showing"
# Solution: Ensure student_app/templates/capture.html was updated

# Issue: "Validation feedback not displayed"
# Solution: Check browser console for JavaScript errors, verify API response format


# ============================================================================
# LOAD TESTING (Optional)
# ============================================================================

# Test with multiple cameras
for i in {0..4}; do
    python scripts/debug_pipeline.py --camera $i &
done

# Monitor metrics aggregation
watch -n 1 'curl -s http://localhost:5000/api/metrics | jq ".fps, .cameras_active"'

# Check for graceful degradation under load
# - If CPU >80%, detection_interval should increase
# - If CPU >80%, antispoof should disable
# Monitor logs: grep -i "graceful_degradation" logs/camera.log


# ============================================================================
# NEXT STEPS
# ============================================================================

# Phase 6: Configuration Documentation
# - Review all parameters in core/config.py
# - Update SETUP.md with new configuration options
# - Document impact of each threshold

# Phase 7: Kubernetes Deployment
# - Configure health probes: liveness=/api/health, readiness=/api/readiness
# - Set resource requests/limits based on load testing
# - Configure security logging retention policy

# Phase 8+: Advanced Features (Optional)
# - Real-time anomaly alerts
# - Advanced analytics dashboards
# - Multi-site federation
# - Edge deployment support
