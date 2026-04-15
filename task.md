# AutoAttendance Upgrade Tasks

Execution mode: complete tasks one-by-one and validate before moving forward.

## Task 1 - Attendance Session Data Layer

- [x] Add `attendance_sessions` collection and indexes
- [x] Add DB functions:
  - `create_attendance_session()`
  - `end_attendance_session()`
  - `get_active_attendance_session()`
  - `get_attendance_session_by_id()`
  - `auto_close_idle_attendance_sessions()`

## Task 2 - Session-Aware Attendance Writes

- [x] Add optional `session_id` in attendance write paths
- [x] Enforce active session required for camera auto-marking
- [x] Preserve one-mark-per-day uniqueness

## Task 3 - Session APIs

- [x] Add start-session endpoint
- [x] Add end-session endpoint
- [x] Add get-active-session endpoint per `camera_id`

## Task 4 - Multi-Frame Confirmation

- [x] Add per-track rolling prediction buffer (size 5)
- [x] Confirm only on stable policy (`>=3/5`)
- [x] Reject unstable predictions

## Task 5 - Two-Stage Recognition

- [x] Implement Stage 1 fast candidate filter
- [x] Implement Stage 2 precise confirmation
- [x] Pass candidates Stage1 -> Stage2
- [x] Run Stage 2 only when needed

## Task 6 - Recognition Cache

- [x] Add `tracker_id` identity cache with expiry
- [x] Skip recomputation when cache valid

## Task 7 - Frame Quality Gate

- [x] Add blur/brightness/face-size quality check
- [x] Reject poor quality early

## Task 8 - Adaptive Thresholds

- [x] Add dynamic threshold function
- [x] Integrate bounded adaptive thresholding

## Task 9 - CameraManager Abstraction

- [x] Introduce `CameraManager` for webcam/IP sources
- [x] Centralize reconnection and diagnostics
- [x] Keep multi-camera compatibility

## Task 10 - GPU Provider Support

- [x] Add GPU config flags
- [x] Use GPU provider when available
- [x] Fallback to CPU safely

## Task 11 - Advanced Metrics

- [x] Add per-stage latency tracking
- [x] Add FPS per camera
- [x] Extend `/api/metrics` JSON

## Task 12 - Unknown Face Logging

- [x] Save unknown snapshots to `unknown_faces/`
- [x] Log confidence and timestamp
- [x] Add cooldown dedup

## Task 13 - Tests and Regression Checks

- [x] Add/update tests for all features
- [x] Run targeted suites
- [x] Verify no meaningful FPS regression

## Task 14 - AutoAttend Structure Migration (Phase 1)

- [x] Create top-level package layout to mirror `AutoAttend/AutoAttendance/`
  - [x] `core/`
  - [x] `recognition/`
  - [x] `anti_spoofing/`
  - [x] `admin_app/`
  - [x] `tasks/`
- [x] Add initial module files to match secondary project structure
- [x] Keep existing runtime stable while introducing new package entrypoints

## Task 15 - AutoAttend Structure Migration (Phase 2)

- [x] Migrate imports from legacy `app_*` modules to new package paths
- [x] Convert root-level shim modules to compatibility wrappers only
- [x] Keep external behavior unchanged during path migration

## Task 16 - DAO and Data Layer Consolidation

- [x] Introduce DAO layer aligned with `AutoAttend/core/models.py`
- [x] Integrate DAO into attendance/session write paths
- [x] Preserve one-mark-per-day uniqueness and active-session guardrails

## Task 17 - Route and Service Integration

- [x] Update web routes to use migrated package paths
- [x] Update Celery/task modules to new `tasks/` package structure
- [x] Ensure admin/student entrypoints resolve new package layout

## Task 18 - Migration Validation and Cleanup

- [x] Add/update tests for import-path migration and DAO behavior
- [x] Run targeted regression suites (DB, routes, camera, pipeline)
- [x] Remove obsolete structural duplication after validation

## Current Focus

- [x] Task 1 completed
- [x] Task 2 completed
- [x] Task 3 completed
- [x] Task 4 completed
- [x] Task 5 completed
- [x] Task 6 completed
- [x] Task 7 completed
- [x] Task 8 completed
- [x] Task 9 completed
- [x] Task 10 completed
- [x] Task 11 completed
- [x] Task 12 completed
- [x] Task 13 completed
- [x] Task 14 completed
- [x] Task 15 completed
- [x] Task 16 completed
- [x] Task 17 completed
- [x] Task 18 completed
