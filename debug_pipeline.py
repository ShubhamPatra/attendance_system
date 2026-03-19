"""
debug_pipeline.py – Standalone pipeline validation script.

Run this to test each stage of the recognition pipeline independently:
  1. Database → encoding cache load
  2. Encoding validation (shape check)
  3. Camera → frame capture
  4. YuNet → face detection
  5. Anti-spoofing → liveness check
  6. Face encoding → 128-D vector
  7. Distance computation → match check

Usage:
    cd d:\Projects\FinalYearProject\attendance_system
    python debug_pipeline.py
"""

import os
import sys
import time

# Ensure the package directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np


def banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def step(n: int, title: str):
    print(f"\n--- Step {n}: {title} ---")


def main():
    banner("AutoAttendance Pipeline Diagnostic")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Step 1: Database & encoding cache ────────────────────────
    step(1, "Load encoding cache from database")
    try:
        import config
        import database
        from face_engine import encoding_cache

        database.ensure_indexes()
        encoding_cache.load()

        flat_enc, flat_idx, ids, names = encoding_cache.get_flat()
        n_students = encoding_cache.size
        n_encodings = len(flat_enc) if flat_enc is not None else 0

        print(f"  ✓ Connected to MongoDB")
        print(f"  Students in cache: {n_students}")
        print(f"  Total encodings:   {n_encodings}")

        if n_students == 0:
            print("  ✗ WARNING: No students found! Recognition cannot work.")
            print("    Register at least one student first.")
        else:
            for i, (sid, name) in enumerate(zip(ids, names)):
                print(f"    [{i}] {name} (id={sid})")
    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        return

    # ── Step 2: Validate all stored encodings ────────────────────
    step(2, "Validate stored encoding shapes")
    _, _, all_ids, all_names = encoding_cache.get_flat()
    _, _, enc_lists = encoding_cache.get_all()[:3] if encoding_cache.size > 0 else ([], [], [])
    all_ok = True

    if encoding_cache.size > 0:
        _, _, enc_lists_full = encoding_cache.get_all()
        for i, (name, encs) in enumerate(zip(all_names, enc_lists_full)):
            for j, enc in enumerate(encs):
                if enc.shape != (128,):
                    print(f"  ✗ CORRUPT: {name} encoding[{j}] shape={enc.shape}")
                    all_ok = False
                elif enc.dtype != np.float64:
                    print(f"  ✗ WRONG DTYPE: {name} encoding[{j}] dtype={enc.dtype}")
                    all_ok = False
        if all_ok:
            print(f"  ✓ All encodings valid (128-D float64)")
    else:
        print("  ⊘ Skipped (no students)")

    # ── Step 3: Camera frame capture ─────────────────────────────
    step(3, "Capture camera frame")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ✗ FAILED: Cannot open camera (index 0)")
        print("    Check if camera is connected and not in use.")
        return

    # Warm up camera (skip first few frames)
    for _ in range(10):
        cap.read()
    time.sleep(0.3)

    ok, frame = cap.read()
    if not ok or frame is None:
        print("  ✗ FAILED: Camera returned no frame")
        cap.release()
        return

    h, w = frame.shape[:2]
    print(f"  ✓ Frame captured: {w}x{h}, dtype={frame.dtype}")

    # ── Step 4: YuNet face detection ─────────────────────────────
    step(4, "YuNet face detection")
    try:
        import pipeline
        if os.path.isfile(config.YUNET_MODEL_PATH):
            pipeline.init_yunet(config.YUNET_MODEL_PATH, config.FRAME_PROCESS_WIDTH)
            boxes = pipeline.detect_faces_yunet(frame)
            print(f"  ✓ Faces detected: {len(boxes)}")
            for i, box in enumerate(boxes):
                print(f"    [{i}] bbox=(x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]})")
            if not boxes:
                print("  ⚠ No faces detected — make sure a face is visible to the camera")
        else:
            print(f"  ✗ YuNet model not found at: {config.YUNET_MODEL_PATH}")
            boxes = []
    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        boxes = []

    if not boxes:
        print("\n  Cannot continue without detected faces.")
        cap.release()
        return

    bbox = boxes[0]  # Use first detected face

    # ── Step 5: Anti-spoofing ────────────────────────────────────
    step(5, "Anti-spoofing liveness check")
    try:
        import anti_spoofing
        anti_spoofing.init_models()
        label, conf = anti_spoofing.check_liveness(frame)
        is_real = (label == 1 and conf >= config.LIVENESS_CONFIDENCE_THRESHOLD)
        print(f"  Label:      {label}  (1=real, 0=spoof)")
        print(f"  Confidence: {conf:.4f}")
        print(f"  Threshold:  {config.LIVENESS_CONFIDENCE_THRESHOLD}")
        print(f"  Decision:   {'✓ REAL' if is_real else '✗ SPOOF/UNCERTAIN'}")
        if not is_real:
            print("  ⚠ Anti-spoofing would BLOCK recognition!")
            print("    Try: set BYPASS_ANTISPOOF=1 to test without it")
    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        print("  ⚠ Anti-spoofing failure returns (0, 0.0) → blocks recognition!")

    # ── Step 6: Face encoding ────────────────────────────────────
    step(6, "Generate face encoding")
    try:
        from recognition import encode_face, check_face_quality_gate

        # Quality gate check first (informational)
        qok, qreason = check_face_quality_gate(frame, bbox)
        print(f"  Quality gate: {'✓ PASS' if qok else f'✗ FAIL ({qreason})'}")

        encoding = encode_face(frame, bbox)
        if encoding is not None:
            print(f"  ✓ Encoding generated: shape={encoding.shape}, dtype={encoding.dtype}")
            print(f"    Min={encoding.min():.4f}, Max={encoding.max():.4f}, "
                  f"Mean={encoding.mean():.4f}")
        else:
            print("  ✗ FAILED: encode_face returned None")
            print("    Try: set BYPASS_QUALITY_GATE=1 to test without quality gate")
            cap.release()
            return
    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        cap.release()
        return

    # ── Step 7: Distance computation & matching ──────────────────
    step(7, "Match against encoding cache")
    if n_students == 0:
        print("  ⊘ Skipped (no students in cache)")
    else:
        from face_engine import recognize_face

        # Compute distances to all students
        flat_enc_arr, flat_idx_arr, c_ids, c_names = encoding_cache.get_flat()
        if flat_enc_arr is not None:
            distances = np.linalg.norm(flat_enc_arr - encoding, axis=1)

            # Per-student min distance
            n_s = len(c_ids)
            min_dists = np.full(n_s, np.inf)
            np.minimum.at(min_dists, flat_idx_arr, distances)

            print(f"\n  Distance table (threshold={config.RECOGNITION_THRESHOLD:.4f}):")
            print(f"  {'Student':<25} {'Min Dist':<12} {'Match?'}")
            print(f"  {'-'*25} {'-'*12} {'-'*6}")
            for i, (name, dist) in enumerate(zip(c_names, min_dists)):
                match = "✓ YES" if dist <= config.RECOGNITION_THRESHOLD else "✗ NO"
                print(f"  {name:<25} {dist:<12.4f} {match}")

            best_idx = int(np.argmin(min_dists))
            best_dist = float(min_dists[best_idx])
            print(f"\n  Best match: {c_names[best_idx]} (distance={best_dist:.4f})")

            if best_dist <= config.RECOGNITION_THRESHOLD:
                conf = 1.0 - best_dist
                print(f"  ✓ RECOGNIZED with confidence={conf:.4f}")
            else:
                print(f"  ✗ NOT RECOGNIZED (distance {best_dist:.4f} > threshold {config.RECOGNITION_THRESHOLD:.4f})")
                if best_dist <= 0.6:
                    print(f"  💡 Would match with threshold=0.6")
                elif best_dist <= 0.7:
                    print(f"  💡 Would match with threshold=0.7")

        # Also run through the actual API
        result = recognize_face(encoding)
        print(f"\n  recognize_face() API result: {result}")

    cap.release()
    banner("Diagnostic Complete")


if __name__ == "__main__":
    main()
