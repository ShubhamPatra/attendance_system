# Face Recognition Tuning Guide

## Understanding "Unknown" Labels

When you see your face labeled as "Unknown" (red box) even though it's clearly visible, it's **not a bug** — it's the system validating that the face is real and matches an enrolled student before marking attendance.

## Recognition Pipeline (Why It Takes Time)

The system runs in this order:

1. **Face Detection** (7-15ms)
   - YuNet model detects face location
   - Runs every N frames (configurable: `DETECTION_INTERVAL`)

2. **Liveness Verification** (20-25ms)  
   - Anti-spoofing model confirms it's a real face, not a photo/video
   - Uses temporal voting across multiple frames
   - **This is the largest time contributor**
   - Your successful recognition logged: `liveness_confidence=0.9735` ✅

3. **Face Recognition** (10-20ms)
   - InsightFace ArcFace generates 512-D embedding
   - Matches against enrolled students in FAISS index
   - Requires confidence above threshold

4. **Temporal Voting** 
   - System waits for consistent recognition across frames
   - Prevents false positives and flickering

## Why You See "Unknown" Then Recognition

**Typical sequence:**
- Frame 1-3: Face detected but liveness not yet confirmed → "Unknown" (red)
- Frame 4-5: Liveness confirmed + recognition matches → "Shubham Patra" (green)
- Frame 6+: Attendance marked once smoothed confidence reaches threshold

From your logs:
```
2026-04-20 19:07:07  INFO  New track #0 created at bbox=(207, 122, 123, 157)
                           [Face detected, liveness pending]

2026-04-20 19:07:18  INFO  Track #0: RECOGNIZED Shubham Patra 
                           (confidence=0.7211, smoothed_frames=2, blinks=0)
                           [Liveness confirmed + recognition matched]

2026-04-20 19:07:19  INFO  Attendance marked for student 69dfd2de458e5eed7684366f
                           [Temporal voting complete]
```

**Time elapsed:** ~11 seconds (due to system load degradation)

## Performance Optimization Settings

### Fast Recognition (Current - Optimized)
```
RECOGNITION_CONFIRM_FRAMES = 1           # Recognize immediately after confirmation
LIVENESS_MIN_HISTORY = 1                 # Fast liveness check (1 frame)
RECOGNITION_MIN_CONFIDENCE = 0.42        # Relaxed confidence threshold
DETECTION_INTERVAL = 3                   # Detect every 3 frames (faster)
```

**Result:** Recognition in 2-5 seconds for good face positioning

### Balanced (More Secure)
```
RECOGNITION_CONFIRM_FRAMES = 2           
LIVENESS_MIN_HISTORY = 2                 
RECOGNITION_MIN_CONFIDENCE = 0.46        
DETECTION_INTERVAL = 6                   
```

**Result:** Recognition in 5-10 seconds, fewer false positives

### Ultra-Secure (Stricter - Slower)
```
RECOGNITION_CONFIRM_FRAMES = 3           
LIVENESS_MIN_HISTORY = 3                 
RECOGNITION_MIN_CONFIDENCE = 0.50        
DETECTION_INTERVAL = 10                  
```

**Result:** Recognition in 10-20 seconds, maximum accuracy

## Key Diagnostic Metrics (From Logs)

| Metric | What It Means | Your Value |
|--------|---------------|-----------|
| `confidence` | How well your face matches enrolled embedding | 0.7211 (71%) - Good |
| `smoothed_frames` | Frames needed to confirm | 2 frames |
| `liveness_confidence` | Probability it's a real face | 0.9735 (97%) - Excellent |
| `composed_confidence` | Final attendance confidence | 0.6526 (65%) - Passed |

## Slowness Causes (Troubleshooting)

### 1. **System Memory > 85%** (Your Issue!)
**Symptom:** Slow frame detected (1595.2 ms instead of ~100 ms)
```
Camera 0: System load degradation detected (Memory 87.5% > 85.0%)
```
**Solution:**
- Close other applications
- Increase available RAM
- Reduce frame resolution in camera settings
- Set environment variable: `GRACEFUL_DEGRADATION_ENABLED=0` to disable (if you have spare CPU)

### 2. **Low Enrollment Quality**
**Symptom:** Confidence stays low (< 0.70)
**Solution:**
- Re-enroll with better lighting (front-facing, well-lit)
- Multiple angles during enrollment
- See [Enrollment Quality Guide](../SETUP.md#enrollment-best-practices)

### 3. **System Under Heavy Load**
**Symptom:** All frames slow, not just first detection
```
total=1595.2 ms > threshold; detection=9.2 ms, recognition=18.2 ms, liveness=21.8 ms
```
**Solution:**
- Run on dedicated hardware
- Reduce other processes
- Upgrade CPU/GPU if possible

## How to Verify It's Working Correctly

Check logs for this pattern (means system is working as designed):

✅ **Good Behavior:**
```
New track created at bbox=(...) 
    → [Wait for liveness confirmation]
RECOGNIZED [Name] (confidence=0.72, smoothed_frames=2)
    → [Temporal voting complete]
Attendance marked (composed=0.65)
    → [Successfully recorded]
```

❌ **Problems to Watch For:**
```
Multiple "Unknown" labels (5+ seconds)
    → Enrollment quality issue or wrong person
Track expires without recognition
    → Face not in database
Sudden slow frames (>500ms)
    → System memory/CPU spike
```

## Tuning for Your Setup

### For Home/Demo Use (Fast Recognition):
```bash
export RECOGNITION_CONFIRM_FRAMES=1
export LIVENESS_MIN_HISTORY=1
export RECOGNITION_MIN_CONFIDENCE=0.40
export DETECTION_INTERVAL=3
```

### For Production (Balanced):
```bash
export RECOGNITION_CONFIRM_FRAMES=2
export LIVENESS_MIN_HISTORY=2
export RECOGNITION_MIN_CONFIDENCE=0.46
export DETECTION_INTERVAL=6
```

### For High-Security (Slow but Accurate):
```bash
export RECOGNITION_CONFIRM_FRAMES=3
export LIVENESS_MIN_HISTORY=3
export RECOGNITION_MIN_CONFIDENCE=0.50
export DETECTION_INTERVAL=10
```

Set these **before** running `python run.py`

## Next Steps

1. **Lower System Memory Pressure**
   - Your biggest blocker: Memory at 87.5%
   - Close unnecessary apps
   - Try reducing to 1 concurrent camera if multiple

2. **Re-enroll with Better Images**
   - Ensure good lighting during enrollment
   - Face fills 60-70% of frame
   - Multiple angles for diversity

3. **Monitor Recognition Quality**
   - Watch for confidence scores
   - Log better enrollment → higher confidence
   - More students in database → faster filtering

4. **Track Performance**
   - Use `/api/metrics` endpoint to monitor frame times
   - Check logs for degradation triggers
   - Profile GPU/CPU usage during peak times

---

**Reference:** See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for all tunable parameters
