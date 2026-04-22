# Anti-Spoofing & Liveness Detection Guide

## Overview

The AutoAttendance system uses **Silent-Face-Anti-Spoofing** for liveness detection, preventing fraudulent attendance marking via printed photos, videos, or synthetic media.

## Current Implementation

### Silent-Face-Anti-Spoofing

**Status**: ✅ **Fully Operational**

Silent-Face-Anti-Spoofing is a CNN-based multi-model ensemble that classifies face frames into three categories:

- **Class 0**: Spoof/Fake (photo, video, mask, etc.)
- **Class 1**: Real/Live face (human face)
- **Class 2**: Other attacks (depth-based attacks, unusual artifacts)

**Models Used**:
1. `2.7_80x80_MiniFASNetV2.pth` - Lightweight MiniFASNet-V2 (80×80 input)
2. `4_0_0_80x80_MiniFASNetV1SE.pth` - MiniFASNet-V1-SE variant (80×80 input)

Located at: `Silent-Face-Anti-Spoofing/resources/anti_spoof_models/`

## How It Works

### Detection Pipeline

```
Frame → Face Crop → Model Ensemble Prediction → Confidence Averaging → Decision Logic
```

### Confidence Thresholds

The system uses **aggressive** anti-spoofing logic by default:

```python
# Acceptance criteria for REAL face
if real_conf >= 0.85 and spoof_conf <= 0.15:
    label = 1 (REAL)     # Accept as live face
elif spoof_conf > real_conf:
    label = 0 (SPOOF)    # Clearly a spoof
else:
    label = 0 (SPOOF)    # Uncertain → reject (conservative)
```

**Key Decision Points**:

| Scenario | Real Conf | Spoof Conf | Decision | Reason |
|----------|-----------|-----------|----------|--------|
| Live face | 0.99 | 0.001 | ✅ ACCEPT | Clear real face |
| Photo/video | 0.22 | 0.62 | ❌ REJECT | Clear spoof |
| Partial face | 0.47 | 0.015 | ❌ REJECT | Uncertain (conservative) |
| Extreme angle | 0.60 | 0.02 | ❌ REJECT | Doesn't meet 0.85 threshold |

### Configuration

```bash
# Liveness thresholds (core/config.py)
LIVENESS_CONFIDENCE_THRESHOLD = 0.55      # Overall minimum confidence
LIVENESS_REAL_FAST_CONFIDENCE = 0.72      # Used in legacy logic
LIVENESS_EARLY_REJECT_CONFIDENCE = 0.5    # Minimum for any valid detection
LIVENESS_SPOOF_CONFIDENCE_MIN = 0.6       # Spoof detection threshold
```

## Performance Metrics

### Accuracy

Testing on real faces vs. printed photos:

| Test Condition | Real Face | Photo | Accuracy |
|---|---|---|---|
| Frontal lighting | 0.9959 | 0.2176 | ✅ High separation |
| Side angle | 0.9950 | 0.4183 | ✅ Good separation |
| Varied pose | 0.9998 | 0.1409 | ✅ Excellent |
| Fast movement | 0.9997 | 0.5986 | ✅ Good |

### Speed

- **Per-frame latency**: ~30-35ms per frame (CUDA)
- **CPU mode**: ~80-100ms per frame
- **Throughput**: ~30 FPS (GPU)

## Supplementary Liveness Checks

### Blink Detection

Eye-Aspect Ratio (EAR) tracking detects blinks:

```python
# EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
# Where p1-p6 are eye landmarks

BLINK_EAR_THRESHOLD = 0.25       # EAR below this = closed eye
BLINK_CONSEC_FRAMES = 3          # Frames to confirm blink
```

**Purpose**: Supplements model-based liveness with motion evidence

### Head Movement Detection

Tracks face centroid displacement over time:

```python
MOVEMENT_VELOCITY_THRESHOLD = 2.0  pixels/frame
MOVEMENT_MIN_FRAMES = 5            consecutive frames needed
```

**Purpose**: Detects subtle natural head movements (rejects static photos)

### Frame Heuristics

Additional quality checks:

```python
LIVENESS_SCREEN_BRIGHTNESS_MIN = 200    # Excessive brightness → screen spoof
LIVENESS_SCREEN_LAPLACIAN_MIN = 100     # Low blur variance → printed photo
LIVENESS_SCREEN_CONTRAST_MIN = 10       # Low contrast → screen content
LIVENESS_SCREEN_HIGHLIGHT_RATIO_MAX = 0.10  # Too many bright pixels
```

## Troubleshooting

### "Photos are marked as real (false positives)"

**Check**:
1. Are thresholds too low?
   ```bash
   LIVENESS_REAL_FAST_CONFIDENCE=0.72   # Consider raising to 0.85+
   ```

2. Is model failing?
   - Check logs for: `[Liveness] Real=X, Spoof=Y, Other=Z`
   - Live face should show: Real=0.95+, Spoof≤0.05
   - Photo should show: Real<0.50 or Spoof>0.40

3. Are face crops too small/blurry?
   - Check: `face_area` and `laplacian_var` in logs
   - May require closer camera positioning

### "Live faces rejected as spoof (false negatives)"

**Check**:
1. Thresholds too strict?
   ```bash
   LIVENESS_REAL_FAST_CONFIDENCE=0.72   # Try lowering to 0.65-0.70
   ```

2. Poor lighting conditions?
   - Ensure even illumination, no backlighting
   - Avoid shadows across face

3. Face too small?
   ```bash
   LIVENESS_MIN_FACE_SIZE_PIXELS=64     # Default; adjust if needed
   ```

4. Extreme pose?
   - Profile views may be rejected (correct behavior for security)
   - Ask user to face camera more directly

### "Spoof detection too aggressive"

**Solution**:
Adjust thresholds in `vision/anti_spoofing.py`:

```python
# Make more lenient
if real_conf >= 0.70 and spoof_conf <= 0.20:  # Lowered from 0.85/0.15
    return 1, real_conf
```

## Advanced Configuration

### Disable Anti-Spoofing (Unsafe - Demo/Testing Only)

```bash
# Environment variable
DISABLE_ANTISPOOFING=1

# Or in code (NOT recommended for production)
config.DISABLE_ANTISPOOFING = True
```

**Note**: This marks ALL faces as real - removes spoofing protection!

### GPU vs CPU Mode

```bash
# Use CPU for anti-spoofing (slower but always available)
MINIFASNET_ENABLE_GPU=0

# Use GPU (default, ~3x faster)
MINIFASNET_ENABLE_GPU=1
```

## Model Architecture Details

### MiniFASNet-V2

**Inputs**: 80×80 RGB image
**Output**: 3-class softmax (spoof, real, other)
**Architecture**: 
- Lightweight CNN with depthwise separable convolutions
- ~400K parameters
- Trained on multi-domain spoofing datasets

### Ensemble Logic

Both models predict independently, results averaged:

```python
prediction = (model1_output + model2_output) / 2
final_class = argmax(prediction)
confidence = prediction[final_class]
```

## References

- **Paper**: "Searching Central Difference Convolutional Networks for Face Anti-Spoofing"
- **Repository**: https://github.com/ChaitanyaBapat/Silent-Face-Anti-Spoofing
- **Datasets Trained**: NUAA, SiW, CASIA-FASD, MSU-MFSD, RealWorld

## Support

For issues or questions:
1. Check logs: `[Liveness]` prefix in console output
2. Run diagnostic: `python scripts/debug_liveness.py` (if available)
3. Adjust thresholds in `core/config.py` and `vision/anti_spoofing.py`
4. See main [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) guide
