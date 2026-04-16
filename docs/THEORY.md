# Theory & ML Algorithms

## Table of Contents

1. [Face Detection (YuNet)](#face-detection-yunet)
2. [Face Recognition (ArcFace)](#face-recognition-arcface)
3. [Cosine Similarity Matching](#cosine-similarity-matching)
4. [Face Alignment & Quality Gating](#face-alignment--quality-gating)
5. [Object Tracking (CSRT)](#object-tracking-csrt)
6. [Anti-Spoofing & Liveness Detection](#anti-spoofing--liveness-detection)
7. [Motion Detection & Optical Flow](#motion-detection--optical-flow)
8. [Multi-Frame Confirmation](#multi-frame-confirmation)
9. [Threshold Calibration & Trade-Offs](#threshold-calibration--trade-offs)

---

## Face Detection (YuNet)

### Overview

**YuNet** is a lightweight face detection model designed for real-time performance on edge devices. It combines:
- **Backbone**: CSPDarkNet-based feature extraction.
- **Detector heads**: Multi-scale face predictions with bounding boxes and 5-point landmarks.
- **Format**: ONNX (Open Neural Network Exchange) for cross-platform inference.

### Model Architecture

```
Input: RGB Image (320×320 or larger)
    ↓
Backbone (CSPDarkNet):
  - Multiple blocks with depthwise-separable convolutions
  - Features at multiple scales (P3, P4, P5)
    ↓
Detection Heads:
  - Bbox predictor: (x, y, w, h)
  - Confidence scorer: [0, 1] probability
  - Landmark predictor: (x₁, y₁, ..., x₅, y₅) 5 points
    ↓
Output: Detections
```

### Mathematical Details

**Detection Head Output**:

For each anchor at scale $s$ and spatial position $(i, j)$:

$$\text{bbox} = (x_c + \Delta x, y_c + \Delta y, s \cdot e^{\Delta w}, s \cdot e^{\Delta h})$$

Where:
- $(x_c, y_c)$ = anchor center
- $(\Delta x, \Delta y)$ = predicted offset (normalized by anchor size)
- $s$ = anchor scale
- $(\Delta w, \Delta h)$ = log-space width/height deltas

**Confidence Score**:

$$P(\text{face}) = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$

Where $\sigma$ is the sigmoid function; typically thresholded at 0.62 (`YUNET_SCORE_THRESHOLD`).

### Implementation in AutoAttendance

**File**: [vision/pipeline.py](../vision/pipeline.py)

```python
def detect_faces_yunet(frame):
    """
    Detect faces using YuNet ONNX model.
    
    Args:
        frame: OpenCV frame (BGR, HxWx3)
    
    Returns:
        List of detections: [(x, y, w, h, confidence, landmarks)]
    """
    detector.setInputSize((320, 320))
    detector.setScoreThreshold(YUNET_SCORE_THRESHOLD)  # 0.62
    
    results = detector.detect(frame)
    return results
```

### Trade-Offs

| Aspect | Benefit | Cost |
|---|---|---|
| **Lightweight** | Runs on CPU in ~50ms | Reduced accuracy vs. larger models |
| **ONNX format** | Cross-platform; GPU via ONNX Runtime | Loss of original PyTorch/TensorFlow optimizations |
| **320×320 input** | Speed | Requires frame resizing; may miss small faces |
| **5-point landmarks** | Sufficient for alignment | Not detailed (68-point or 106-point models available) |

---

## Face Recognition (ArcFace)

### Overview

**ArcFace** (Additive Angular Margin) is a face embedding method that learns discriminative 512-dimensional vectors:

- Each face maps to a point on a hypersphere.
- Similar faces cluster together.
- Dissimilar faces are pushed far apart.

### Mathematical Foundation

**Softmax Loss** (baseline):

$$L = -\log\frac{e^{W_{y_i}^T f_i}}{e^{W_{y_i}^T f_i} + \sum_{j \neq y_i} e^{W_j^T f_i}}$$

Where:
- $f_i$ = embedding (unit L2-norm: $\|f_i\| = 1$)
- $W_j$ = class weight vectors (also L2-normalized)
- $y_i$ = true class (identity)

**ArcFace Loss** (proposed):

$$L = -\log\frac{e^{s(\cos(\theta_{y_i} + m) - \cos(\theta_{y_i}))}}{\sum_j e^{s \cos(\theta_j)}}$$

Or equivalently:

$$L = -\log\frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cos(\theta_j)}}$$

Where:
- $\theta_j$ = angle between $f_i$ and $W_j$
- $m$ = angular margin (typically 0.5 radians)
- $s$ = scale factor (typically 64)

**Key Insight**: By adding margin $m$ to the correct class angle, ArcFace encourages embeddings to be far from class boundaries, increasing robustness to variations.

### Embedding Space Properties

In 512-D embedding space with L2-normalization:

**Cosine Similarity**:

$$\cos(\theta_{ij}) = f_i \cdot f_j = \sum_{k=1}^{512} f_{i,k} \times f_{j,k}$$

Since $\|f_i\| = \|f_j\| = 1$:

$$\cos(\theta_{ij}) \in [-1, 1]$$

- $\cos(\theta) = 1.0$ → Same person (0° angle)
- $\cos(\theta) = 0.5$ → 60° angle (threshold ~0.38)
- $\cos(\theta) = 0.0$ → 90° angle (orthogonal)
- $\cos(\theta) = -1.0$ → Opposite direction (180°)

### Implementation in AutoAttendance

**File**: [vision/face_engine.py](../vision/face_engine.py)

```python
class ArcFaceEmbeddingBackend:
    def __init__(self, gpu_providers=None):
        # Initialize InsightFace FaceAnalysis
        self.app = FaceAnalysis(
            name='buffalo_l',           # Model name: large, high accuracy
            providers=gpu_providers,    # GPU or CPU providers
            root=MODEL_PATH
        )
        self.app.prepare(ctx_id=0)      # ctx_id=0 for GPU, -1 for CPU
    
    def generate(self, image_or_path):
        """
        Generate 512-D ArcFace embedding.
        
        Returns:
            embedding: np.ndarray, shape (512,), L2-normalized float32
        """
        det = self.app.get(image_or_path)
        if len(det) == 0:
            return None
        
        embedding = det[0]['embedding']  # 512-D vector
        # Verify L2-normalization
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-6
        return embedding
```

### Why ArcFace for AutoAttendance?

1. **Stability**: 512-D embeddings encode robust face identity across poses and lighting.
2. **Speed**: L2-normalized embeddings enable efficient cosine similarity (one dot product).
3. **Proven accuracy**: 99%+ accuracy on LFW (Labeled Faces in the Wild) benchmark.
4. **Industry standard**: Widely used in production systems.

---

## Cosine Similarity Matching

### Principle

Given a new face embedding $f_{\text{new}}$ and a database of student encodings $\{f_1, f_2, ..., f_n\}$:

$$\text{match}_i = \cos(\theta_i) = f_{\text{new}} \cdot f_i$$

Find the student with highest similarity:

$$\text{match} = \arg\max_i (f_{\text{new}} \cdot f_i)$$

### Threshold Decision

Attendance is marked if:

$$\max_i (f_{\text{new}} \cdot f_i) \geq \tau$$

Where $\tau$ = `RECOGNITION_THRESHOLD` = 0.38 (configurable).

### False Positive vs. False Negative Trade-Off

```
Threshold spectrum:
0.0 ────────── 0.38 (default) ────────── 0.95 ────── 1.0
 │              │                        │          │
 └─ False +ve ──┴─ Balanced ──────────────┴─ False -ve ──┘
   (high)                                 (high)
```

**Raising threshold $\tau$** (e.g., 0.45):
- Reduces false positives (fewer incorrect matches).
- Increases false negatives (legitimate students rejected).

**Lowering threshold $\tau$** (e.g., 0.35):
- Increases false positives (more incorrect matches).
- Reduces false negatives (fewer students rejected).

### Implementation in AutoAttendance

**File**: [vision/recognition.py](../vision/recognition.py)

```python
def match_embeddings(query_embedding, database_encodings):
    """
    Match query embedding against database.
    
    Args:
        query_embedding: np.ndarray, shape (512,)
        database_encodings: dict {reg_no: [emb1, emb2, ...]}
    
    Returns:
        (best_match_reg_no, highest_similarity_score)
    """
    best_match = None
    best_score = -1.0
    
    for reg_no, encodings_list in database_encodings.items():
        for db_emb in encodings_list:
            # Cosine similarity
            similarity = np.dot(query_embedding, db_emb)
            
            if similarity > best_score:
                best_score = similarity
                best_match = reg_no
    
    if best_score >= RECOGNITION_THRESHOLD:
        return (best_match, best_score)
    else:
        return (None, best_score)  # No match
```

---

## Face Alignment & Quality Gating

### Why Alignment Matters

ArcFace models are trained on frontal, aligned 112×112 images. Misaligned crops reduce embedding quality.

### Alignment Strategy

**Method 1: ArcFace 5-Point Alignment** (preferred)

1. Detect 5 landmarks from YuNet: (left-eye, right-eye, nose, left-mouth, right-mouth).
2. Define template landmarks for 112×112 image.
3. Compute affine transformation matrix.
4. Warp crop to 112×112 aligned space.

```python
# Template landmarks (pre-defined for 112×112 image)
template_landmarks = np.array([
    [30.2946, 51.6963],   # Left eye
    [65.5318, 51.5014],   # Right eye
    [48.0252, 71.7366],   # Nose
    [33.5493, 92.3655],   # Left mouth
    [62.7299, 92.2041]    # Right mouth
])

# Detected landmarks from YuNet
detected_landmarks = np.array([
    [x₁, y₁],             # Left eye
    [x₂, y₂],             # Right eye
    [x₃, y₃],             # Nose
    [x₄, y₄],             # Left mouth
    [x₅, y₅]              # Right mouth
])

# Compute affine transformation
M = cv2.getAffineTransform(
    detected_landmarks[:3],    # Use 3 landmarks for affine
    template_landmarks[:3]
)

# Warp to aligned space
aligned = cv2.warpAffine(crop, M, (112, 112))
```

**Method 2: Eye-Center Fallback** (legacy, if landmarks unavailable)

1. Detect eye positions manually.
2. Compute distance between eyes; scale/rotate to template.

### Quality Gating

Before encoding, validate face quality to reject poor samples:

**Criteria**:

1. **Blur Check** (Laplacian variance):
   $$\text{blur_score} = \text{Var}(\nabla^2 I)$$
   where $\nabla^2$ is the Laplacian operator.
   - Reject if $\text{blur_score} < 6.0$ (`BLUR_THRESHOLD`).

2. **Brightness Check**:
   $$\text{brightness} = \text{mean}(I)$$
   - Reject if brightness $< 40$ (too dark) or $> 250$ (overexposed).

3. **Face Size Check**:
   $$\text{area_ratio} = \frac{\text{face_area}}{\text{frame_area}}$$
   - Reject if area_ratio $< 0.005$ (face too small: < 36×36 pixels).

**Implementation**:

```python
def check_face_quality_gate(crop, landmarks):
    """Check if face meets quality criteria."""
    
    # Blur check
    laplacian_var = cv2.Laplacian(crop, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        return False, "BLURRY"
    
    # Brightness check
    brightness = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()
    if brightness < BRIGHTNESS_THRESHOLD or brightness > BRIGHTNESS_MAX:
        return False, "BRIGHTNESS"
    
    # Size check
    h, w = crop.shape[:2]
    if h < MIN_FACE_SIZE_PIXELS or w < MIN_FACE_SIZE_PIXELS:
        return False, "TOO_SMALL"
    
    return True, None
```

---

## Object Tracking (CSRT)

### Why Tracking?

Without tracking, every frame requires running face detection (expensive). Tracking reduces detection frequency:

```
Without tracking:  Detect every frame → ~50ms per frame → 20 FPS
With tracking:     Detect every 6 frames, track in between → ~10ms per frame → 100 FPS
```

### CSRT Algorithm (Correlation Filter + Spatial Regularization)

**Principle**: Given a target bounding box in frame $t$, predict its position in frame $t+1$.

**Correlation Filter**:

$$f(x) = \mathcal{F}^{-1}(\mathcal{F}(K) \odot \mathcal{F}(\alpha))$$

where:
- $K$ = kernel correlation matrix (response map for different target positions)
- $\alpha$ = filter weights
- $\odot$ = Hadamard (element-wise) product
- $\mathcal{F}$ = FFT (Fast Fourier Transform)

**Response Map**:

$$R(x) = \sum_y w(y) K(x - y)$$

Target in frame $t+1$ = $\arg\max_x R(x)$.

### Implementation in AutoAttendance

**File**: [vision/pipeline.py](../vision/pipeline.py)

```python
def create_tracker():
    """
    Create tracker object (CSRT preferred, MIL fallback).
    """
    try:
        tracker = cv2.TrackerCSRT_create()
    except:
        tracker = cv2.TrackerMIL_create()  # Fallback if CSRT unavailable
    
    return tracker

class FaceTrack:
    def __init__(self, bbox, landmarks, tracker):
        self.bbox = bbox
        self.landmarks = landmarks
        self.tracker = tracker
        self.frames_missing = 0
    
    def update(self, frame):
        """Update tracker with new frame."""
        ok, bbox = self.tracker.update(frame)
        
        if ok:
            self.bbox = bbox
            self.frames_missing = 0
        else:
            self.frames_missing += 1
        
        return ok
```

---

## Anti-Spoofing & Liveness Detection

### Challenge

Presentation attacks (printed photos, video replay, masks) attempt to spoof face recognition systems.

### Silent-Face-Anti-Spoofing Approach

**Model Architecture**: CNN with binary classification (Real vs. Spoof).

**Key Insight**: Real faces contain subtle micro-textures and light reflections that photos/videos lack.

### Mathematical Model

```
Input: Face image crop (112×112, RGB)
    ↓
CNN Feature Extraction:
  - Layers 1-2: Low-level textures (edges, patterns)
  - Layers 3-4: Mid-level features (skin texture, light interaction)
  - Layers 5+: High-level semantic features
    ↓
Classifier (final FC layer):
  logit = W·features + b
  probability = σ(logit)
    ↓
Output: P(real) ∈ [0, 1]
```

**Decision**:

$$\text{label} = \begin{cases} 1 & \text{if } P(\text{real}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

### Supplementary Heuristics

Beyond CNN classification, AutoAttendance tracks:

**1. Blink Detection (Eye Aspect Ratio)**:

```
EAR = (‖P₂ - P₆‖ + ‖P₃ - P₅‖) / (2 × ‖P₁ - P₄‖)
```

where $P_1, ..., P_6$ are eye landmark coordinates.

- EAR > 0.2: Eyes open
- EAR < 0.1: Eyes closed (blink)
- Blinking is a sign of a live face (printed photos can't blink).

**2. Motion Detection**:

Optical flow heuristics detect subtle head movement, confirming genuine presence.

**3. Frame-Level Heuristics**:

- **Brightness variance**: Real faces show natural shadows; screenshots are uniform.
- **Saturation analysis**: Real skin has natural color; photos may be oversaturated or desaturated.
- **Frequency content**: Real faces have richer texture than screen displays.

### Implementation in AutoAttendance

**File**: [vision/anti_spoofing.py](../vision/anti_spoofing.py)

```python
def check_liveness(frame):
    """
    Check if face is live (real) or spoof (presentation attack).
    
    Returns:
        (label, confidence)
        label: 1=real, 0=spoof, -1=error
        confidence: [0, 1]
    """
    # CNN classification
    label, confidence = _predictor.predict(frame)
    
    # Supplementary blink detection
    if label == 1:
        ear = compute_ear_from_landmarks(frame)
        if ear < 0.05:  # Recent blink detected
            confidence = min(1.0, confidence + 0.1)
    
    return label, confidence
```

---

## Motion Detection & Optical Flow

### Principle

Motion between consecutive frames indicates a live person.

### Optical Flow (Lucas-Kanade Method)

Tracks movement of features between frames:

$$I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)$$

Differentiating:

$$I_x u + I_y v + I_t = 0$$

Where:
- $I_x, I_y$ = spatial gradients
- $I_t$ = temporal gradient
- $(u, v)$ = flow vector (pixel displacement)

Solving via least-squares (Lucas-Kanade):

$$\begin{bmatrix} u \\ v \end{bmatrix} = \left( \sum_i \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} \right)^{-1} \sum_i \begin{bmatrix} -I_x I_t \\ -I_y I_t \end{bmatrix}$$

### Implementation in AutoAttendance

**File**: [vision/pipeline.py](../vision/pipeline.py)

```python
def detect_motion(frame_t, frame_t_minus_1):
    """
    Detect motion between consecutive frames.
    
    Returns:
        (motion_detected: bool, gray_frame)
    """
    gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(frame_t_minus_1, cv2.COLOR_BGR2GRAY)
    
    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_t, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, n8=True, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # Compute magnitude of flow
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Motion detected if average flow magnitude > threshold
    motion_detected = mag.mean() > MOTION_THRESHOLD
    
    return motion_detected, gray_t
```

---

## Multi-Frame Confirmation

### Motivation

A single frame match can be noisy (lighting artifacts, partial faces). Multiple confirmations reduce false positives.

### Voting Scheme

For a track, collect recognition results over consecutive frames:

```
Frame 1: Student A (confidence 0.92) 
Frame 2: No match (Laplacian blur)    ✗
Frame 3: Student A (confidence 0.89)  
Frame 4: Student B (confidence 0.50)  ✗
Frame 5: Student A (confidence 0.91)  
```

**Majority Vote**:

$$\text{majority} = \begin{cases} \text{Student A (3 votes)} & \text{if } 3 \geq 3 \\ \text{Reject} & \text{otherwise} \end{cases}$$

**Configuration**:
- `RECOGNITION_CONFIRM_FRAMES=2`: Require match in at least 2 of last 5 frames.
- `LIVENESS_HISTORY_SIZE=5`: Rolling buffer of 5 liveness decisions.

### Implementation

```python
class FaceTrack:
    def __init__(self):
        self.identity_votes = {}      # {student_id: count}
        self.liveness_votes = []      # [1, 1, 0, 1, ...]
    
    def record_result(self, student_id, liveness_label):
        """Record recognition and liveness for this frame."""
        self.identity_votes[student_id] = \
            self.identity_votes.get(student_id, 0) + 1
        
        self.liveness_votes.append(liveness_label)
        if len(self.liveness_votes) > LIVENESS_HISTORY_SIZE:
            self.liveness_votes.pop(0)
    
    def should_confirm(self):
        """Check if track meets confirmation criteria."""
        # Majority identity
        if not self.identity_votes:
            return False, None
        
        best_student = max(self.identity_votes, 
                          key=self.identity_votes.get)
        vote_count = self.identity_votes[best_student]
        
        # Check thresholds
        if vote_count < RECOGNITION_CONFIRM_FRAMES:
            return False, None
        
        # Check liveness
        liveness_mean = np.mean(self.liveness_votes)
        if liveness_mean < LIVENESS_CONFIDENCE_THRESHOLD:
            return False, None
        
        return True, best_student
```

---

## Threshold Calibration & Trade-Offs

### Parameter Sensitivity

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `RECOGNITION_THRESHOLD` | 0.38 | 0.30–0.50 | Match strictness |
| `LIVENESS_CONFIDENCE_THRESHOLD` | 0.55 | 0.40–0.75 | Liveness gate |
| `BLUR_THRESHOLD` | 6.0 | 2.0–10.0 | Quality control |
| `RECOGNITION_CONFIRM_FRAMES` | 2 | 1–5 | Confirmation voting |

### Calibration Methodology

**1. Collect Validation Data**:
- 50–100 genuine students.
- 50–100 presentation attack samples (printed photos, videos).

**2. Sweep Parameters**:

```python
for threshold in [0.30, 0.35, 0.38, 0.40, 0.45]:
    for confirm_frames in [1, 2, 3]:
        # Run pipeline on validation set
        genuine_accuracy = measure_genuine_acceptance_rate()
        spoof_detection_rate = measure_spoof_rejection_rate()
        
        # Log metrics
        log_metric(threshold, confirm_frames, 
                  genuine_accuracy, spoof_detection_rate)
```

**3. Select Optimal**:

Choose parameters maximizing:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Deployment Trade-Offs

```
Strict (RECOGNITION_THRESHOLD=0.50):
├─ Pro: Very low false positive rate (spurious matches)
└─ Con: Some legitimate students rejected (false negative)

Balanced (RECOGNITION_THRESHOLD=0.38, CONFIRM_FRAMES=2):
├─ Pro: Good false positive / false negative balance
└─ Con: Moderate computational cost

Lenient (RECOGNITION_THRESHOLD=0.30, CONFIRM_FRAMES=1):
├─ Pro: Faster, fewer legitimate rejects
└─ Con: More false positives (higher fraud risk)
```

---

## Summary

AutoAttendance combines several mature computer vision techniques:

1. **Detection**: YuNet for real-time, lightweight face detection.
2. **Embedding**: ArcFace for robust, L2-normalized 512-D vectors.
3. **Matching**: Cosine similarity with threshold.
4. **Anti-spoofing**: CNN + supplementary heuristics.
5. **Tracking**: CSRT for low-latency frame-to-frame continuity.
6. **Confirmation**: Multi-frame voting reduces false positives.

Each component is tuned for accuracy, speed, and robustness. See [PIPELINE.md](PIPELINE.md) for how these components integrate in real-time processing.

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0

