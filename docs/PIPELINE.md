# Processing Pipeline & Real-Time Loop

## Table of Contents

1. [Overview](#overview)
2. [Frame Capture & Preprocessing](#frame-capture--preprocessing)
3. [Motion Detection & Adaptive Sampling](#motion-detection--adaptive-sampling)
4. [Face Detection & Landmark Extraction](#face-detection--landmark-extraction)
5. [Track Association & Management](#track-association--management)
6. [Face Alignment & Quality Validation](#face-alignment--quality-validation)
7. [Embedding Generation](#embedding-generation)
8. [Face Recognition & Matching](#face-recognition--matching)
9. [Liveness & Anti-Spoofing](#liveness--anti-spoofing)
10. [Multi-Frame Confirmation](#multi-frame-confirmation)
11. [Attendance Marking & Event Emission](#attendance-marking--event-emission)
12. [Performance Optimization](#performance-optimization)

---

## Overview

The camera pipeline orchestrates real-time face detection, tracking, recognition, and liveness verification. It runs in a dedicated thread, processing frames at 30 FPS or higher.

### High-Level Flow

```
┌─────────────────────────────────────────────┐
│ 1. CAPTURE FRAME                            │
│    OpenCV VideoCapture → BGR frame          │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│ 2. MOTION DETECTION (every DETECTION_INTERVAL)
│    Optical flow → motion_detected?          │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
      NO  │             YES │
          │                 │
          ▼                 ▼
    UPDATE TRACKERS   RUN YUNET DETECTION
    (CSRT.update)     Extract landmarks
                           │
                           ▼
                  ASSOCIATE DETECTIONS
                  Match to existing tracks
                           │
                ┌──────────┴──────────┐
                │                     │
          NEW TRACK             MATCHED TRACK
                │                     │
                └──────────┬──────────┘
                           ▼
                  FOR EACH TRACK:
                  ├─ Extract face chip
                  ├─ Quality gate (blur, brightness, size)
                  ├─ Align face (ArcFace 5-point)
                  ├─ Generate embedding (ArcFace)
                  ├─ Recognize (cosine similarity)
                  ├─ Liveness check (Silent-Face)
                  ├─ Multi-frame voting
                  └─ Mark attendance (if confirmed)
                           │
                           ▼
                  EMIT SOCKETIO EVENTS
                  Update admin UI
```

---

## Frame Capture & Preprocessing

### Capture

```python
# In camera/camera.py
import cv2

cap = cv2.VideoCapture(camera_id)  # 0 for default webcam, IP for network camera

while True:
    ret, frame = cap.read()  # Capture frame
    
    if not ret:
        logger.error("Failed to capture frame")
        break
```

### Resizing (Performance Optimization)

Large frames slow down detection. AutoAttendance resizes to improve speed:

```python
# Define resize factor
frame_width = int(frame.shape[1] * FRAME_RESIZE_FACTOR)
frame_height = int(frame.shape[0] * FRAME_RESIZE_FACTOR)

# Resize for processing (detector runs at reduced resolution)
frame_small = cv2.resize(
    frame, (frame_width, frame_height),
    interpolation=cv2.INTER_LINEAR
)

# Keep original for display and high-quality encoding
frame_display = frame.copy()
```

**Trade-Off**:
- Smaller frames → faster detection (50ms → 20ms).
- Loss of small-face detection (faces < 36×36 pixels may be missed).

### Default Values

- `FRAME_RESIZE_FACTOR=0.25`: Resize to 25% of original size.
- `FRAME_PROCESS_WIDTH=512`: Additional resize to 512-pixel width if needed.

---

## Motion Detection & Adaptive Sampling

### Why Motion Detection?

Detecting faces every frame is expensive (~50ms per frame). Motion-gated detection reduces computational load:

```
Scenario: Quiet classroom (no student movement)
├─ Without motion gating: Detect every frame → 50ms × N frames = slow
└─ With motion gating: Skip detection in still frames → only ~10ms

Scenario: Students entering room (high motion)
├─ Without motion gating: Detect every frame (acceptable load)
└─ With motion gating: More frequent detection (responsive)
```

### Motion Detection Logic

```python
# In vision/pipeline.py
import cv2

def detect_motion(frame_t, frame_t_minus_1, threshold=1.5):
    """
    Detect motion using optical flow.
    
    Args:
        frame_t: Current frame (grayscale)
        frame_t_minus_1: Previous frame (grayscale)
        threshold: Motion magnitude threshold
    
    Returns:
        motion_detected: bool
    """
    # Compute dense optical flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(
        frame_t_minus_1, frame_t,
        None,
        pyr_scale=0.5,      # Image pyramid scale
        levels=3,           # Pyramid levels
        winsize=15,         # Window size
        iterations=3,       # Iterations at each level
        n8=True,            # 8-neighborhood
        poly_n=5,           # Polynomial expansion kernel size
        poly_sigma=1.2,     # Standard deviation for Gaussian
        flags=0
    )
    
    # Compute magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Motion detected if average magnitude exceeds threshold
    avg_motion = mag.mean()
    return avg_motion > threshold
```

### Adaptive Sampling

```python
# In camera/camera.py
while True:
    ret, frame = cap.read()
    
    # Motion detection (every DETECTION_INTERVAL frames)
    if frame_count % DETECTION_INTERVAL == 0:
        motion_detected, gray = detect_motion(frame_gray, prev_frame_gray)
    
    if motion_detected or frame_count % DETECTION_INTERVAL == 0:
        # Run YuNet detection
        detections = detector.detect(frame)
    else:
        # Skip detection; just update existing trackers
        detections = []
    
    prev_frame_gray = frame_gray.copy()
    frame_count += 1
```

### Configuration

- `DETECTION_INTERVAL=6`: Run full detection every 6 frames (5 tracking-only frames between).
- `MOTION_THRESHOLD=1.5`: Optical flow magnitude threshold.

---

## Face Detection & Landmark Extraction

### YuNet Detection

```python
# In vision/pipeline.py
import cv2

def detect_faces_yunet(frame):
    """
    Detect faces using YuNet ONNX model.
    
    Args:
        frame: BGR image (H×W×3)
    
    Returns:
        List of detections:
        [
            (x, y, w, h, confidence, landmarks_5points),
            ...
        ]
    """
    detector.setInputSize((320, 320))
    detector.setScoreThreshold(YUNET_SCORE_THRESHOLD)  # 0.62
    
    results = detector.detect(frame)
    
    return results
```

### Output Format

For each detection:

```python
detection = {
    'bbox': (x, y, w, h),                    # Bounding box (top-left, width, height)
    'confidence': 0.95,                      # Detection confidence [0, 1]
    'landmarks': [                           # 5-point landmarks
        (x_left_eye, y_left_eye),
        (x_right_eye, y_right_eye),
        (x_nose, y_nose),
        (x_left_mouth, y_left_mouth),
        (x_right_mouth, y_right_mouth)
    ]
}
```

### Coordinate Transformation (if frame was resized)

If detection ran on resized frame, scale coordinates back to original:

```python
scale_x = original_width / resized_width
scale_y = original_height / resized_height

x_orig = x_resized * scale_x
y_orig = y_resized * scale_y
w_orig = w_resized * scale_x
h_orig = h_resized * scale_y
```

---

## Track Association & Management

### Goal

Match detections to existing tracks to enable continuous tracking and reduce re-identification overhead.

### Association Algorithm

```python
# In vision/pipeline.py
from scipy.spatial.distance import cdist

def detect_and_associate_detailed(detections, existing_tracks, iou_threshold=0.5):
    """
    Associate detections with existing tracks.
    
    Returns:
        new_tracks: Detections not matched to existing tracks
        matched_indices: List of (track_idx, detection_idx) matches
    """
    if not existing_tracks or not detections:
        return detections, []
    
    # Compute IoU (Intersection over Union) between each track and detection
    ious = np.zeros((len(existing_tracks), len(detections)))
    
    for i, track in enumerate(existing_tracks):
        for j, detection in enumerate(detections):
            iou = compute_iou(track.bbox, detection['bbox'])
            ious[i, j] = iou
    
    # Greedy matching: highest IoU first
    matches = []
    matched_tracks = set()
    matched_detections = set()
    
    while True:
        max_iou = 0
        best_match = None
        
        for i in range(len(existing_tracks)):
            if i in matched_tracks:
                continue
            for j in range(len(detections)):
                if j in matched_detections:
                    continue
                if ious[i, j] > max_iou:
                    max_iou = ious[i, j]
                    best_match = (i, j)
        
        if best_match is None or max_iou < iou_threshold:
            break
        
        i, j = best_match
        matches.append((i, j))
        matched_tracks.add(i)
        matched_detections.add(j)
    
    # Unmatched detections become new tracks
    new_detections = [det for idx, det in enumerate(detections) 
                      if idx not in matched_detections]
    
    return new_detections, matches
```

### Track Management

```python
class FaceTrack:
    """Represents a single tracked face."""
    
    def __init__(self, detection, tracker):
        self.bbox = detection['bbox']
        self.landmarks = detection['landmarks']
        self.confidence = detection['confidence']
        self.tracker = tracker           # CSRT or KCF
        
        # Tracking state
        self.frames_missing = 0
        self.frames_since_recognized = 0
        
        # Identity and liveness state
        self.identity_votes = {}         # {student_id: count}
        self.liveness_votes = []         # [1, 0, 1, 1, ...]
        self.recognition_cache = None    # (student_id, confidence, timestamp)
        self.cache_expires_at = None
    
    def update(self, frame):
        """Update tracker with new frame."""
        ok, updated_bbox = self.tracker.update(frame)
        
        if ok:
            self.bbox = updated_bbox
            self.frames_missing = 0
        else:
            self.frames_missing += 1
        
        return ok
    
    def is_expired(self, max_frames_missing=30):
        """Check if track should be removed."""
        return self.frames_missing > max_frames_missing

# In camera.py
self._tracks = []  # List of FaceTrack objects

# Purge expired tracks
self._tracks = [t for t in self._tracks if not t.is_expired()]
```

---

## Face Alignment & Quality Validation

### Extraction

Extract face crop from frame using bounding box:

```python
x, y, w, h = track.bbox
face_crop = frame[int(y):int(y+h), int(x):int(x+w)]
```

### Quality Gates

Before proceeding to encoding, validate quality:

```python
# In vision/recognition.py
def check_face_quality_gate(crop, landmarks):
    """
    Check if face crop meets quality criteria.
    
    Returns:
        (is_valid: bool, reason: str or None)
    """
    h, w = crop.shape[:2]
    
    # 1. Size check
    min_size = MIN_FACE_SIZE_PIXELS  # 36
    if h < min_size or w < min_size:
        return False, "FACE_TOO_SMALL"
    
    # 2. Blur check (Laplacian variance)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < BLUR_THRESHOLD:  # 6.0
        return False, "BLURRY"
    
    # 3. Brightness check
    brightness = gray.mean()
    if brightness < BRIGHTNESS_THRESHOLD or brightness > BRIGHTNESS_MAX:
        return False, "BRIGHTNESS"
    
    # 4. Landmarks validity check
    if landmarks is None or len(landmarks) < 5:
        return False, "NO_LANDMARKS"
    
    return True, None
```

### Alignment to 112×112

Face recognition models (ArcFace) expect 112×112 aligned images.

**ArcFace 5-Point Alignment**:

```python
# In vision/recognition.py
def align_face_arcface(crop, landmarks):
    """
    Align face using 5-point landmarks to 112×112 canonical orientation.
    
    Args:
        crop: Face crop (BGR image)
        landmarks: List of 5 (x, y) tuples
    
    Returns:
        aligned_crop: 112×112 aligned face image (BGR)
    """
    # Template landmarks (pre-defined for 112×112 image)
    template_landmarks = np.array([
        [30.2946, 51.6963],   # Left eye
        [65.5318, 51.5014],   # Right eye
        [48.0252, 71.7366],   # Nose (center)
        [33.5493, 92.3655],   # Left mouth
        [62.7299, 92.2041]    # Right mouth
    ], dtype=np.float32)
    
    # Detected landmarks
    detected_landmarks = np.array(landmarks, dtype=np.float32)
    
    # Use first 3 landmarks for affine transformation
    # (More robust than all 5; eye-based alignment)
    M = cv2.getAffineTransform(
        detected_landmarks[:3],
        template_landmarks[:3]
    )
    
    # Warp to 112×112 aligned space
    aligned = cv2.warpAffine(
        crop, M, (112, 112),
        borderMode=cv2.BORDER_REFLECT_101
    )
    
    return aligned
```

---

## Embedding Generation

### Pre-Alignment Optimization

Store aligned crop in cache to avoid re-alignment:

```python
encoding = recognize_pipeline.encode_face_with_reason(
    face_crop,
    landmarks=track.landmarks
)
```

### ArcFace Backend

```python
# In vision/face_engine.py
class ArcFaceEmbeddingBackend:
    def generate(self, image_or_path):
        """
        Generate ArcFace embedding for image.
        
        Args:
            image_or_path: PIL Image or path to image
        
        Returns:
            embedding: np.ndarray of shape (512,), L2-normalized
        """
        # FaceAnalysis automatically detects, aligns, and encodes
        det = self.app.get(image_or_path)
        
        if len(det) == 0:
            return None
        
        # Extract embedding (already L2-normalized by InsightFace)
        embedding = det[0].embedding
        
        return embedding
    
    def generate_from_aligned(self, aligned_crop):
        """
        Generate embedding from pre-aligned 112×112 crop (faster).
        
        Skips detection and alignment; runs only recognition model.
        
        Returns:
            embedding: np.ndarray of shape (512,)
        """
        # Pass to recognition model only
        embedding = self.app.get_from_aligned(aligned_crop)
        return embedding
```

### Caching

Cache generated embeddings for quick lookup:

```python
# In camera.py
def process_frame(self, frame):
    for track in self._tracks:
        # Check recognition cache first
        if track.recognition_cache and track.cache_expires_at > time.time():
            # Use cached result
            continue
        
        # Generate embedding
        encoding = self.face_engine.generate(face_crop)
        
        # Store in cache with TTL
        track.recognition_cache = encoding
        track.cache_expires_at = time.time() + RECOGNITION_TRACK_CACHE_TTL_SECONDS
```

---

## Face Recognition & Matching

### Cosine Similarity Matching

```python
# In vision/recognition.py
def match_embeddings(query_embedding, database_encodings):
    """
    Match query embedding against student database.
    
    Args:
        query_embedding: np.ndarray of shape (512,), L2-normalized
        database_encodings: dict {reg_no: [enc1, enc2, ...]}
    
    Returns:
        (best_match_reg_no, best_score) or (None, -1.0)
    """
    best_match = None
    best_score = -1.0
    
    for reg_no, encodings_list in database_encodings.items():
        for db_enc in encodings_list:
            # Cosine similarity = dot product (both L2-normalized)
            similarity = np.dot(query_embedding, db_enc)
            
            if similarity > best_score:
                best_score = similarity
                best_match = reg_no
    
    # Apply threshold
    if best_score >= RECOGNITION_THRESHOLD:  # 0.38
        return best_match, best_score
    else:
        return None, best_score
```

### Cache Optimization

Load all student encodings into RAM on startup:

```python
# In camera/camera.py (or core/database.py)
def load_encoding_cache():
    """Load all student encodings into RAM."""
    students_db = database.get_all_students()
    
    encoding_cache = {}
    for student in students_db:
        reg_no = student['reg_no']
        encodings = student.get('encodings', [])
        
        encoding_cache[reg_no] = [
            np.frombuffer(enc_bytes, dtype=np.float32) 
            for enc_bytes in encodings
        ]
    
    return encoding_cache
```

---

## Liveness & Anti-Spoofing

### Silent-Face Model Inference

```python
# In vision/anti_spoofing.py
def check_liveness(frame, landmarks=None):
    """
    Check if face is live (real) or spoof (presentation attack).
    
    Args:
        frame: Full frame (BGR) or face crop
        landmarks: Optional 5-point landmarks
    
    Returns:
        (label, confidence)
        label: 1=real, 0=spoof, 2=other_attack, -1=error
        confidence: [0, 1]
    """
    if not _predictor or not _cropper:
        # Models not loaded; return permissive default
        return 1, 1.0
    
    try:
        # Run CNN classification
        label = _predictor.predict(frame)
        
        # Extract confidence
        confidence = _predictor.confidence(frame)
        
        # Supplementary blink detection
        if landmarks and label == 1:
            ear = compute_ear_from_5point(landmarks)
            if ear < 0.05:  # Eyes likely closed (blink)
                confidence = min(1.0, confidence + 0.1)
        
        return label, confidence
    
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return -1, 0.0
```

### Frame-Level Heuristics

Supplement CNN with texture analysis:

```python
def analyze_liveness_frame(crop):
    """
    Analyze frame-level features for liveness assessment.
    
    Returns:
        heuristic_score: [0, 1]
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Feature 1: Laplacian variance (texture richness)
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var() / 10.0
    texture_score = min(1.0, texture_score)  # Normalize
    
    # Feature 2: Saturation (real skin has natural color)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean() / 255.0
    saturation_score = saturation if 0.2 < saturation < 0.8 else 0.3
    
    # Feature 3: Brightness consistency (prints may have artifacts)
    brightness_variance = gray.std()
    brightness_score = min(1.0, brightness_variance / 50.0)
    
    # Combine heuristics
    heuristic_score = (texture_score + saturation_score + brightness_score) / 3.0
    
    return heuristic_score
```

---

## Multi-Frame Confirmation

### Voting & Confirmation

```python
def should_confirm_attendance(track, CONFIRM_THRESHOLD=2, LIVENESS_THRESHOLD=0.55):
    """
    Check if track meets attendance confirmation criteria.
    
    Criteria:
    1. At least CONFIRM_THRESHOLD recognitions of same student
    2. Liveness mean >= LIVENESS_THRESHOLD over history
    
    Returns:
        (should_confirm: bool, student_id: str or None)
    """
    if not track.identity_votes:
        return False, None
    
    # Find majority identity
    best_student = max(track.identity_votes, key=track.identity_votes.get)
    vote_count = track.identity_votes[best_student]
    
    # Check recognition threshold
    if vote_count < CONFIRM_THRESHOLD:
        return False, None
    
    # Check liveness
    if len(track.liveness_votes) == 0:
        return False, None
    
    liveness_mean = np.mean(track.liveness_votes)
    if liveness_mean < LIVENESS_THRESHOLD:
        return False, None
    
    return True, best_student
```

### Update Track State

```python
# In camera.py
for track in self._tracks:
    # ... (detection, alignment, embedding, recognition, liveness)
    
    # Record votes
    if recognized_student_id:
        track.identity_votes[recognized_student_id] = \
            track.identity_votes.get(recognized_student_id, 0) + 1
    
    track.liveness_votes.append(liveness_label)
    if len(track.liveness_votes) > LIVENESS_HISTORY_SIZE:
        track.liveness_votes.pop(0)
    
    # Check confirmation
    should_confirm, student_id = should_confirm_attendance(track)
    
    if should_confirm:
        mark_attendance(student_id, confidence, track)
        track.identity_votes.clear()  # Reset votes after marking
```

---

## Attendance Marking & Event Emission

### Database Update

```python
# In core/database.py
def mark_attendance(student_id, confidence, session_id, camera_id):
    """
    Mark attendance for student (only once per day per camera session).
    
    Enforces uniqueness via MongoDB unique index: {student_id, date}
    """
    today = datetime.utcnow().date()
    
    attendance_record = {
        'student_id': student_id,
        'reg_no': database.get_student_reg_no(student_id),
        'date': today,
        'status': 'Present',
        'marked_at': datetime.utcnow(),
        'confidence': confidence,
        'session_id': session_id,
        'camera_id': camera_id,
        'verified': True,
        'notes': ''
    }
    
    # Upsert with unique constraint
    db.attendance.update_one(
        {
            'student_id': student_id,
            'date': today
        },
        {'$set': attendance_record},
        upsert=True
    )
```

### SocketIO Event Emission

Notify admin UI in real-time:

```python
# In camera.py (with SocketIO injected)
def emit_attendance_event(self, student_id, confidence, track_id):
    """Emit real-time attendance event to admin dashboard."""
    
    student_name = self.database.get_student_name(student_id)
    
    event_data = {
        'student_id': str(student_id),
        'student_name': student_name,
        'confidence': float(confidence),
        'timestamp': datetime.utcnow().isoformat(),
        'camera_id': self.camera_id,
        'track_id': track_id
    }
    
    self.socketio.emit(
        'attendance_event',
        event_data,
        namespace='/admin'
    )
    
    logger.info(f"Attendance marked: {student_name} ({confidence:.2f})")
```

### Admin Dashboard Update

WebSocket listener updates the admin UI:

```javascript
// In admin dashboard (JavaScript)
socket.on('attendance_event', function(data) {
    // Add attendance record to table
    const row = `
        <tr>
            <td>${data.timestamp}</td>
            <td>${data.student_name}</td>
            <td>${(data.confidence * 100).toFixed(1)}%</td>
        </tr>
    `;
    
    document.querySelector('#attendance_table').insertAdjacentHTML('beforeend', row);
});
```

---

## Performance Optimization

### Bottleneck Analysis

```
Operation                    | Latency (CPU)  | Latency (GPU)
YuNet detection              | ~50ms          | ~20ms
ArcFace encoding             | ~30ms          | ~10ms
CSRT tracking (5 tracks)     | ~20ms          | ~5ms
Silent-Face liveness         | ~15ms          | ~5ms
Cosine similarity (1000 students) | ~2ms      | ~1ms
                             |
Total per frame              | ~120ms         | ~40ms
FPS achievable               | ~8 FPS         | ~25 FPS
```

### Optimization Strategies

**1. Motion-Gated Detection**
```
Skip detection in static frames
Impact: 80% reduction in detection latency during idle periods
```

**2. Frame Resize**
```
Process at lower resolution (e.g., 512-pixel width)
Impact: YuNet detection 30% faster; minimal accuracy loss
```

**3. Encoding Cache**
```
Load all student encodings into RAM at startup
Impact: Cosine similarity O(n) → O(1) effective time
```

**4. Track Reuse**
```
Avoid re-detecting same face; use CSRT tracker
Impact: 5 tracked faces = 5 tracks × ~4ms each = 20ms vs. 250ms detection
```

**5. GPU Acceleration**
```
Use NVIDIA GPU (CUDA) for YuNet and ArcFace
Impact: 3–5× speedup vs. CPU
```

**6. KCF Fallback** (optional, less accurate)
```
Use KCF tracker instead of CSRT for speed
Impact: 2× faster tracking, reduced accuracy
Configuration: PERF_USE_KCF_TRACKER=1
```

### Memory Usage

```
Component                           | Memory (per 1000 students)
Student encodings (512-D float32)   | ~2 MB
Track objects (max 5)               | ~50 KB
Frame buffer (5 MJPEG frames)       | ~10 MB
Models (YuNet, ArcFace)             | ~800 MB
Total                               | ~820 MB
```

---

## End-to-End Example

### Scenario: Student Alice Enters Classroom

**Frame 1–5: Motion detected, YuNet runs**
```
[Frame 1] Detect Alice (conf 0.92) → Create track_1
[Frame 2] Track track_1 (CSRT)
[Frame 3] Motion gated → Skip detection, track track_1
[Frame 4] Motion gated → Skip detection, track track_1
[Frame 5] Detect Alice (conf 0.91) → Match track_1, update bbox
```

**Frame 6–10: Alignment & Embedding**
```
[Frame 6] 
  └─ Extract crop (112×112 aligned)
  └─ Quality gate: PASS (not blurry, good brightness)
  └─ Encode (ArcFace) → [0.123, 0.456, ..., 0.789]

[Frame 7]
  └─ Cosine similarity vs. database
  └─ Best match: Alice (0.92 confidence)
  └─ Liveness: Silent-Face → REAL (0.88 confidence)
  └─ Record vote: {Alice: 1}, liveness_votes=[1]

[Frame 8]
  └─ (repeat detection, alignment, encoding, recognition)
  └─ Best match: Alice (0.89 confidence)
  └─ Liveness: REAL (0.90)
  └─ Record vote: {Alice: 2}, liveness_votes=[1, 1]

[Frame 9–10] (repeat)
```

**Frame 11: Confirmation**
```
[Frame 11] Check confirmation:
  ├─ Identity votes: {Alice: 3} ≥ CONFIRM_THRESHOLD (2) ✓
  ├─ Liveness mean: (1 + 1 + 1 + ...) / N ≥ 0.55 ✓
  └─ CONFIRMED: Mark attendance for Alice

Emit SocketIO event:
  {
    "student_id": "...",
    "student_name": "Alice",
    "confidence": 0.90,
    "timestamp": "2024-09-15T09:05:23.123Z"
  }

Admin dashboard updates in real-time ✓
```

---

## Summary

The pipeline integrates multiple CV techniques into a cohesive real-time system:

1. **Efficiency**: Motion-gating, frame resizing, and caching reduce latency.
2. **Accuracy**: Multi-frame voting and anti-spoofing minimize false positives.
3. **Resilience**: Graceful degradation if models fail.
4. **Scalability**: GPU acceleration and batch processing support large deployments.

Next steps:
- See [DATABASE.md](DATABASE.md) for attendance persistence.
- See [BACKEND.md](BACKEND.md) for Flask API design.
- See [DEPLOYMENT.md](DEPLOYMENT.md) for production setup.

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0
