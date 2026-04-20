# Code Walkthrough: AutoAttendance Codebase Architecture

Complete module-by-module explanation of the AutoAttendance codebase, key classes, functions, data flow, and integration patterns.

---

## Overview

**Project Structure**: Flask-based web application with ML pipeline

```
attendance_system/
├── recognition/           # Face detection, embedding, matching
├── anti_spoofing/         # Liveness detection
├── vision/                # Pipeline orchestration
├── core/                  # Services (DB, config, auth)
├── admin_app/             # Admin Flask application
├── student_app/           # Student Flask application
├── camera/                # Camera interface
├── models/                # Pre-trained ONNX models
├── scripts/               # Utility scripts
└── templates/ & static/   # Web assets
```

**Data Flow**: 
```
Camera Input → Vision Pipeline → Recognition → Anti-Spoofing → 
Database → Web Application → HTTP Response/SocketIO
```

---

## Module 1: recognition/ — Face Detection & Embedding

**Purpose**: Detect faces, extract embeddings, and match against database.

**Files**:
- `detector.py` — Face detection (YuNet)
- `aligner.py` — Face alignment (5-point landmarks)
- `embedder.py` — Embedding generation (ArcFace)
- `matcher.py` — Two-stage matching (FAISS + detailed)
- `tracker.py` — Temporal tracking (CSRT)
- `pipeline.py` — Orchestration
- `config.py` — Detection parameters

### detector.py — Face Detection

```python
import cv2
import numpy as np

class FaceDetector:
    """YuNet face detector with ONNX runtime"""
    
    def __init__(self, model_path):
        # Load YuNet model
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",  # backend (empty = auto-select)
            (320, 240),  # input size (adaptive)
            0.6,  # score threshold
            0.3,  # NMS threshold
            5000  # top_k
        )
        
    def detect(self, frame):
        """
        Detect faces in frame
        
        Args:
            frame: np.array (H, W, 3) RGB image
        
        Returns:
            detections: np.array of shape (N, 15)
                N = number of faces
                15 values: x, y, w, h, + 5 landmarks (x,y each) + confidence
        """
        # Set input size based on frame
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        
        # Detect
        detections = self.detector.detect(frame)
        
        return detections[1]  # Return faces (drop success flag)
    
    def get_boxes(self, detections):
        """Extract bounding boxes from detections"""
        if detections is None or len(detections) == 0:
            return []
        
        boxes = []
        for det in detections:
            x, y, w, h = det[:4].astype(int)
            confidence = det[-1]
            
            # Filter by confidence
            if confidence > 0.5:
                boxes.append({
                    'bbox': (x, y, x + w, y + h),
                    'confidence': confidence,
                    'landmarks': det[4:14].reshape(5, 2)  # 5 landmarks
                })
        
        return boxes
```

**Key Constants** (from config.py):
- `YuNet_MODEL_PATH` = "models/face_detection_yunet_2023mar.onnx"
- `DETECTION_THRESHOLD` = 0.5 (confidence)
- `NMS_THRESHOLD` = 0.3 (non-maximum suppression)
- `TARGET_SIZE` = (320, 240) (input size)

**Performance**: ~30 FPS on CPU (33ms per frame)

### aligner.py — Face Alignment

```python
import cv2
import numpy as np

class FaceAligner:
    """Align face to canonical orientation (112x112)"""
    
    # Reference face landmarks (standard 5-point)
    REFERENCE_FACIAL_POINTS = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    
    OUTPUT_SIZE = (112, 112)  # ArcFace standard
    
    def __init__(self):
        pass
    
    def align(self, frame, landmarks):
        """
        Align face using 5-point landmarks
        
        Args:
            frame: Input image
            landmarks: np.array of shape (5, 2), 5 face landmarks
        
        Returns:
            aligned_face: np.array (112, 112, 3) aligned face
            affine_matrix: np.array (2, 3) transformation matrix
        """
        # Estimate affine transformation
        affine_matrix = cv2.getAffineTransform(
            landmarks[:3].astype(np.float32),
            self.REFERENCE_FACIAL_POINTS[:3]
        )
        
        # Apply affine transformation
        aligned = cv2.warpAffine(
            frame,
            affine_matrix,
            self.OUTPUT_SIZE,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        return aligned, affine_matrix
    
    def get_alignment_quality(self, landmarks):
        """
        Estimate alignment quality (0-1)
        Based on landmarks variance and position
        """
        # Calculate spread of landmarks
        center = landmarks.mean(axis=0)
        distances = np.linalg.norm(landmarks - center, axis=1)
        variance = distances.std()
        
        # Normalize to 0-1 (higher = better)
        quality = min(1.0, variance / 30.0)
        return quality
```

**Output**: 112×112 canonical face image (ArcFace standard)

### embedder.py — Embedding Generation

```python
import onnxruntime as ort
import numpy as np

class ArcFaceEmbedder:
    """Generate 512-D embeddings using ArcFace ResNet-100"""
    
    def __init__(self, model_path):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def embed(self, aligned_face):
        """
        Generate embedding for aligned face
        
        Args:
            aligned_face: np.array (112, 112, 3), values in [0, 255]
        
        Returns:
            embedding: np.array (512,), L2-normalized
        """
        # Preprocess
        # 1. Resize to (112, 112)
        if aligned_face.shape != (112, 112, 3):
            aligned_face = cv2.resize(aligned_face, (112, 112))
        
        # 2. Normalize: (X - mean) / std
        aligned_face = aligned_face.astype(np.float32)
        aligned_face = (aligned_face - 127.5) / 128.0
        
        # 3. Convert to NCHW format
        input_data = np.transpose(aligned_face, (2, 0, 1))
        input_data = np.expand_dims(input_data, 0)  # Add batch dimension
        
        # Inference
        embedding = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )[0]
        
        # L2 normalize
        embedding = embedding.astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding, ord=2)
        
        return embedding.flatten()
```

**Output**: 512-D L2-normalized embedding

**Latency**: ~18ms per face

### matcher.py — Two-Stage Matching

```python
import faiss
import numpy as np

class TwoStageMatcher:
    """Two-stage matching: FAISS (coarse) + detailed (fine)"""
    
    COARSE_THRESHOLD = 0.30   # FAISS similarity threshold
    DETAILED_THRESHOLD = 0.38 # Final similarity threshold
    
    def __init__(self, embeddings_db, labels_db):
        """
        Initialize matcher with enrollment embeddings
        
        Args:
            embeddings_db: np.array (N, 512), all student embeddings
            labels_db: np.array (N,), student IDs corresponding to embeddings
        """
        # Build FAISS index
        self.embeddings_db = embeddings_db.astype(np.float32)
        self.labels_db = labels_db
        
        # Create L2 index
        dimension = embeddings_db.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings_db)
    
    def match(self, embedding):
        """
        Match embedding using two-stage approach
        
        Args:
            embedding: np.array (512,), query embedding
        
        Returns:
            match: dict with 'student_id', 'confidence', 'rank'
        """
        embedding = embedding.astype(np.float32).reshape(1, -1)
        
        # Stage 1: FAISS coarse filter (top-5)
        distances, indices = self.index.search(embedding, k=5)
        
        # Convert L2 distance to similarity
        # similarity = 1 / (1 + distance)
        similarities = 1.0 / (1.0 + distances[0])
        
        # Filter by coarse threshold
        coarse_matches = [
            (idx, sim) 
            for idx, sim in zip(indices[0], similarities)
            if sim > self.COARSE_THRESHOLD
        ]
        
        if not coarse_matches:
            return {'student_id': None, 'confidence': 0.0}
        
        # Stage 2: Detailed matching (cosine similarity)
        best_match = None
        best_confidence = 0.0
        
        for db_idx, coarse_sim in coarse_matches:
            # Cosine similarity
            db_embedding = self.embeddings_db[db_idx]
            cosine_sim = np.dot(embedding[0], db_embedding) / (
                np.linalg.norm(embedding[0]) * np.linalg.norm(db_embedding)
            )
            
            if cosine_sim > best_confidence:
                best_confidence = cosine_sim
                best_match = db_idx
        
        if best_confidence < self.DETAILED_THRESHOLD:
            return {'student_id': None, 'confidence': 0.0}
        
        student_id = self.labels_db[best_match]
        
        return {
            'student_id': student_id,
            'confidence': float(best_confidence),
            'rank': 1
        }
```

**Two-Stage Benefits**:
- Stage 1 (FAISS): Fast (1ms), returns top-5 candidates
- Stage 2 (Cosine): Accurate (10ms), refines to single match
- **Total: ~15ms** (vs 200ms single-stage at scale)

### tracker.py — CSRT Tracking

```python
import cv2

class CSRTTracker:
    """CSRT tracker for temporal consistency"""
    
    def __init__(self, frame, bbox):
        """
        Initialize tracker with first detection
        
        Args:
            frame: Current frame
            bbox: (x1, y1, x2, y2) bounding box
        """
        # Convert to (x, y, w, h) format for OpenCV
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, (x1, y1, w, h))
    
    def update(self, frame):
        """
        Update tracker position in new frame
        
        Args:
            frame: New frame
        
        Returns:
            bbox: (x1, y1, x2, y2) updated bounding box or None
            confidence: Tracker confidence (0-1)
        """
        success, bbox = self.tracker.update(frame)
        
        if not success:
            return None, 0.0
        
        # Convert back to (x1, y1, x2, y2)
        x, y, w, h = bbox
        bbox = (int(x), int(y), int(x + w), int(y + h))
        
        return bbox, 1.0
```

**Benefit**: Reduce inference by 60% by reusing embedding from previous frame

---

## Module 2: anti_spoofing/ — Liveness Detection

**Purpose**: Detect spoofing attacks (print, replay, mask, deepfake).

**Files**:
- `model.py` — Silent-Face CNN
- `blink_detector.py` — Eye blink detection
- `movement_checker.py` — Head motion detection
- `spoof_detector.py` — Multi-layer aggregation

### model.py — Silent-Face CNN

```python
import onnxruntime as ort
import numpy as np

class SilentFaceCNN:
    """3-class liveness classifier: spoof/real/other"""
    
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, face_image):
        """
        Predict liveness class
        
        Args:
            face_image: np.array (H, W, 3), aligned face
        
        Returns:
            prediction: dict with 'class' and 'scores' (3,)
        """
        # Preprocess: resize to model input size, normalize
        face = cv2.resize(face_image, (224, 224))
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Convert to NCHW
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, 0)
        
        # Inference
        output = self.session.run(
            [self.output_name],
            {self.input_name: face}
        )[0]
        
        # Softmax
        scores = np.exp(output[0]) / np.sum(np.exp(output[0]))
        
        # Classes: 0=spoof, 1=real, 2=other
        class_names = ['spoof', 'real', 'other']
        predicted_class = class_names[np.argmax(scores)]
        
        return {
            'class': predicted_class,
            'scores': scores,
            'real_confidence': scores[1]
        }
```

**Output**: Real confidence (0-1)

### blink_detector.py — Eye Blink Detection

```python
import numpy as np

class BlinkDetector:
    """Detect blink using Eye Aspect Ratio (EAR)"""
    
    # Eye landmark indices (from 68-point face alignment)
    LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
    
    EAR_THRESHOLD = 0.25  # Blink threshold
    
    def compute_eye_aspect_ratio(self, eye_landmarks):
        """
        Compute Eye Aspect Ratio (EAR)
        
        Formula:
        EAR = (||p2−p6|| + ||p3−p5||) / (2×||p1−p4||)
        
        Where p1-p6 are eye corner points
        """
        # Vertical distances
        vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR calculation
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    def detect_blink(self, landmarks):
        """
        Detect if person is blinking
        
        Args:
            landmarks: np.array (68, 2), face landmarks
        
        Returns:
            blink_detected: bool
            ear_left: float (Eye Aspect Ratio)
            ear_right: float
        """
        # Extract eye landmarks
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        
        # Compute EAR
        ear_left = self.compute_eye_aspect_ratio(left_eye)
        ear_right = self.compute_eye_aspect_ratio(right_eye)
        ear_avg = (ear_left + ear_right) / 2.0
        
        # Detect blink
        blink = ear_avg < self.EAR_THRESHOLD
        
        return blink, ear_left, ear_right
```

**EAR Formula**: $ \text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||} $

### spoof_detector.py — Multi-Layer Aggregation

```python
import numpy as np

class SpoofDetector:
    """Multi-layer liveness verification"""
    
    # Confidence weights
    WEIGHTS = {
        'cnn': 0.40,       # Silent-Face CNN
        'blink': 0.25,     # Blink detection
        'motion': 0.20,    # Head motion
        'heuristics': 0.15 # Frame heuristics
    }
    
    def __init__(self):
        self.cnn_model = SilentFaceCNN(...)
        self.blink_detector = BlinkDetector()
        self.movement_checker = HeadMovementChecker()
    
    def verify_liveness(self, frames, landmarks_seq):
        """
        Multi-frame liveness verification
        
        Args:
            frames: list of frame images (typically 5)
            landmarks_seq: list of facial landmarks (5 frames)
        
        Returns:
            liveness: dict with 'is_live', 'confidence', 'breakdown'
        """
        cnn_scores = []
        blink_scores = []
        motion_scores = []
        heuristic_scores = []
        
        # Process each frame
        for frame, landmarks in zip(frames, landmarks_seq):
            # 1. CNN prediction
            cnn_result = self.cnn_model.predict(frame)
            cnn_scores.append(cnn_result['real_confidence'])
            
            # 2. Blink detection
            blink_detected, _, _ = self.blink_detector.detect_blink(landmarks)
            blink_scores.append(1.0 if blink_detected else 0.0)
            
            # 3. Motion detection
            motion_score = self.movement_checker.check_movement(frame, landmarks)
            motion_scores.append(motion_score)
            
            # 4. Frame heuristics
            heuristic_score = self.check_frame_heuristics(frame)
            heuristic_scores.append(heuristic_score)
        
        # Aggregate (multi-frame voting)
        cnn_avg = np.mean(cnn_scores)
        blink_avg = np.mean(blink_scores)
        motion_avg = np.mean(motion_scores)
        heuristic_avg = np.mean(heuristic_scores)
        
        # Weighted combination
        overall_confidence = (
            self.WEIGHTS['cnn'] * cnn_avg +
            self.WEIGHTS['blink'] * blink_avg +
            self.WEIGHTS['motion'] * motion_avg +
            self.WEIGHTS['heuristics'] * heuristic_avg
        )
        
        return {
            'is_live': overall_confidence > 0.50,
            'confidence': overall_confidence,
            'breakdown': {
                'cnn': cnn_avg,
                'blink': blink_avg,
                'motion': motion_avg,
                'heuristics': heuristic_avg
            }
        }
```

**Output**: Liveness confidence (0-1)

---

## Module 3: vision/ — Pipeline Orchestration

**Purpose**: Coordinate all modules into complete pipeline.

```python
class VisionPipeline:
    """Complete face recognition + anti-spoofing pipeline"""
    
    def __init__(self):
        self.detector = FaceDetector(...)
        self.aligner = FaceAligner()
        self.embedder = ArcFaceEmbedder(...)
        self.matcher = TwoStageMatcher(...)
        self.spoof_detector = SpoofDetector(...)
        self.trackers = {}  # Active trackers
    
    def process_frame(self, frame):
        """
        Single frame processing
        
        Returns:
            results: list of dict with 'student_id', 'confidence', 'liveness'
        """
        # Step 1: Detect faces
        detections = self.detector.detect(frame)
        boxes = self.detector.get_boxes(detections)
        
        results = []
        
        for box in boxes:
            bbox, confidence, landmarks = (
                box['bbox'], 
                box['confidence'], 
                box['landmarks']
            )
            
            # Step 2: Align face
            face_roi = self.get_roi(frame, bbox)
            aligned_face, _ = self.aligner.align(face_roi, landmarks)
            
            # Step 3: Check quality
            if not self.check_quality(aligned_face, frame, bbox):
                continue
            
            # Step 4: Generate embedding
            embedding = self.embedder.embed(aligned_face)
            
            # Step 5: Match against database
            match_result = self.matcher.match(embedding)
            
            if match_result['student_id'] is None:
                continue
            
            results.append({
                'student_id': match_result['student_id'],
                'recognition_confidence': match_result['confidence'],
                'bbox': bbox,
                'embedding': embedding
            })
        
        return results
    
    def verify_attendance(self, video_stream, num_frames=5):
        """
        Multi-frame verification for attendance
        
        Args:
            video_stream: Video capture object
            num_frames: Number of frames to process (default 5)
        
        Returns:
            attendance_result: dict with 'student_id', 'attendance_confidence'
        """
        frame_results = []
        
        # Capture frames
        for _ in range(num_frames):
            ret, frame = video_stream.read()
            if not ret:
                break
            
            results = self.process_frame(frame)
            frame_results.append(results)
        
        # Multi-frame voting
        if not frame_results:
            return None
        
        # Aggregate results
        attendance = self.aggregate_results(frame_results)
        
        return attendance
```

---

## Module 4: core/ — Services

**Purpose**: Database, configuration, authentication, utilities.

### database.py — MongoDB Connection

```python
from pymongo import MongoClient
from contextlib import contextmanager

class DatabaseManager:
    """MongoDB connection and operations"""
    
    def __init__(self, uri, db_name='attendance_db'):
        self.client = MongoClient(
            uri,
            maxPoolSize=50,
            minPoolSize=5,
            maxIdleTimeMS=30000
        )
        self.db = self.client[db_name]
    
    def get_student_embeddings(self, student_id):
        """Retrieve embeddings for student"""
        doc = self.db.students.find_one(
            {'student_id': student_id},
            {'face_embeddings': 1}
        )
        
        if not doc:
            return []
        
        embeddings = []
        for emb_doc in doc.get('face_embeddings', []):
            embedding_binary = emb_doc['embedding']
            # Convert from binary back to array
            embedding = binary_to_embed(embedding_binary)
            embeddings.append(embedding)
        
        return embeddings
    
    def mark_attendance(self, student_id, course_id, status):
        """Mark attendance"""
        self.db.attendance.insert_one({
            'student_id': student_id,
            'course_id': course_id,
            'marked_at': datetime.now(timezone.utc),
            'status': status,
            'created_at': datetime.now(timezone.utc)
        })
```

### config.py — Configuration Management

```python
import os

class Config:
    """Base configuration"""
    
    # Face Detection
    DETECTION_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.3
    
    # Recognition
    RECOGNITION_THRESHOLD = 0.38
    MIN_CONFIDENCE = 0.42
    DISTANCE_GAP = 0.08
    
    # Liveness
    LIVENESS_THRESHOLD = 0.50
    BLINK_DETECTION_ENABLED = True
    MOTION_THRESHOLD = 0.15
    
    # Pipeline
    FRAME_SKIP_INTERVAL = 3
    TRACKER_REUSE_ENABLED = True
    INFERENCE_BATCH_SIZE = 1
    
    # Database
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
    DB_NAME = 'attendance_db'
    
    # Paths
    MODELS_DIR = 'models/'
```

---

## Module 5: admin_app/ & student_app/ — Flask Applications

**Purpose**: Web interface for attendance marking.

### admin_app/routes/ — Admin Routes

```python
from flask import Blueprint, render_template, jsonify, request
from flask_socketio import emit

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/dashboard')
def dashboard():
    """Admin dashboard"""
    courses = db.courses.find()
    return render_template('admin/dashboard.html', courses=courses)

@admin_bp.route('/attendance/<course_id>', methods=['GET'])
def get_attendance(course_id):
    """Get attendance records for course"""
    attendance = db.attendance.aggregate([
        {'$match': {'course_id': course_id}},
        {'$group': {
            '_id': '$student_id',
            'present_count': {'$sum': {'$cond': [{'$eq': ['$status', 'present']}, 1, 0]}},
            'total': {'$sum': 1}
        }},
        {'$project': {
            'student_id': '$_id',
            'attendance_percentage': {'$multiply': [{'$divide': ['$present_count', '$total']}, 100]}
        }}
    ])
    
    return jsonify(list(attendance))

@admin_bp.route('/verify-attendance', methods=['POST'])
def verify_attendance():
    """Manual verification of attendance"""
    data = request.json
    student_id = data['student_id']
    course_id = data['course_id']
    
    db.attendance.update_one(
        {'student_id': student_id, 'course_id': course_id},
        {'$set': {'admin_verified': True, 'verified_by': current_user.id}}
    )
    
    return jsonify({'status': 'verified'})
```

### student_app/routes/ — Student Routes

```python
from flask import Blueprint, render_template
from flask_socketio import emit, on

student_bp = Blueprint('student', __name__)

@student_bp.route('/enroll', methods=['GET', 'POST'])
def enroll():
    """Enrollment page"""
    if request.method == 'POST':
        # Process enrollment
        camera_frames = request.files['video']
        # Extract embeddings and store
        pass
    
    return render_template('student/enroll.html')

@student_bp.route('/attendance')
def mark_attendance():
    """Mark attendance page (camera capture)"""
    return render_template('student/attendance.html')

# SocketIO events
@on('start_camera')
def handle_camera_start():
    """Start camera feed"""
    emit('camera_ready', {'status': 'ok'})

@on('frame_data')
def handle_frame(data):
    """Process frame from client"""
    # Decode frame, run inference
    results = vision_pipeline.process_frame(frame)
    
    emit('frame_result', {
        'detections': results,
        'timestamp': time.time()
    }, broadcast=False)
```

---

## Data Flow Example: Attendance Marking

```
1. Student opens app → /student/attendance

2. Client: Capture camera frame
   ↓
3. SocketIO: emit('frame_data', frame_bytes)
   ↓
4. Server: vision_pipeline.process_frame()
   - detector.detect() → boxes (33ms)
   - aligner.align() → aligned faces (5ms)
   - embedder.embed() → embeddings (18ms)
   - matcher.match() → student_id (15ms)
   - spoof_detector.verify_liveness() → confidence (20ms)
   ↓
5. Database: db.attendance.insert_one()
   ↓
6. SocketIO: emit('attendance_marked', result)
   ↓
7. Client: Display "Marked" confirmation
```

**Total Latency**: ~100ms per frame

---

## Key Integration Points

### Embedding Reuse Optimization

```python
# Once embedding generated, reuse for:
# 1. Current frame matching
# 2. Tracker initialization
# 3. FAISS index (multi-frame voting)
# 4. Database storage (for future reference)

embedding = embedder.embed(aligned_face)  # 18ms

# Reuse 1: Match immediately
match = matcher.match(embedding)

# Reuse 2: Store in embedding cache
embedding_cache[face_id] = embedding

# Reuse 3: Use in next frame with CSRT tracker
# (avoid re-embedding if tracked)
```

### Quality Gating Pipeline

```python
def check_quality(aligned_face, frame, bbox):
    """Multi-criterion quality check"""
    
    # 1. Blur check (Laplacian variance)
    blur = cv2.Laplacian(aligned_face, cv2.CV_64F).var()
    if blur < 100:  # Too blurry
        return False
    
    # 2. Brightness check
    brightness = np.mean(aligned_face)
    if brightness < 40 or brightness > 250:  # Too dark/bright
        return False
    
    # 3. Face size check
    face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    frame_area = frame.shape[0] * frame.shape[1]
    face_ratio = face_area / frame_area
    
    if face_ratio < 0.05 or face_ratio > 0.80:  # Too small/large
        return False
    
    return True
```

---

## Entry Points

### 1. HTTP Route Entry

```python
@app.route('/mark-attendance', methods=['POST'])
def http_mark_attendance():
    """RESTful API for attendance"""
    video_file = request.files['video']
    result = vision_pipeline.verify_attendance(video_file)
    return jsonify(result)
```

### 2. SocketIO Entry

```python
@socketio.on('capture_frame')
def on_capture_frame(data):
    """Real-time frame processing"""
    frame = decode_frame(data)
    results = vision_pipeline.process_frame(frame)
    emit('frame_processed', results, broadcast=False)
```

### 3. Script Entry

```bash
# scripts/seed_demo_data.py
# scripts/debug_pipeline.py
# scripts/calibrate_liveness_threshold.py
```

---

## Performance Characteristics

| Operation | Time | Memory |
|-----------|------|--------|
| Detection | 33ms | 50MB |
| Alignment | 5ms | 10MB |
| Embedding | 18ms | 30MB |
| Matching (coarse) | 5ms | 100MB (index) |
| Matching (detailed) | 10ms | 10MB |
| Liveness (CNN) | 20ms | 40MB |
| **Total per frame** | **~100ms** | **~240MB** |

---

## References

1. OpenCV Documentation: https://docs.opencv.org
2. ONNX Runtime: https://onnxruntime.ai
3. FAISS: https://github.com/facebookresearch/faiss
4. Flask-SocketIO: https://flask-socketio.readthedocs.io
5. MongoDB PyMongo: https://pymongo.readthedocs.io
