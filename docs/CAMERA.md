# Camera Module & Real-Time Processing

## Table of Contents

1. [Camera Module Architecture](#camera-module-architecture)
2. [Frame Capture Loop](#frame-capture-loop)
3. [State Machine](#state-machine)
4. [Event System](#event-system)
5. [Error Handling & Recovery](#error-handling--recovery)
6. [Debugging & Diagnostics](#debugging--diagnostics)

---

## Camera Module Architecture

### Module Structure

```
camera/
├── __init__.py
├── camera.py           # Core Camera class
└── utils.py            # Helper functions (frame encoding, etc.)

vision/
├── __init__.py
├── pipeline.py         # Detection, tracking, recognition orchestration
├── recognition.py      # Embedding generation, matching
├── anti_spoofing.py    # Liveness detection
├── face_engine.py      # ArcFace backend abstraction
└── ...
```

### Class Hierarchy

```python
# camera/camera.py

class Camera:
    """
    Real-time face detection, tracking, and recognition.
    
    Responsibilities:
    - Frame capture from camera
    - Track management (create, update, close)
    - Identity matching and liveness checking
    - Attendance marking on confirmation
    - Event streaming (SocketIO broadcast)
    - Error recovery and graceful degradation
    """
    
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.cap = None  # OpenCV VideoCapture
        self._tracks = []  # List of FaceTrack objects
        self._track_identity_cache = {}  # {track_id: (student_id, confidence, timestamp)}
        self._seen = {}  # {student_id: last_timestamp} for cooldown
        self._events = deque(maxlen=100)  # Recent events for HTTP streaming
        self._log_buffer = deque(maxlen=1000)  # Recent logs
        self._session = None  # Current AttendanceSession
        self._stop_event = threading.Event()
        self._thread = None
        
    def run_async(self) -> threading.Thread:
        """Start capture loop in background thread."""
        ...
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """Process single frame (detect, track, recognize)."""
        ...
    
    def update_track(self, track: FaceTrack, frame: np.ndarray) -> bool:
        """Update track (CSRT) and run full pipeline."""
        ...
    
    def stop(self):
        """Graceful shutdown."""
        ...
```

---

## Frame Capture Loop

### Main Processing Loop

```python
# camera/camera.py

def _capture_loop(self):
    """Main frame capture and processing loop."""
    
    self.cap = cv2.VideoCapture(0)  # Open camera
    
    if not self.cap.isOpened():
        logger.error(f"Camera {self.camera_id} failed to open")
        self._emit_event({
            'type': 'error',
            'message': 'Camera failed to open',
            'camera_id': self.camera_id
        })
        return
    
    # Set camera properties for consistency
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    self.cap.set(cv2.CAP_PROP_FPS, 30)
    
    logger.info(f"Camera opened: {self.camera_id}")
    self._emit_event({'type': 'camera_opened', 'camera_id': self.camera_id})
    
    frame_count = 0
    last_frame = None
    
    while not self._stop_event.is_set():
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame from {self.camera_id}")
                continue
            
            frame_count += 1
            
            # Process frame
            try:
                result = self.process_frame(frame)
                
                # Emit event for web UI
                self._emit_event({
                    'type': 'frame_processed',
                    'frame_count': frame_count,
                    'tracks': len(self._tracks),
                    'timestamp': datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Frame processing error: {e}", exc_info=True)
                self._log_error(e)
            
            last_frame = frame
            
            # Periodic cleanup
            if frame_count % 300 == 0:  # Every 10 seconds @ 30 FPS
                self._cleanup_stale_tracks()
                logger.info(f"Active tracks: {len(self._tracks)}")
        
        except KeyboardInterrupt:
            logger.info("Camera loop interrupted by user")
            break
        except Exception as e:
            logger.error(f"Critical camera loop error: {e}", exc_info=True)
            self._emit_event({
                'type': 'camera_error',
                'error': str(e)
            })
            break
    
    # Cleanup
    if self.cap:
        self.cap.release()
    
    logger.info(f"Camera {self.camera_id} closed")
    self._emit_event({'type': 'camera_closed', 'camera_id': self.camera_id})
```

### Frame Processing Pipeline

```python
def process_frame(self, frame: np.ndarray) -> dict:
    """
    Process single frame: detect → track → match → recognize → liveness → mark attendance
    
    Pipeline:
    1. Detect faces (YuNet, every N frames or on motion)
    2. Associate detections to tracks (IoU-based)
    3. Update tracks (CSRT)
    4. For new/updated tracks: encode face → match in database → check liveness
    5. Multi-frame voting → confirm identity
    6. Mark attendance if confirmed
    """
    
    start_time = time.perf_counter()
    
    # Resize for processing
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    target_height = FRAME_PROCESS_WIDTH // aspect_ratio
    resized = cv2.resize(frame, (FRAME_PROCESS_WIDTH, int(target_height)))
    
    # 1. Detect faces (motion-gated)
    frame_count = getattr(self, '_frame_count', 0)
    detections = self._detect_and_associate(resized, frame_count)
    self._frame_count = frame_count + 1
    
    # 2. Update existing tracks (CSRT tracker)
    for track in self._tracks[:]:
        if track.id not in [d['track_id'] for d in detections]:
            # Update tracker (even without detection)
            tracker_result = track.tracker.update(resized)
            
            if tracker_result[0]:  # Tracker succeeded
                track.frames_missing = 0
                track.last_update = datetime.utcnow()
                
                # Re-run recognition on tracked face
                x1, y1, w, h = [int(v) for v in tracker_result[1]]
                face_roi = resized[y1:y1+h, x1:x1+w]
                
                if face_roi.size > 0:
                    self._process_track(track, face_roi, resized)
            else:
                track.frames_missing += 1
        
        # Remove if lost for too long
        if track.frames_missing > MAX_TRACK_AGE:
            self._tracks.remove(track)
            logger.debug(f"Track {track.id} removed (lost {track.frames_missing} frames)")
    
    # 3. Process new detections
    for detection in detections:
        if detection['track_id'] is None:  # New face
            new_track = self._create_track(detection, resized)
            self._tracks.append(new_track)
            logger.debug(f"New track created: {new_track.id}")
    
    # Performance metrics
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return {
        'frame_count': frame_count,
        'tracks': len(self._tracks),
        'detections': len(detections),
        'latency_ms': elapsed_ms,
        'fps': 1000 / elapsed_ms if elapsed_ms > 0 else 0
    }
```

---

## State Machine

### Track State Transitions

```
FaceTrack states:

DETECTED
  ├─ frames_matched ≥ RECOGNITION_CONFIRM_FRAMES?
  │  └─ YES → RECOGNIZED
  └─ NO → frames_missing++ after N frames → LOST

RECOGNIZED
  ├─ Still visible (tracker updated)?
  │  └─ YES → RECOGNIZED (stable)
  └─ NO → frames_missing++ → LOST

LOST
  ├─ frames_missing > MAX_TRACK_AGE?
  │  └─ YES → DELETE (remove track)
  └─ NO → frames_missing++ (waiting)
        │
        └─ Re-detection? → DETECTED (restart)
```

### State Tracking Code

```python
# vision/pipeline.py

class FaceTrack:
    """Represents a detected face being tracked across frames."""
    
    def __init__(self, track_id: str, detection: dict, tracker):
        self.id = track_id  # UUID
        self.tracker = tracker  # CSRT or KCF tracker
        self.bbox = detection['bbox']
        self.created_at = datetime.utcnow()
        self.last_update = datetime.utcnow()
        
        # Identity tracking
        self.identity_votes = []  # [(student_id, confidence), ...]
        self.liveness_votes = []  # [True/False/None, ...]
        self.frames_matched = 0
        self.frames_missing = 0
        
        # Cached results (TTL-based)
        self._identity_cache = None
        self._identity_cache_time = None
        self._embedding_cache = None
        self._embedding_cache_time = None
    
    @property
    def is_confirmed(self) -> bool:
        """Check if identity is confirmed via voting."""
        if len(self.identity_votes) < RECOGNITION_CONFIRM_FRAMES:
            return False
        
        # All votes same identity?
        recent_votes = self.identity_votes[-RECOGNITION_CONFIRM_FRAMES:]
        student_ids = [v[0] for v in recent_votes]
        return len(set(student_ids)) == 1  # All same
    
    @property
    def confirmed_identity(self) -> tuple:
        """Get confirmed identity (student_id, avg_confidence)."""
        if not self.is_confirmed:
            return None, 0.0
        
        recent_votes = self.identity_votes[-RECOGNITION_CONFIRM_FRAMES:]
        student_id = recent_votes[0][0]
        confidence = np.mean([v[1] for v in recent_votes])
        
        return student_id, confidence
    
    @property
    def is_alive(self) -> bool:
        """Check if face passed liveness check."""
        if not self.liveness_votes:
            return None
        
        recent_votes = self.liveness_votes[-LIVENESS_CONFIRM_THRESHOLD:]
        alive_count = sum(1 for v in recent_votes if v is True)
        
        return alive_count >= LIVENESS_CONFIRM_THRESHOLD
    
    def add_vote(self, student_id: str, confidence: float, liveness: bool):
        """Add identity and liveness vote."""
        self.identity_votes.append((student_id, confidence))
        self.liveness_votes.append(liveness)
        self.frames_matched += 1
        
        # Keep only recent votes
        if len(self.identity_votes) > RECOGNITION_HISTORY_SIZE:
            self.identity_votes.pop(0)
            self.liveness_votes.pop(0)

# Usage
track.add_vote(
    student_id='CS21001',
    confidence=0.94,
    liveness=True
)

if track.is_confirmed and track.is_alive:
    mark_attendance(track.confirmed_identity[0])
```

---

## Event System

### Event Emission

```python
# camera/camera.py

def _emit_event(self, event: dict):
    """Emit event to WebSocket clients via SocketIO."""
    
    event['camera_id'] = self.camera_id
    event['timestamp'] = datetime.utcnow().isoformat()
    
    self._events.append(event)
    
    # Broadcast to all clients
    try:
        socketio.emit(
            'camera_event',
            event,
            room=f"camera:{self.camera_id}",
            namespace='/'
        )
    except Exception as e:
        logger.warning(f"SocketIO emit failed: {e}")

# Events emitted during pipeline

def _on_face_detected(self, track_id: str):
    """Emit when new face detected."""
    self._emit_event({
        'type': 'face_detected',
        'track_id': track_id,
        'event_id': str(uuid4())
    })

def _on_identity_matched(self, track_id: str, student_id: str, confidence: float):
    """Emit when face matched to student."""
    self._emit_event({
        'type': 'identity_matched',
        'track_id': track_id,
        'student_id': student_id,
        'confidence': confidence,
        'event_id': str(uuid4())
    })

def _on_attendance_marked(self, student_id: str, confidence: float):
    """Emit when attendance actually marked."""
    self._emit_event({
        'type': 'attendance_marked',
        'student_id': student_id,
        'confidence': confidence,
        'timestamp': datetime.utcnow().isoformat(),
        'event_id': str(uuid4())
    })

def _on_liveness_failed(self, track_id: str):
    """Emit when liveness check failed (spoof detected)."""
    self._emit_event({
        'type': 'liveness_failed',
        'track_id': track_id,
        'reason': 'Anti-spoofing check failed',
        'event_id': str(uuid4())
    })
```

### Real-Time Streaming (SocketIO)

```html
<!-- Frontend: web/static/camera.html -->

<script>
  const socket = io('/');
  const cameraId = 'lab-1';
  
  socket.on('connect', function() {
    socket.emit('join_camera', { camera_id: cameraId });
    console.log('Connected to camera stream');
  });
  
  socket.on('camera_event', function(event) {
    console.log('Event:', event);
    
    switch(event.type) {
      case 'face_detected':
        updateUI(`Face detected: ${event.track_id}`);
        break;
      
      case 'identity_matched':
        updateUI(`Matched: ${event.student_id} (${event.confidence.toFixed(2)})`);
        break;
      
      case 'attendance_marked':
        displayNotification(`✓ Attendance marked: ${event.student_id}`);
        break;
      
      case 'liveness_failed':
        displayWarning('⚠ Liveness check failed (possible spoof attempt)');
        break;
    }
  });
</script>
```

---

## Error Handling & Recovery

### Camera Crash Recovery

```python
# camera/camera.py

def ensure_camera_running(self):
    """Ensure camera thread is running; restart if crashed."""
    
    if self._thread is None or not self._thread.is_alive():
        logger.warning(f"Camera {self.camera_id} thread not running, restarting...")
        
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"Camera-{self.camera_id}"
        )
        self._thread.start()
        
        self._emit_event({
            'type': 'camera_restarted',
            'camera_id': self.camera_id
        })

# Scheduled check (every 10 seconds)
@scheduler.scheduled_job('interval', seconds=10)
def check_camera_health():
    """Verify all active cameras are running."""
    for camera_id, camera in ACTIVE_CAMERAS.items():
        try:
            camera.ensure_camera_running()
        except Exception as e:
            logger.error(f"Camera health check failed for {camera_id}: {e}")
```

### Graceful Degradation

```python
# vision/anti_spoofing.py

def check_liveness(face_roi):
    """Check liveness with graceful fallback on model failure."""
    
    try:
        if not _predictor:
            logger.warning("Liveness model not loaded; skipping check")
            return 1, 1.0  # Mark as real
        
        # Normal liveness check
        label, confidence = _predictor.predict(face_roi)
        return label, confidence
    
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        # Graceful degradation: assume real if model fails
        return 1, 1.0  # Mark as real with high confidence
```

### Connection Loss Handling

```python
# core/database.py (MongoDB circuit breaker)

class CircuitBreaker:
    """Fail-safe pattern for database operations."""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.is_open = False
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker."""
        
        if self.is_open:
            # Check if timeout expired
            if time.time() - self.last_failure_time > self.timeout:
                logger.info("Circuit breaker: Attempting recovery...")
                self.is_open = False
            else:
                raise Exception("Circuit breaker is OPEN (too many failures)")
        
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0  # Reset on success
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                logger.error(f"Circuit breaker OPEN: {self.failure_count} failures")
                self.is_open = True
            
            raise

# Usage
breaker = CircuitBreaker(failure_threshold=5, timeout=60)

try:
    breaker.call(mark_attendance, student_id, 'Present')
except Exception as e:
    logger.error(f"Attendance marking failed: {e}")
    # Continue processing (skip marking, retry later)
```

---

## Debugging & Diagnostics

### Logging Pipeline

```python
# camera/camera.py

import logging

# Configure structured logging
logger = logging.getLogger(__name__)

def _process_track(self, track: FaceTrack, face_roi: np.ndarray):
    """Process track with detailed logging."""
    
    logger.debug(f"[Track {track.id}] Processing face ({face_roi.shape})")
    
    # Check quality
    is_valid, reason = check_face_quality_gate(face_roi)
    if not is_valid:
        logger.debug(f"[Track {track.id}] Quality check failed: {reason}")
        return
    
    # Generate embedding
    emb = encode_face(face_roi)
    logger.debug(f"[Track {track.id}] Embedding generated (shape={emb.shape})")
    
    # Match in database
    match_id, similarity = match_embeddings(emb, self._student_embeddings)
    logger.debug(f"[Track {track.id}] Match result: {match_id} (sim={similarity:.3f})")
    
    # Check liveness
    label, liveness_conf = check_liveness(face_roi)
    logger.debug(f"[Track {track.id}] Liveness: label={label}, conf={liveness_conf:.3f}")
    
    if match_id:
        track.add_vote(match_id, similarity, label == 1)
        logger.info(f"[Track {track.id}] Vote added for {match_id}: "
                   f"similarity={similarity:.3f}, liveness={label == 1}")
        
        if track.is_confirmed and track.is_alive:
            logger.info(f"[Track {track.id}] ✓ CONFIRMED: {match_id}")
```

### Visualization Debug Tool

```python
# scripts/debug_pipeline.py

import cv2
import numpy as np
from vision.pipeline import detect_faces_yunet
from vision.recognition import encode_face, match_embeddings
from vision.anti_spoofing import check_liveness

def visualize_pipeline(image_path):
    """Visualize all pipeline stages for debugging."""
    
    frame = cv2.imread(image_path)
    
    # Stage 1: Detection
    detections = detect_faces_yunet(frame)
    frame_detected = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(frame_detected, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_detected, f"Conf: {det['confidence']:.2f}",
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    
    # Stage 2: Quality & Liveness
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        face_roi = frame[y1:y2, x1:x2]
        
        # Quality
        is_valid, reason = check_face_quality_gate(face_roi)
        label, liveness = check_liveness(face_roi)
        
        color = (0, 255, 0) if is_valid and label == 1 else (0, 0, 255)
        text = f"Q:{reason}, L:{label}, C:{liveness:.2f}"
        cv2.putText(frame_detected, text,
                   (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    
    # Save visualization
    cv2.imwrite('debug_output.jpg', frame_detected)
    print("Saved: debug_output.jpg")
    
    # Display
    cv2.imshow('Pipeline Debug', frame_detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage
python scripts/debug_pipeline.py --image=test.jpg
```

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0
