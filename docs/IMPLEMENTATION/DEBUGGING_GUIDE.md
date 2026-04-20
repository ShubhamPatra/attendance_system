# Debugging Guide: Troubleshooting Common Issues

Comprehensive guide for debugging, profiling, and troubleshooting AutoAttendance system.

---

## Quick Troubleshooting Matrix

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| **No face detected** | Poor lighting, face too far | Improve lighting, move closer |
| **"Face not recognized"** | Low enrollment quality, face angle | Re-enroll with better photos |
| **"Spoofing detected"** | False positive | Check liveness threshold in config |
| **Slow inference** | CPU-only mode, batch size too large | Use GPU, reduce batch size |
| **MongoDB connection failed** | Service not running | `mongosh` to test connection |
| **"Models not found"** | Files not downloaded | Download from script or GitHub |
| **Low FPS** | GPU out of memory | Reduce frame resolution or batch size |
| **API timeout** | Database slow | Check indexes, add cache layer |

---

## Part 1: Setup Debugging Environment

### Enable Debug Logging

**In .env file**:
```env
FLASK_ENV=development
FLASK_DEBUG=true
LOG_LEVEL=DEBUG
```

**In Python code**:
```python
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Use throughout code
logger.debug("Processing frame")
logger.info("Face detected")
logger.warning("Low confidence")
logger.error("Database error")
```

### Log File Output

**Redirect logs to file**:
```bash
python app.py 2>&1 | tee logs/debug.log

# On Windows:
python app.py > logs/debug.log 2>&1
```

**View logs in real-time**:
```bash
tail -f logs/debug.log          # Linux/macOS
Get-Content logs/debug.log -Tail 100 -Wait  # Windows PowerShell
```

### Use Debug Breakpoints (VS Code)

**In launch.json**:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Flask",
      "type": "python",
      "request": "launch",
      "module": "flask",
      "env": {"FLASK_APP": "app.py", "FLASK_ENV": "development"},
      "args": ["run", "--no-debugger"],
      "jinja": true
    }
  ]
}
```

**Set breakpoint**: Click line number, then run debugger

---

## Part 2: Common Errors & Solutions

### Error 1: "cv2.error: (-5:Bad argument)"

**Cause**: Invalid frame format or resolution

**Debug**:
```python
import cv2

# Check frame properties
frame = cv2.imread('test.jpg')
print(f"Frame shape: {frame.shape}")      # Should be (H, W, 3)
print(f"Frame dtype: {frame.dtype}")      # Should be uint8
print(f"Frame min/max: {frame.min()}/{frame.max()}")  # Should be 0-255

# If shape wrong:
if frame.shape != (480, 640, 3):
    frame = cv2.resize(frame, (640, 480))
```

**Fix**:
```python
# Validate frame before processing
def validate_frame(frame):
    if frame is None:
        raise ValueError("Frame is None")
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if len(frame.shape) != 3:
        raise ValueError(f"Invalid frame shape: {frame.shape}")
    return frame

frame = validate_frame(frame)
```

### Error 2: "RuntimeError: ONNX model failed to load"

**Cause**: Model file corrupted or path wrong

**Debug**:
```python
import os
import onnxruntime as ort

model_path = 'models/face_detection_yunet_2023mar.onnx'

# Check file exists
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in models/: {os.listdir('models/')}")
else:
    print(f"✓ Model found, size: {os.path.getsize(model_path)} bytes")

# Check file integrity
if os.path.getsize(model_path) < 100000:  # Less than 100KB
    print("ERROR: Model file too small (corrupted?)")

# Try loading
try:
    session = ort.InferenceSession(model_path)
    print("✓ ONNX model loaded successfully")
except Exception as e:
    print(f"ERROR loading ONNX: {e}")
```

**Fix**:
```bash
# Re-download model
rm models/face_detection_yunet_2023mar.onnx

# Download fresh copy
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -O models/face_detection_yunet_2023mar.onnx

# Verify size (should be ~230 KB)
ls -lh models/face_detection_yunet_2023mar.onnx
```

### Error 3: "mongoDB connection refused"

**Cause**: MongoDB service not running

**Debug**:
```python
from pymongo import MongoClient

try:
    client = MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=2000)
    client.admin.command('ping')
    print("✓ MongoDB connected")
except Exception as e:
    print(f"ERROR: MongoDB connection failed - {e}")
```

**Check MongoDB status**:
```bash
# Windows
net start MongoDB  # Start service

# Linux
sudo systemctl status mongodb

# macOS
brew services list
```

**Check connection string**:
```bash
# Test with mongosh
mongosh
> db.version()
> exit
```

### Error 4: "NMS: nms_threshold is not in [0, 1)!"

**Cause**: Invalid NMS threshold in config

**Debug**:
```python
# Check config values
nms_threshold = 0.3  # Should be 0.0 < x < 1.0
score_threshold = 0.5

if not (0 <= nms_threshold < 1):
    print(f"ERROR: Invalid NMS threshold: {nms_threshold}")
else:
    print(f"✓ NMS threshold valid: {nms_threshold}")
```

**Fix**:
```python
# Clamp to valid range
nms_threshold = max(0.01, min(0.99, nms_threshold))
score_threshold = max(0.0, min(1.0, score_threshold))
```

### Error 5: "CUDA out of memory"

**Cause**: GPU insufficient memory or batch size too large

**Debug**:
```python
import onnxruntime as ort

# Check GPU availability
print(f"ONNX Execution Provider: {ort.get_available_providers()}")

# If 'CUDAExecutionProvider' is available, CUDA is working
```

**Fix**:
```python
# Reduce batch size
BATCH_SIZE = 1  # Was 8

# Reduce input resolution
INPUT_SIZE = (320, 240)  # Was (640, 480)

# Clear GPU cache periodically
import torch
torch.cuda.empty_cache()
```

### Error 6: "IndexError: list index out of range"

**Cause**: Empty detections or wrong indexing

**Debug**:
```python
detections = detector.detect(frame)  # Returns None or empty

# WRONG:
bbox = detections[0]  # Crashes if empty!

# CORRECT:
if detections is None or len(detections) == 0:
    print("No faces detected")
    return

for detection in detections:
    bbox = detection[:4]
```

**Fix**:
```python
def safe_get_detections(detections):
    """Safely extract detections with validation"""
    if detections is None:
        return []
    if isinstance(detections, tuple):
        success, detections = detections
        if not success or detections is None:
            return []
    if len(detections) == 0:
        return []
    return detections
```

---

## Part 3: Performance Profiling

### Profile Function Latency

```python
import time
from functools import wraps

def profile_latency(func):
    """Decorator to measure function latency"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        print(f"{func.__name__}: {elapsed:.2f}ms")
        return result
    return wrapper

# Usage
@profile_latency
def process_frame(frame):
    # ... processing ...
    return result

# Output: process_frame: 95.34ms
```

### Pipeline Latency Breakdown

```python
import time

def benchmark_pipeline(frame):
    """Measure latency of each stage"""
    
    results = {}
    
    # Stage 1: Detection
    start = time.perf_counter()
    detections = detector.detect(frame)
    results['detection'] = (time.perf_counter() - start) * 1000
    
    # Stage 2: Alignment
    start = time.perf_counter()
    aligned = aligner.align(frame, landmarks)
    results['alignment'] = (time.perf_counter() - start) * 1000
    
    # Stage 3: Embedding
    start = time.perf_counter()
    embedding = embedder.embed(aligned)
    results['embedding'] = (time.perf_counter() - start) * 1000
    
    # Stage 4: Matching
    start = time.perf_counter()
    match = matcher.match(embedding)
    results['matching'] = (time.perf_counter() - start) * 1000
    
    # Stage 5: Liveness
    start = time.perf_counter()
    liveness = spoof_detector.verify_liveness([frame])
    results['liveness'] = (time.perf_counter() - start) * 1000
    
    # Print breakdown
    print("\n=== Pipeline Latency Breakdown ===")
    total = 0
    for stage, latency in results.items():
        print(f"{stage:15} : {latency:6.2f}ms")
        total += latency
    print(f"{'TOTAL':15} : {total:6.2f}ms")
    
    return results
```

**Expected Output**:
```
=== Pipeline Latency Breakdown ===
detection       :  33.45ms
alignment       :   5.23ms
embedding       :  18.67ms
matching        :  14.89ms
liveness        :  20.12ms
TOTAL           :  92.36ms
```

### Memory Profiling

```python
import tracemalloc

def profile_memory(func):
    """Measure memory usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Memory: {current / 1024 / 1024:.1f} MB (peak: {peak / 1024 / 1024:.1f} MB)")
        tracemalloc.stop()
        return result
    return wrapper

# Usage
@profile_memory
def embedder.embed(frame):
    # ... processing ...
    return embedding
```

### Database Query Performance

```python
import time

def profile_query(collection, query):
    """Measure database query time"""
    start = time.perf_counter()
    result = collection.find(query)
    list(result)  # Force execution
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Query time: {elapsed:.2f}ms")
    return result

# Usage
profile_query(
    db.attendance,
    {'student_id': 'STU2025001', 'course_id': 'CS101'}
)
```

---

## Part 4: Model Inference Debugging

### Verify Model Inputs/Outputs

```python
import onnxruntime as ort

session = ort.InferenceSession('models/arcface.onnx')

# Get input/output details
for input_spec in session.get_inputs():
    print(f"Input: {input_spec.name}")
    print(f"  Shape: {input_spec.shape}")
    print(f"  Type: {input_spec.type}")

for output_spec in session.get_outputs():
    print(f"Output: {output_spec.name}")
    print(f"  Shape: {output_spec.shape}")
    print(f"  Type: {output_spec.type}")

# Typical output:
# Input: input.1
#   Shape: ['batch_size', 3, 112, 112]
#   Type: float32
# Output: output
#   Shape: ['batch_size', 512]
#   Type: float32
```

### Debug Embedding Generation

```python
import numpy as np

def debug_embedding_generation(frame):
    """Step through embedding process"""
    
    print("=== Embedding Generation Debug ===")
    
    # Step 1: Preprocess
    aligned = cv2.resize(frame, (112, 112))
    print(f"1. Aligned shape: {aligned.shape}")
    
    aligned = aligned.astype(np.float32)
    aligned = (aligned - 127.5) / 128.0
    print(f"2. Normalized range: [{aligned.min():.3f}, {aligned.max():.3f}]")
    
    # Step 2: Convert to NCHW
    input_data = np.transpose(aligned, (2, 0, 1))
    input_data = np.expand_dims(input_data, 0)
    print(f"3. NCHW shape: {input_data.shape}")
    print(f"   Expected: (1, 3, 112, 112)")
    
    # Step 3: Inference
    output = session.run(['output'], {'input.1': input_data})[0]
    print(f"4. Raw embedding shape: {output.shape}")
    print(f"   Embedding range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Step 4: Normalize
    embedding = output[0].astype(np.float32)
    norm = np.linalg.norm(embedding, ord=2)
    embedding = embedding / norm
    print(f"5. L2 normalized embedding:")
    print(f"   Norm: {np.linalg.norm(embedding):.6f} (should be ~1.0)")
    print(f"   Range: [{embedding.min():.6f}, {embedding.max():.6f}]")
    
    return embedding
```

### Debug Liveness Verification

```python
def debug_liveness_verification(frames):
    """Step through liveness verification"""
    
    print("=== Liveness Verification Debug ===")
    
    cnn_scores = []
    blink_scores = []
    motion_scores = []
    heuristic_scores = []
    
    for i, frame in enumerate(frames):
        print(f"\nFrame {i+1}:")
        
        # CNN
        cnn_result = spoof_detector.cnn_model.predict(frame)
        cnn_score = cnn_result['real_confidence']
        cnn_scores.append(cnn_score)
        print(f"  CNN: {cnn_score:.3f} ({cnn_result['class']})")
        
        # Blink
        blink_detected, ear_l, ear_r = spoof_detector.blink_detector.detect_blink(landmarks)
        blink_score = 1.0 if blink_detected else 0.0
        blink_scores.append(blink_score)
        print(f"  Blink: {blink_score:.3f} (EAR_L={ear_l:.3f}, EAR_R={ear_r:.3f})")
        
        # Motion
        motion_score = spoof_detector.movement_checker.check_movement(frame, landmarks)
        motion_scores.append(motion_score)
        print(f"  Motion: {motion_score:.3f}")
        
        # Heuristics
        heuristic_score = spoof_detector.check_frame_heuristics(frame)
        heuristic_scores.append(heuristic_score)
        print(f"  Heuristics: {heuristic_score:.3f}")
    
    # Aggregate
    print("\n=== Aggregation ===")
    print(f"CNN avg: {np.mean(cnn_scores):.3f}")
    print(f"Blink avg: {np.mean(blink_scores):.3f}")
    print(f"Motion avg: {np.mean(motion_scores):.3f}")
    print(f"Heuristics avg: {np.mean(heuristic_scores):.3f}")
    
    overall = (
        0.40 * np.mean(cnn_scores) +
        0.25 * np.mean(blink_scores) +
        0.20 * np.mean(motion_scores) +
        0.15 * np.mean(heuristic_scores)
    )
    
    print(f"\nOverall confidence: {overall:.3f}")
    print(f"Is live: {overall > 0.50}")
```

---

## Part 5: Logging Strategy

### Application-Level Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Setup logging
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# File handler (rotating)
file_handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Usage
logger.debug("Detailed debug info")
logger.info("Important event")
logger.warning("Potential issue")
logger.error("Error occurred")
logger.critical("Critical error")
```

### Structured Logging (JSON)

```python
import json
import logging

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for better parsing"""
    
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        return json.dumps(log_data)

# Usage
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
```

### Security Event Logging

```python
def log_security_event(event_type, severity, details):
    """Log security-relevant events"""
    
    log_entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'event_type': event_type,
        'severity': severity,
        'student_id': details.get('student_id'),
        'device_id': details.get('device_id'),
        'details': details
    }
    
    # Write to security log
    db.security_logs.insert_one(log_entry)
    
    # Also log to application log
    logger.warning(f"Security event: {event_type} - {details}")

# Usage
log_security_event(
    'spoofing_attempt',
    'medium',
    {'student_id': 'STU001', 'device_id': 'CAM001', 'liveness_score': 0.22}
)
```

---

## Part 6: Performance Optimization Checklist

```python
# 1. Check frame resolution
if frame.shape != (480, 640, 3):
    frame = cv2.resize(frame, (640, 480))

# 2. Verify GPU usage
print(ort.get_available_providers())  # Should include CUDAExecutionProvider

# 3. Check inference batch size
BATCH_SIZE = 1  # Reduce if OOM

# 4. Verify database indexes
db.attendance.list_indexes()  # Should include key indexes

# 5. Check connection pool
print(f"Pool size: {db.client.topology_description}")

# 6. Enable query caching
cache.get('embeddings:STU001')  # Check Redis

# 7. Profile hot paths
@profile_latency
def matcher.match(embedding):
    # Most expensive operation
    pass

# 8. Monitor memory
print(f"Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
```

---

## References

1. Python Logging: https://docs.python.org/3/library/logging.html
2. ONNX Runtime Debugging: https://onnxruntime.ai/docs/
3. PyMongo Debugging: https://pymongo.readthedocs.io/
4. Flask Debugging: https://flask.palletsprojects.com/debugging/
5. Python Profiling: https://docs.python.org/3/library/profile.html
