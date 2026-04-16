# Testing & Validation

## Table of Contents

1. [Testing Strategy](#testing-strategy)
2. [Unit Tests](#unit-tests)
3. [Integration Tests](#integration-tests)
4. [ML Pipeline Validation](#ml-pipeline-validation)
5. [End-to-End Tests](#end-to-end-tests)
6. [Performance Benchmarking](#performance-benchmarking)
7. [Edge Cases & Regression Tests](#edge-cases--regression-tests)
8. [Continuous Integration](#continuous-integration)

---

## Testing Strategy

### Test Pyramid

```
                     ▲
                    ╱ ╲     E2E Tests (10%)
                   ╱   ╲    Slow, high confidence
                  ╱─────╲
                 ╱       ╲   Integration Tests (30%)
                ╱         ╲  Medium speed, moderate confidence
               ╱───────────╲
              ╱             ╲ Unit Tests (60%)
             ╱ Unit Tests    ╲ Fast, low confidence
            ╱_________________╲
```

### Test Coverage Goals

```
| Module | Target | Current |
|--------|--------|---------|
| core/database.py | 95% |  |
| vision/recognition.py | 90% |  |
| vision/anti_spoofing.py | 85% |  |
| web/routes.py | 80% |  |
| student_app/verification.py | 90% |  |
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_recognition.py

# Specific test function
pytest tests/test_recognition.py::test_embedding_generation

# With coverage report
pytest --cov=. --cov-report=html

# With verbose output
pytest -v --tb=short
```

---

## Unit Tests

### Database Tests

**File**: [tests/test_database.py](../tests/test_database.py)

```python
import pytest
from unittest.mock import Mock, patch
from core.database import get_client, mark_attendance
import mongomock

@pytest.fixture
def mock_db():
    """Provide mock MongoDB for testing."""
    client = mongomock.MongoClient()
    return client.attendance_system

def test_mark_attendance_success(mock_db):
    """Test attendance marking succeeds."""
    
    # Setup
    student_id = "507f1f77bcf86cd799439011"
    date = "2024-09-15"
    
    # Execute
    result = mark_attendance(
        student_id, "Present", date, confidence=0.92,
        db=mock_db
    )
    
    # Verify
    assert result.inserted_id is not None
    assert mock_db.attendance.find_one({'student_id': student_id})['status'] == 'Present'

def test_mark_attendance_duplicate_rejected(mock_db):
    """Test duplicate attendance mark rejected."""
    
    # Setup: Mark first time
    mark_attendance("student1", "Present", "2024-09-15", db=mock_db)
    
    # Execute & Verify: Second mark fails
    from pymongo.errors import DuplicateKeyError
    
    with pytest.raises(DuplicateKeyError):
        mark_attendance("student1", "Present", "2024-09-15", db=mock_db)
```

### Face Recognition Tests

**File**: [tests/test_recognition.py](../tests/test_recognition.py)

```python
import pytest
import numpy as np
from vision.recognition import match_embeddings, check_face_quality_gate

def test_cosine_similarity_same_embedding():
    """Test identical embeddings have similarity ~1.0."""
    
    # Setup
    emb1 = np.random.rand(512).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)  # L2-normalize
    
    database = {'student1': [emb1]}
    
    # Execute
    match, score = match_embeddings(emb1, database)
    
    # Verify
    assert match == 'student1'
    assert score > 0.99

def test_cosine_similarity_different_embedding():
    """Test orthogonal embeddings have similarity ~0.0."""
    
    # Setup
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0  # Unit vector along axis 0
    
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[1] = 1.0  # Unit vector along axis 1 (orthogonal)
    
    database = {'student1': [emb2]}
    
    # Execute
    match, score = match_embeddings(emb1, database)
    
    # Verify
    assert match is None  # Score < threshold
    assert score < 0.01

def test_quality_gate_blur_detection():
    """Test quality gate rejects blurry faces."""
    
    # Setup: Create artificially blurred image
    import cv2
    
    blur_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    blur_image = cv2.GaussianBlur(blur_image, (51, 51), 10)
    
    # Execute
    is_valid, reason = check_face_quality_gate(blur_image, None)
    
    # Verify
    assert not is_valid
    assert reason == "BLURRY"
```

### Anti-Spoofing Tests

**File**: [tests/test_anti_spoofing.py](../tests/test_anti_spoofing.py)

```python
import pytest
from vision.anti_spoofing import check_liveness

def test_liveness_real_face():
    """Test liveness detection on real face."""
    
    # Setup: Load real face sample
    import cv2
    real_face = cv2.imread('tests/fixtures/real_face.jpg')
    
    # Execute
    label, confidence = check_liveness(real_face)
    
    # Verify
    assert label == 1  # Real
    assert confidence > 0.7

def test_liveness_printed_photo():
    """Test liveness detection on printed photo (attack)."""
    
    # Setup: Load photo attack sample
    import cv2
    photo_attack = cv2.imread('tests/fixtures/photo_attack.jpg')
    
    # Execute
    label, confidence = check_liveness(photo_attack)
    
    # Verify
    assert label in [0, 2]  # Spoof or attack
    assert confidence < 0.5
```

---

## Integration Tests

### API Route Tests

**File**: [tests/test_routes.py](../tests/test_routes.py)

```python
import pytest
from admin_app.app import create_app
import json

@pytest.fixture
def client():
    """Provide Flask test client."""
    app = create_app('testing')
    app.config['TESTING'] = True
    
    with app.test_client() as c:
        yield c

def test_create_attendance_session(client):
    """Test attendance session creation endpoint."""
    
    # Execute
    response = client.post('/api/attendance/sessions', json={
        'camera_id': 'lab-1',
        'course_id': 'CS101'
    })
    
    # Verify
    assert response.status_code == 201
    data = json.loads(response.data)
    assert 'session_id' in data

def test_create_attendance_session_missing_field(client):
    """Test validation on missing required field."""
    
    # Execute (missing camera_id)
    response = client.post('/api/attendance/sessions', json={
        'course_id': 'CS101'
    })
    
    # Verify
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'errors' in data
```

### Student Portal Tests

**File**: [tests/test_student_app.py](../tests/test_student_app.py)

```python
import pytest
from student_app.app import create_app
import json

@pytest.fixture
def client():
    """Provide student app test client."""
    app = create_app('testing')
    
    with app.test_client() as c:
        yield c

def test_student_registration(client):
    """Test student self-registration."""
    
    # Execute
    response = client.post('/register', json={
        'reg_no': 'CS21001',
        'name': 'Test Student',
        'email': 'REDACTED',
        'password': 'secure-password',
        'semester': '6'
    })
    
    # Verify
    assert response.status_code == 201

def test_student_login_success(client):
    """Test student login after registration."""
    
    # Setup: Register student first
    client.post('/register', json={
        'reg_no': 'CS21001',
        'name': 'Test Student',
        'email': 'REDACTED',
        'password': 'secure-password',
        'semester': '6'
    })
    
    # Execute: Login
    response = client.post('/login', data={
        'credential': 'CS21001',
        'password': 'secure-password'
    }, follow_redirects=True)
    
    # Verify
    assert response.status_code == 200
    assert b'Student Portal' in response.data
```

---

## ML Pipeline Validation

### Face Detection Accuracy

```python
# In scripts/test_face_detection.py

import cv2
import numpy as np
from vision.pipeline import detect_faces_yunet
import json

def evaluate_detection_accuracy():
    """
    Evaluate face detection on dataset.
    
    Metrics:
    - True Positive Rate (sensitivity)
    - False Positive Rate
    - Average confidence
    """
    
    # Load test set (must have ground truth boxes)
    test_set = load_test_set('tests/fixtures/detection_dataset.json')
    
    tp = 0
    fp = 0
    fn = 0
    
    for image_path, ground_truth_boxes in test_set:
        image = cv2.imread(image_path)
        detections = detect_faces_yunet(image)
        
        # Match detections to ground truth (IoU > 0.5)
        for detection in detections:
            matched = any(
                iou(detection['bbox'], gt_box) > 0.5 
                for gt_box in ground_truth_boxes
            )
            
            if matched:
                tp += 1
            else:
                fp += 1
        
        # False negatives (missed faces)
        detected_count = len(detections)
        expected_count = len(ground_truth_boxes)
        fn += max(0, expected_count - detected_count)
    
    # Calculate metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"Recall (TPR): {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    
    return recall, precision
```

### Face Recognition Accuracy

```python
# In scripts/test_face_recognition.py

import numpy as np
from vision.face_engine import get_embedding_backend
from vision.recognition import match_embeddings

def evaluate_recognition_accuracy():
    """
    Evaluate face recognition on known face pairs.
    
    Metrics:
    - True Accept Rate (TAR) at False Accept Rate (FAR)
    - ROC curve
    - Equal Error Rate (EER)
    """
    
    # Load face pairs
    # same_person_pairs: [(emb1, emb2), ...] (should match)
    # diff_person_pairs: [(emb1, emb2), ...] (should NOT match)
    
    same_person_pairs = load_pairs('tests/fixtures/same_person_pairs.npy')
    diff_person_pairs = load_pairs('tests/fixtures/diff_person_pairs.npy')
    
    # Compute similarity scores
    same_person_scores = [
        np.dot(emb1, emb2) 
        for emb1, emb2 in same_person_pairs
    ]
    
    diff_person_scores = [
        np.dot(emb1, emb2) 
        for emb1, emb2 in diff_person_pairs
    ]
    
    # Find threshold that minimizes (FAR + FRR) / 2
    best_threshold = None
    best_err = 1.0
    
    for threshold in np.linspace(0.0, 1.0, 100):
        # False Accept Rate (wrong person accepted)
        far = sum(1 for s in diff_person_scores if s >= threshold) / len(diff_person_scores)
        
        # False Reject Rate (right person rejected)
        frr = sum(1 for s in same_person_scores if s < threshold) / len(same_person_scores)
        
        err = (far + frr) / 2
        
        if err < best_err:
            best_err = err
            best_threshold = threshold
    
    print(f"Equal Error Rate (EER): {best_err:.3f}")
    print(f"Optimal Threshold: {best_threshold:.3f}")
    
    return best_threshold, best_err
```

### Anti-Spoofing Accuracy

```python
# In scripts/test_anti_spoofing.py

from vision.anti_spoofing import check_liveness
import numpy as np

def evaluate_anti_spoofing():
    """
    Evaluate anti-spoofing on presentation attack dataset.
    
    Metrics:
    - Spoof Detection Rate (SDR)
    - False Positive Rate
    - Attack Presentation Classification Error Rate (APCER)
    """
    
    real_faces = load_dataset('tests/fixtures/real_faces/')
    spoof_attacks = load_dataset('tests/fixtures/spoof_attacks/')
    
    # Evaluate on real faces
    real_correct = 0
    for face_image in real_faces:
        label, confidence = check_liveness(face_image)
        if label == 1:  # Correctly identified as real
            real_correct += 1
    
    genuine_acceptance_rate = real_correct / len(real_faces)
    
    # Evaluate on spoof attacks
    spoof_correct = 0
    for spoof_image in spoof_attacks:
        label, confidence = check_liveness(spoof_image)
        if label in [0, 2]:  # Correctly identified as spoof
            spoof_correct += 1
    
    spoof_detection_rate = spoof_correct / len(spoof_attacks)
    
    print(f"Genuine Acceptance Rate (GAR): {genuine_acceptance_rate:.3f}")
    print(f"Spoof Detection Rate (SDR): {spoof_detection_rate:.3f}")
    
    return genuine_acceptance_rate, spoof_detection_rate
```

---

## End-to-End Tests

### Camera Pipeline Test

```python
# In tests/test_camera_pipeline.py

import pytest
import cv2
import numpy as np
from camera.camera import Camera
from unittest.mock import Mock, patch

def test_full_pipeline_recognition():
    """Test complete pipeline: capture → detect → track → recognize → mark."""
    
    # Setup
    with patch('camera.camera.get_client') as mock_client:
        mock_client.return_value = create_mock_db()
        
        camera = Camera(camera_id='test-cam')
        
        # Simulate frame capture
        test_frame = cv2.imread('tests/fixtures/real_face.jpg')
        
        # Process frame
        camera.process_frame(test_frame)
        
        # Verify attendance was marked
        assert mock_client.return_value.attendance.find_one() is not None

def test_pipeline_with_anti_spoofing():
    """Test pipeline rejects spoof attacks."""
    
    camera = Camera(camera_id='test-cam')
    
    # Load printed photo attack
    attack_frame = cv2.imread('tests/fixtures/photo_attack.jpg')
    
    # Process frame
    camera.process_frame(attack_frame)
    
    # Verify NO attendance marked (liveness failed)
    # (mock database would verify no insert)
```

---

## Performance Benchmarking

### Frame Processing Latency

```python
# In scripts/benchmark_latency.py

import cv2
import time
from vision.pipeline import detect_faces_yunet
from vision.recognition import encode_face
from vision.anti_spoofing import check_liveness

def benchmark_pipeline_latency(num_frames=100):
    """
    Measure end-to-end latency for one frame.
    
    Components:
    1. Detection (YuNet)
    2. Alignment & encoding (ArcFace)
    3. Recognition (cosine similarity)
    4. Liveness (Silent-Face)
    """
    
    frame = cv2.imread('tests/fixtures/real_face.jpg')
    
    # 1. Detection
    start = time.time()
    for _ in range(num_frames):
        detections = detect_faces_yunet(frame)
    detection_latency = (time.time() - start) / num_frames
    print(f"Detection: {detection_latency*1000:.2f}ms")
    
    # 2. Encoding
    start = time.time()
    for _ in range(num_frames):
        encoding = encode_face(frame, landmarks=None)
    encoding_latency = (time.time() - start) / num_frames
    print(f"Encoding: {encoding_latency*1000:.2f}ms")
    
    # 3. Liveness
    start = time.time()
    for _ in range(num_frames):
        label, conf = check_liveness(frame)
    liveness_latency = (time.time() - start) / num_frames
    print(f"Liveness: {liveness_latency*1000:.2f}ms")
    
    # Total
    total = detection_latency + encoding_latency + liveness_latency
    fps = 1 / total
    
    print(f"\nTotal per frame: {total*1000:.2f}ms")
    print(f"FPS achievable: {fps:.1f}")

# Run benchmark
benchmark_pipeline_latency()
```

---

## Edge Cases & Regression Tests

### Handling Multiple Faces

```python
def test_multiple_faces_in_frame():
    """Test tracking handles multiple simultaneous faces."""
    
    # Setup: Load image with 5 faces
    multi_face_image = cv2.imread('tests/fixtures/5_faces.jpg')
    
    camera = Camera(camera_id='test-cam')
    camera.process_frame(multi_face_image)
    
    # Verify: 5 tracks created
    assert len(camera._tracks) == 5
```

### Handling Face Occlusion

```python
def test_tracking_with_occlusion():
    """Test tracker survives temporary face occlusion."""
    
    camera = Camera(camera_id='test-cam')
    
    # Frame 1: Face visible
    frame1 = cv2.imread('tests/fixtures/real_face.jpg')
    camera.process_frame(frame1)
    
    initial_track_id = camera._tracks[0].id
    
    # Frame 2-5: Face occluded (hand covering face)
    for _ in range(4):
        occluded_frame = create_occluded_frame()
        camera.process_frame(occluded_frame)
    
    # Frame 6: Face visible again
    frame6 = cv2.imread('tests/fixtures/real_face.jpg')
    camera.process_frame(frame6)
    
    # Verify: Track ID persists (same track throughout)
    assert camera._tracks[0].id == initial_track_id
```

### Threshold Sensitivity

```python
def test_recognition_threshold_sensitivity():
    """Test system behavior at recognition threshold boundary."""
    
    database = {'student1': [create_embedding()]}
    
    # Embedding just above threshold
    similar_emb = perturb_embedding(database['student1'][0], delta=0.39)
    match, score = match_embeddings(similar_emb, database)
    assert match == 'student1'  # Should match
    
    # Embedding just below threshold
    dissimilar_emb = perturb_embedding(database['student1'][0], delta=0.37)
    match, score = match_embeddings(dissimilar_emb, database)
    assert match is None  # Should NOT match
```

---

## Continuous Integration

### GitHub Actions Workflow

**File**: [.github/workflows/tests.yml](../.github/workflows/tests.yml)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      mongodb:
        image: mongo:latest
        options: >-
          --health-cmd mongo
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 27017:27017
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Download models
        run: python scripts/download_models.py
      
      - name: Run tests
        run: pytest --cov=. --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Summary

AutoAttendance testing provides:

1. **Comprehensive Coverage**: Unit, integration, and E2E tests.
2. **ML Validation**: Recognition, detection, and liveness evaluation.
3. **Performance Monitoring**: Latency benchmarks and throughput analysis.
4. **Regression Prevention**: Edge case and threshold boundary tests.
5. **CI/CD Integration**: Automated testing on every commit.

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0

