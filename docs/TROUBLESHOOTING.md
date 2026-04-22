# Troubleshooting Guide

## Table of Contents

1. [Startup & Configuration Issues](#startup--configuration-issues)
2. [Recognition & Detection Issues](#recognition--detection-issues)
3. [Database & Connection Issues](#database--connection-issues)
4. [Performance Issues](#performance-issues)
5. [Deployment Issues](#deployment-issues)

---

## Startup & Configuration Issues

### "ModuleNotFoundError: No module named 'cv2'"

**Symptom**:
```
Traceback (most recent call last):
  File "run.py", line 1, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
```

**Causes**:
- OpenCV not installed
- Virtual environment not activated
- Wrong Python version

**Solution**:
```bash
# 1. Verify virtual environment
which python
# Should show: /path/to/venv/bin/python

# 2. If not in venv, activate
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 3. Install OpenCV
pip install opencv-contrib-python==4.8.1.78
```

### "KeyError: SECRET_KEY not found in environment"

**Symptom**:
```
KeyError: 'SECRET_KEY'
```

**Cause**: `.env` file not created or SECRET_KEY not set.

**Solution**:
```bash
# Generate and add to .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
# abc123def456...

# Add to .env
echo "SECRET_KEY=abc123def456..." >> .env

# Verify
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('SECRET_KEY'))"
```

### "YuNet model failed to load"

**Symptom**:
```
RuntimeError: Failed to load model: face_detection_yunet_2023mar.onnx
```

**Cause**: Model file missing or corrupted.

**Solution**:
```bash
# 1. Check if exists
ls -la models/face_detection_yunet_2023mar.onnx

# 2. If not, download
python scripts/download_models.py

# 3. If download fails, manual download
cd models
wget https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
cd ..

# 4. Verify MD5 checksum
md5sum models/face_detection_yunet_2023mar.onnx
# Should match: 4b16851d2ebd1b9a87d3fe2c96b1f7a8
```

### "Port 5000 already in use"

**Symptom**:
```
OSError: [Errno 48] Address already in use
```

**Solution**:
```bash
# Option 1: Find and kill process
lsof -i :5000
kill -9 <PID>

# Option 2: Use different port
export FLASK_PORT=5555
python run.py

# Option 3: Configuration
FLASK_ENV=development FLASK_PORT=5555 python run.py
```

---

## Recognition & Detection Issues

### "No faces detected in classroom"

**Symptoms**:
- Bounding boxes don't appear
- Faces not marked as present
- Error logs show "0 detections"

**Common Causes**:

| Cause | Symptoms | Fix |
|---|---|---|
| **Lighting too dim** | Faces hard to see with camera | Increase lighting to 500+ lux; use IR camera |
| **Faces too small** | Students far from camera | Move camera closer or use wide-angle lens |
| **Wrong camera ID** | Camera not opening | Verify: `ls /dev/video*` (Linux) |
| **Detection interval too high** | Only checking every 10+ frames | Reduce: `DETECTION_INTERVAL=4` |
| **Detection confidence too high** | Threshold > 0.8 rejects borderline faces | Lower: `DETECTION_CONFIDENCE=0.5` |

**Debugging Steps**:
```bash
# 1. Test camera with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# 2. Test face detection directly
python scripts/debug_pipeline.py --camera=0 --visualize=true

# 3. Check detection settings
python -c "from core.config import *; print(f'Detection interval: {DETECTION_INTERVAL}'); print(f'Min face size: {MIN_FACE_WIDTH_PIXELS}x{MIN_FACE_HEIGHT_PIXELS}')"

# 4. Test lighting
python -c "import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read(); brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean(); print(f'Brightness: {brightness}')"
```

### "High False Positive Rate (wrong students marked)"

**Symptoms**:
- Student A's face marks Student B as present
- Same student marked multiple times with different identities

**Root Causes**:

1. **Recognition threshold too low**
   - Lenient matching
   - Fix: `RECOGNITION_THRESHOLD=0.42` (was 0.38)

2. **Insufficient confirmation frames**
   - Voting too relaxed
   - Fix: `RECOGNITION_CONFIRM_FRAMES=3` (was 2)

3. **Enrollment quality issues**
   - Student enrolled with poor-quality images
   - Fix: Re-enroll student with better samples

4. **Similar-looking students**
   - System can't distinguish twins
   - Fix: Use supplementary student ID card verification

**Diagnostic Steps**:
```python
# In admin dashboard or Python console:

# 1. Check recent attendance marks
from core.database import get_client
db = get_client().attendance_system

recent = db.attendance.find({
    'created_at': { '$gte': datetime.utcnow() - timedelta(hours=1) }
}).limit(100)

# 2. Check confidence scores
for mark in recent:
    if mark['confidence'] < 0.75:
        print(f"⚠ Low confidence mark: {mark['student_id']} ({mark['confidence']:.2f})")

# 3. Check duplicate marks (same student, same day)
from collections import Counter
from datetime import datetime

marks = db.attendance.find({ 'date': '2024-09-15' })
student_marks = Counter([m['student_id'] for m in marks])

duplicates = {sid: count for sid, count in student_marks.items() if count > 1}
if duplicates:
    print(f"⚠ Potential duplicates: {duplicates}")
```

### "High False Negative Rate (students not marked as present)"

**Symptoms**:
- Student's face visible but not marked
- Attendance records missing entries
- Confidence scores < 0.5

**Causes**:

1. **Recognition threshold too high**
   - Too strict matching
   - Fix: `RECOGNITION_THRESHOLD=0.35` (was 0.38)

2. **Liveness threshold too high**
   - Anti-spoofing too strict
   - Fix: `LIVENESS_THRESHOLD=0.50` (was 0.55)

3. **Multi-frame voting too strict**
   - Requires too many confirmations
   - Fix: `RECOGNITION_CONFIRM_FRAMES=2` (was 3)

4. **Student not enrolled**
   - Student's embedding not in database
   - Fix: Have student complete self-enrollment

5. **Significant appearance change**
   - Student's face changed (beard, glasses, weight)
   - Fix: Student re-enrolls

**Diagnostic Steps**:
```python
# Check if student exists in database
student = db.students.find_one({'registration_number': 'CS21001'})
if not student:
    print("⚠ Student not found in database")
    print("Action: Student needs to self-enroll")
else:
    print(f" Student found: {student['name']}")
    embedding = student.get('face_embedding')
    if embedding:
        print(f" Embedding present (shape: {len(embedding)})")
    else:
        print("⚠ No embedding found - student enrollment incomplete")
```

---

## Database & Connection Issues

### "MongoDB connection refused"

**Symptom**:
```
pymongo.errors.ServerSelectionTimeoutError: No servers found yet!
```

**Cause**: MongoDB not running or wrong connection string.

**Solution**:

```bash
# 1. Check if MongoDB is running (local)
sudo systemctl status mongod

# 2. If not running, start it
sudo systemctl start mongod

# 3. If using MongoDB Atlas, verify connection string
# Should look like: mongodb+srv://user:REDACTED/attendance_system?retryWrites=true&w=majority

# 4. Test connection manually
mongosh "mongodb://localhost:27017"
# or
mongosh "mongodb+srv://user:REDACTED/"

# 5. Verify environment variable
echo $MONGODB_URI
```

### "Circuit breaker open: too many database failures"

**Symptom**:
```
Exception: Circuit breaker is OPEN (too many failures)
```

**Cause**: Database connection failures reached threshold (default: 5).

**Solution**:
```bash
# 1. Check database status
mongosh
> db.serverStatus()

# 2. Check MongoDB logs
tail -f /var/log/mongodb/mongod.log

# 3. Restart MongoDB
sudo systemctl restart mongod

# 4. Increase circuit breaker timeout (if needed)
export CIRCUIT_BREAKER_TIMEOUT=120  # 2 minutes instead of 60
python run.py

# 5. Check network connectivity
ping cluster.mongodb.net  # For Atlas
telnet localhost 27017   # For local MongoDB
```

### "No space left on device"

**Symptom**:
```
OSError: [Errno 28] No space left on device
```

**Cause**: Disk full (logs, embeddings, or uploads filling up).

**Solution**:
```bash
# 1. Check disk usage
df -h

# 2. Find large directories
du -sh /* | sort -rh

# 3. Clean old logs (keep last 30 days)
find logs/ -mtime +30 -delete

# 4. Clean old uploads
find uploads/ -mtime +90 -delete

# 5. Clear model cache
rm -rf ~/.insightface/models/*.tmp
rm -rf ~/.cache/torch/*

# 6. Check MongoDB storage
# In mongosh:
> db.stats()

# Export and compress old data
mongoexport --uri="mongodb://localhost" --collection=attendance --out=attendance_backup.json
gzip attendance_backup.json
# Delete from MongoDB if needed
```

---

## Performance Issues

### "System running at low FPS (< 10 FPS)"

**Symptoms**:
- Video playback is choppy
- Attendance marking delayed
- FPS displayed < 10

**Diagnosis**:
```bash
# 1. Profile individual components
python scripts/benchmark_latency.py

# 2. Check CPU usage
top -p $(pgrep -f "python run.py")
# Should be < 80% per core

# 3. Check GPU (if available)
nvidia-smi  # Should show low memory usage if GPU enabled

# 4. Check frame processing width
python -c "from core.config import FRAME_PROCESS_WIDTH; print(f'Width: {FRAME_PROCESS_WIDTH}')"
```

**Solutions** (in order of impact):

1. **Reduce frame processing width** (3× latency reduction)
   ```bash
   export ATTENDANCE_FRAME_PROCESS_WIDTH=384  # Default: 512
   ```

2. **Increase detection interval** (5× for static scenes)
   ```bash
   export ATTENDANCE_DETECTION_INTERVAL=8  # Default: 6
   ```

3. **Enable GPU** (4–7× speedup)
   ```bash
   export ATTENDANCE_ENABLE_GPU=1
   ```

4. **Reduce number of tracked faces**
   ```bash
   export ATTENDANCE_MAX_STUDENTS_PER_SESSION=50  # Default: 500
   ```

### "Memory usage keeps increasing (memory leak)"

**Symptom**:
```
Process memory: 100MB → 500MB → 1GB (grows over hours)
```

**Common Causes**:

1. **Embedding cache growing unbounded**
   - Fix: Check TTL is working
   - `EMBEDDING_CACHE_TTL_SECONDS=2` should auto-evict

2. **Track accumulation** (not deleting old tracks)
   - Fix: Verify `_cleanup_stale_tracks()` is called
   - Default: Every 300 frames (10 seconds @ 30 FPS)

3. **Event buffer filling up**
   - Fix: Set max size: `self._events = deque(maxlen=100)`

**Debugging**:
```python
# In Python console:
import tracemalloc
tracemalloc.start()

# After 1 hour
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1e6:.1f} MB")
print(f"Peak: {peak / 1e6:.1f} MB")

tracemalloc.stop()
```

---

## Deployment Issues

### "Nginx 502 Bad Gateway after deployment"

**Symptoms**:
```
Error 502: Bad Gateway
```

**Cause**: Gunicorn not responding to Nginx.

**Solution**:
```bash
# 1. Check Gunicorn is running
ps aux | grep gunicorn

# 2. Check logs
tail -f logs/gunicorn.log

# 3. Restart Gunicorn
sudo systemctl restart gunicorn

# 4. Verify Gunicorn listening on expected socket/port
ss -tlnp | grep gunicorn
# Should show listening on 127.0.0.1:5000 or /tmp/gunicorn.sock

# 5. Check Nginx config
sudo nginx -t
sudo systemctl restart nginx
```

### "HTTPS certificate errors"

**Solution**:
```bash
# Use Let's Encrypt for free SSL/TLS
sudo apt install certbot python3-certbot-nginx

# Generate certificate
sudo certbot certonly --nginx -d your-domain.com

# Verify certificate
sudo certbot certificates

# Auto-renew (automatic)
sudo systemctl enable certbot.timer
```

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0

