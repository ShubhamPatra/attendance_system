# Database Design & MongoDB Schema

## Table of Contents

1. [Overview](#overview)
2. [Collections & Schema](#collections--schema)
3. [Indexes & Query Optimization](#indexes--query-optimization)
4. [Relationships & Denormalization](#relationships--denormalization)
5. [Session Lifecycle](#session-lifecycle)
6. [Attendance Uniqueness Enforcement](#attendance-uniqueness-enforcement)
7. [Query Patterns & Examples](#query-patterns--examples)
8. [Connection & Circuit Breaker](#connection--circuit-breaker)
9. [Backup & Recovery](#backup--recovery)

---

## Overview

AutoAttendance uses **MongoDB Atlas** (cloud) or **MongoDB Server** (on-premise) for data persistence. The design emphasizes:

- **Flexibility**: Document schema adapts to encoding dimensions (128-D dlib, 512-D ArcFace).
- **Performance**: Strategic indexes and denormalization for fast queries.
- **Reliability**: Unique indexes enforce data integrity; circuit breaker handles failures.
- **Scalability**: Sharding support for multi-institution deployments.

### Connection Configuration

```python
# In core/config.py
MONGO_URI = os.getenv(
    'MONGO_URI',
    'mongodb+srv://user:pass@cluster.mongodb.net/attendance_system'
)
MONGO_MAX_POOL_SIZE = int(os.getenv('MONGO_MAX_POOL_SIZE', 50))
MONGO_CIRCUIT_BREAKER_THRESHOLD = int(os.getenv('MONGO_CIRCUIT_BREAKER_THRESHOLD', 5))
```

---

## Collections & Schema

### 1. students Collection

Enrollment records with face encodings.

```json
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "reg_no": "CS21001",
  "name": "Alice Johnson",
  "email": "alice@university.edu",
  "phone": "+1-555-0001",
  "semester": "6",
  "section": "A",
  "department": "Computer Science",
  "status": "approved",
  "encodings": [
    BinData(0, "base64-encoded-512d-vector-1"),
    BinData(0, "base64-encoded-512d-vector-2"),
    BinData(0, "base64-encoded-512d-vector-3")
  ],
  "legacy_dlib_encoding": null,
  "enrollment_date": ISODate("2024-01-15T09:00:00.000Z"),
  "verified_at": ISODate("2024-01-16T14:30:00.000Z"),
  "verification_samples": 4,
  "created_by": ObjectId("507f1f77bcf86cd799439012")
}
```

**Schema Details**:

| Field | Type | Purpose | Indexed |
|---|---|---|---|
| `_id` | ObjectId | Primary key | Yes (automatic) |
| `reg_no` | String | Unique registration number | Yes (unique) |
| `name` | String | Full name | No |
| `email` | String | Email address | Yes (unique) |
| `phone` | String | Contact phone | No |
| `semester` | String | Academic semester | No |
| `section` | String | Class section | No |
| `department` | String | Department name | No |
| `status` | String | Enrollment status: "approved", "pending", "rejected" | Yes |
| `encodings` | Array[Binary] | Face embeddings (512-D, L2-normalized) | No |
| `legacy_dlib_encoding` | Binary | Legacy 128-D dlib (backward compat) | No |
| `enrollment_date` | ISODate | Enrollment timestamp | No |
| `verified_at` | ISODate | Verification completion time | No |
| `verification_samples` | Int32 | Number of samples captured | No |
| `created_by` | ObjectId | Admin who created record | No |

**Encoding Format**:

Each encoding is stored as binary to save space:

```python
# Serialization (saving)
embedding = np.random.rand(512).astype(np.float32)  # 512-D vector
binary_data = embedding.tobytes()  # Convert to bytes
db.students.insert_one({
    'reg_no': 'CS21001',
    'encodings': [binary_data]
})

# Deserialization (loading)
doc = db.students.find_one({'reg_no': 'CS21001'})
binary_data = doc['encodings'][0]
embedding = np.frombuffer(binary_data, dtype=np.float32)  # Restore vector
```

---

### 2. attendance Collection

Daily attendance records.

```json
{
  "_id": ObjectId("607f1f77bcf86cd799439021"),
  "student_id": ObjectId("507f1f77bcf86cd799439011"),
  "reg_no": "CS21001",
  "date": ISODate("2024-09-15T00:00:00.000Z"),
  "status": "Present",
  "marked_at": ISODate("2024-09-15T09:15:30.123Z"),
  "confidence": 0.92,
  "session_id": ObjectId("607f1f77bcf86cd799439031"),
  "camera_id": "lab-1",
  "course_id": "CS101",
  "verified": true,
  "notes": ""
}
```

**Schema Details**:

| Field | Type | Purpose | Indexed |
|---|---|---|---|
| `_id` | ObjectId | Primary key | Yes (automatic) |
| `student_id` | ObjectId | References students._id | Yes (compound) |
| `reg_no` | String | Denormalized student ID | No |
| `date` | ISODate | Attendance date (ISO date only, time normalized to 00:00) | Yes (compound) |
| `status` | String | "Present", "Absent", "Late" | Yes |
| `marked_at` | ISODate | Timestamp when marked | No |
| `confidence` | Double | Cosine similarity score [0, 1] | No |
| `session_id` | ObjectId | References attendance_sessions | No |
| `camera_id` | String | Camera ID (for audit trail) | No |
| `course_id` | String | Course code | Yes |
| `verified` | Boolean | Admin-verified flag (for disputes) | No |
| `notes` | String | Optional notes (e.g., "Marked late due to emergency") | No |

**Uniqueness Constraint**:

Only one attendance record per student per date:

```javascript
db.attendance.createIndex(
  { "student_id": 1, "date": 1 },
  { unique: true }
)
```

---

### 3. attendance_sessions Collection

Active attendance sessions (per camera, course, time period).

```json
{
  "_id": ObjectId("607f1f77bcf86cd799439031"),
  "camera_id": "lab-1",
  "course_id": "CS101",
  "session_start": ISODate("2024-09-15T09:00:00.000Z"),
  "session_end": ISODate("2024-09-15T12:00:00.000Z"),
  "status": "active",
  "created_by": ObjectId("507f1f77bcf86cd799439012"),
  "attendance_count": 45,
  "auto_closed_at": null,
  "last_activity": ISODate("2024-09-15T11:50:00.000Z")
}
```

**Schema Details**:

| Field | Type | Purpose | Indexed |
|---|---|---|---|
| `_id` | ObjectId | Primary key | Yes (automatic) |
| `camera_id` | String | Physical camera identifier | Yes (compound) |
| `course_id` | String | Course code | Yes |
| `session_start` | ISODate | Session start timestamp | Yes |
| `session_end` | ISODate | Session end timestamp | No |
| `status` | String | "active", "ended", "auto_closed" | Yes |
| `created_by` | ObjectId | Admin who started session | No |
| `attendance_count` | Int32 | Students marked present | No |
| `auto_closed_at` | ISODate | If auto-closed due to idle, timestamp | No |
| `last_activity` | ISODate | Last attendance mark or event | No |

**Uniqueness Constraint**:

Only one active session per camera:

```javascript
db.attendance_sessions.createIndex(
  { "camera_id": 1, "status": 1 },
  { unique: true, partialFilterExpression: { status: "active" } }
)
```

---

### 4. users Collection

Admin and teacher accounts.

```json
{
  "_id": ObjectId("507f1f77bcf86cd799439012"),
  "username": "admin",
  "email": "admin@university.edu",
  "password_hash": "$2b$12$...",
  "role": "admin",
  "created_at": ISODate("2024-01-01T00:00:00.000Z"),
  "last_login": ISODate("2024-09-15T09:00:00.000Z"),
  "is_active": true
}
```

**Schema Details**:

| Field | Type | Purpose | Indexed |
|---|---|---|---|
| `_id` | ObjectId | Primary key | Yes (automatic) |
| `username` | String | Unique login identifier | Yes (unique) |
| `email` | String | Email address | Yes (unique) |
| `password_hash` | String | Bcrypt hash (cost=12) | No |
| `role` | String | "admin", "teacher", "viewer" | Yes |
| `created_at` | ISODate | Account creation date | No |
| `last_login` | ISODate | Last login timestamp | No |
| `is_active` | Boolean | Account active flag | No |

**Password Hashing**:

```python
# In core/auth.py
import bcrypt

def hash_password(password):
    """Hash password with bcrypt."""
    return bcrypt.hashpw(
        password.encode('utf-8'),
        bcrypt.gensalt(rounds=12)
    )

def check_password(password, password_hash):
    """Verify password against hash."""
    return bcrypt.checkpw(
        password.encode('utf-8'),
        password_hash
    )
```

---

### 5. notification_events Collection (Optional)

Absence alerts and system events.

```json
{
  "_id": ObjectId("707f1f77bcf86cd799439041"),
  "student_id": ObjectId("507f1f77bcf86cd799439011"),
  "event_type": "absence_alert",
  "threshold": 75,
  "current_attendance": 72,
  "course_id": "CS101",
  "created_at": ISODate("2024-09-15T12:00:00.000Z"),
  "sent_to": "alice@university.edu",
  "acknowledged": false
}
```

---

## Indexes & Query Optimization

### Index Strategy

```javascript
// 1. students collection
db.students.createIndex({ "reg_no": 1 }, { unique: true })
db.students.createIndex({ "email": 1 }, { unique: true })
db.students.createIndex({ "status": 1 })

// 2. attendance collection
db.attendance.createIndex(
  { "student_id": 1, "date": 1 },
  { unique: true }
)
db.attendance.createIndex({ "date": 1, "status": 1 })
db.attendance.createIndex({ "student_id": 1, "date": 1 })
db.attendance.createIndex({ "course_id": 1, "date": 1 })

// 3. attendance_sessions collection
db.attendance_sessions.createIndex(
  { "camera_id": 1, "status": 1 },
  { unique: true, partialFilterExpression: { status: "active" } }
)
db.attendance_sessions.createIndex({ "status": 1, "session_start": 1 })

// 4. users collection
db.users.createIndex({ "username": 1 }, { unique: true })
db.users.createIndex({ "email": 1 }, { unique: true })
db.users.createIndex({ "role": 1 })
```

### Query Performance

**Fast Queries** (indexed):

```javascript
// Get student by reg_no (unique index)
db.students.findOne({ "reg_no": "CS21001" })   // ~1ms

// Get attendance for student on date (compound index)
db.attendance.findOne({
  "student_id": ObjectId(...),
  "date": ISODate("2024-09-15")
})   // ~1ms

// Get all active sessions (index on status)
db.attendance_sessions.find({ "status": "active" })   // ~5ms
```

**Slow Queries** (non-indexed):

```javascript
// Get student by name (no index)
db.students.find({ "name": "Alice" })   // ~100ms (full scan)

// Aggregation across 1M records (requires sorting)
db.attendance.aggregate([
  { $match: { "date": { $gte: ISODate("2024-01-01") } } },
  { $group: { _id: "$student_id", count: { $sum: 1 } } },
  { $sort: { count: -1 } }
])   // ~500ms (uses index on date if available)
```

---

## Relationships & Denormalization

### Foreign Key Pattern

MongoDB doesn't enforce foreign keys, but documents use `ObjectId` references:

```json
{
  "_id": ObjectId("607f1f77bcf86cd799439021"),
  "student_id": ObjectId("507f1f77bcf86cd799439011"),  // Reference
  "session_id": ObjectId("607f1f77bcf86cd799439031"),  // Reference
  ...
}
```

To fetch related documents:

```python
# In core/database.py
def get_attendance_with_student(attendance_id):
    """Fetch attendance record with student details (join)."""
    
    attendance = db.attendance.find_one({'_id': attendance_id})
    student = db.students.find_one({'_id': attendance['student_id']})
    
    return {
        'attendance': attendance,
        'student': student
    }
```

### Denormalization for Performance

To avoid joins, critical fields are denormalized:

```json
// attendance document includes student data
{
  "_id": ObjectId(...),
  "student_id": ObjectId(...),
  "reg_no": "CS21001",           // Denormalized from students
  "course_id": "CS101",          // Denormalized
  "date": ISODate(...),
  ...
}
```

**Trade-Off**:
- **Benefit**: Avoid expensive joins; faster queries.
- **Cost**: Potential data inconsistency if student info changes.

**Mitigation**:
- Only denormalize immutable fields (reg_no, course_id).
- Update denormalized data via application-level transactions.

---

## Session Lifecycle

### Attendance Session States

```
┌─────────────────────────────────────────┐
│ CREATED (admin clicks "Start Session")  │
│ status: "active"                        │
│ session_start: now                      │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    Manual End          Idle Timeout
         │              (15 min no activity)
         │                   │
    ┌────▼────┐          ┌───▼───┐
    │  ENDED  │          │ AUTO_ │
    │ (admin) │          │CLOSED │
    └─────────┘          └───────┘
```

### Auto-Close Logic

```python
# In core/database.py
def auto_close_idle_attendance_sessions(idle_timeout_seconds=900):
    """
    Auto-close sessions inactive for idle_timeout_seconds (default 15 min).
    """
    cutoff_time = datetime.utcnow() - timedelta(seconds=idle_timeout_seconds)
    
    result = db.attendance_sessions.update_many(
        {
            'status': 'active',
            'last_activity': {'$lt': cutoff_time}
        },
        {
            '$set': {
                'status': 'auto_closed',
                'session_end': datetime.utcnow(),
                'auto_closed_at': datetime.utcnow()
            }
        }
    )
    
    return result.modified_count
```

### Session Queries

```python
# Get active session for camera
def get_active_session(camera_id):
    return db.attendance_sessions.find_one({
        'camera_id': camera_id,
        'status': 'active'
    })

# Get all sessions for date range
def get_sessions_for_date_range(start_date, end_date):
    return db.attendance_sessions.find({
        'session_start': {
            '$gte': ISODate(start_date),
            '$lt': ISODate(end_date)
        }
    })
```

---

## Attendance Uniqueness Enforcement

### Unique Index

```javascript
db.attendance.createIndex(
  { "student_id": 1, "date": 1 },
  { unique: true }
)
```

### Behavior

Attempting to insert duplicate marks:

```python
# First mark (succeeds)
db.attendance.insert_one({
    'student_id': alice_id,
    'date': '2024-09-15',
    'status': 'Present',
    ...
})
# ✓ Success

# Duplicate mark (fails)
db.attendance.insert_one({
    'student_id': alice_id,
    'date': '2024-09-15',
    'status': 'Present',
    ...
})
# ✗ E11000 duplicate key error
```

### Application Handling

```python
# In web/attendance_routes.py
try:
    result = database.mark_attendance(student_id, status, date)
except pymongo.errors.DuplicateKeyError:
    # Already marked today; skip or update
    logger.info(f"Attendance already marked for {student_id} on {date}")
    return {'status': 'skipped', 'reason': 'duplicate'}
```

---

## Query Patterns & Examples

### 1. Attendance Report (Daily CSV Export)

```python
def get_daily_attendance_report(date, course_id=None):
    """
    Generate daily attendance CSV.
    
    Returns:
        List of dicts: [{'reg_no', 'name', 'status', 'confidence'}, ...]
    """
    query = {
        'date': ISODate(date)
    }
    
    if course_id:
        query['course_id'] = course_id
    
    attendance_records = list(db.attendance.find(query))
    
    # Enrich with student names
    report = []
    for rec in attendance_records:
        student = db.students.find_one({'_id': rec['student_id']})
        report.append({
            'reg_no': rec['reg_no'],
            'name': student['name'],
            'status': rec['status'],
            'confidence': rec['confidence'],
            'time': rec['marked_at']
        })
    
    return report
```

### 2. At-Risk Student Detection

```python
def get_at_risk_students(threshold_percentage=75):
    """
    Identify students with attendance below threshold.
    """
    # Aggregate: count present days per student
    pipeline = [
        {
            '$group': {
                '_id': '$student_id',
                'present_count': {
                    '$sum': {'$cond': [{'$eq': ['$status', 'Present']}, 1, 0]}
                },
                'total_count': {'$sum': 1}
            }
        },
        {
            '$project': {
                'attendance_percentage': {
                    '$multiply': [
                        {'$divide': ['$present_count', '$total_count']},
                        100
                    ]
                }
            }
        },
        {
            '$match': {
                'attendance_percentage': {'$lt': threshold_percentage}
            }
        }
    ]
    
    at_risk = list(db.attendance.aggregate(pipeline))
    
    # Enrich with student details
    for student in at_risk:
        student_doc = db.students.find_one({'_id': student['_id']})
        student['student_name'] = student_doc['name']
        student['reg_no'] = student_doc['reg_no']
    
    return at_risk
```

### 3. Duplicate Enrollment Detection

```python
def check_duplicate_enrollment(new_encoding, threshold=0.38):
    """
    Check if face encoding matches existing student (duplicate).
    
    Args:
        new_encoding: np.ndarray, 512-D float32
        threshold: Cosine similarity threshold
    
    Returns:
        (is_duplicate: bool, matched_student_id: str or None)
    """
    # Fetch all enrolled students
    students = list(db.students.find({'status': 'approved'}))
    
    best_match = None
    best_score = 0.0
    
    for student in students:
        for db_encoding_bytes in student['encodings']:
            db_encoding = np.frombuffer(db_encoding_bytes, dtype=np.float32)
            
            # Cosine similarity
            similarity = np.dot(new_encoding, db_encoding)
            
            if similarity > best_score:
                best_score = similarity
                best_match = student['_id']
    
    if best_score >= threshold:
        return True, best_match
    else:
        return False, None
```

### 4. Heatmap Data (Attendance by Time of Day)

```python
def get_attendance_heatmap(start_date, end_date, course_id=None):
    """
    Generate heatmap: time slot vs. course attendance.
    
    Returns:
        Dict: {date: {hour: count}}
    """
    query = {
        'date': {
            '$gte': ISODate(start_date),
            '$lte': ISODate(end_date)
        },
        'status': 'Present'
    }
    
    if course_id:
        query['course_id'] = course_id
    
    pipeline = [
        {'$match': query},
        {
            '$group': {
                '_id': {
                    'date': '$date',
                    'hour': {'$hour': '$marked_at'}
                },
                'count': {'$sum': 1}
            }
        },
        {'$sort': {'_id.date': 1, '_id.hour': 1}}
    ]
    
    heatmap_data = list(db.attendance.aggregate(pipeline))
    
    # Transform to friendly format
    heatmap = {}
    for entry in heatmap_data:
        date_key = str(entry['_id']['date'])
        hour = entry['_id']['hour']
        count = entry['count']
        
        if date_key not in heatmap:
            heatmap[date_key] = {}
        
        heatmap[date_key][hour] = count
    
    return heatmap
```

---

## Connection & Circuit Breaker

### Connection Initialization

```python
# In core/database.py
from pymongo import MongoClient
from pymongo.errors import ConnectionError, ServerSelectionTimeoutError

def _create_connection():
    """Create MongoDB connection with pooling."""
    client = MongoClient(
        MONGO_URI,
        maxPoolSize=MONGO_MAX_POOL_SIZE,
        minPoolSize=10,
        maxIdleTimeMS=45000,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=10000,
        retryWrites=True
    )
    
    # Verify connection
    client.admin.command('ping')
    return client

# Global connection
_client = None
_breaker = CircuitBreaker(
    failure_threshold=MONGO_CIRCUIT_BREAKER_THRESHOLD,
    timeout=60
)

def get_client():
    """Get MongoDB client with circuit breaker."""
    global _client
    
    try:
        return _breaker.call(_create_connection)
    except CircuitBreakerException:
        logger.error("Database circuit open; using cached data")
        raise
```

### Circuit Breaker Class

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.opened_at = None
    
    def is_open(self):
        """Check if circuit is open."""
        if self.opened_at is None:
            return False
        
        elapsed = time.time() - self.opened_at
        if elapsed > self.timeout:
            # Timeout expired; try again (half-open state)
            self.failure_count = 0
            self.opened_at = None
            return False
        
        return True
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.is_open():
            raise CircuitBreakerException("Circuit open")
        
        try:
            result = func(*args, **kwargs)
            self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise
    
    def record_failure(self):
        """Record failure and open circuit if threshold exceeded."""
        self.failure_count += 1
        
        if self.failure_count >= self.failure_threshold:
            self.opened_at = time.time()
            logger.warning(f"Circuit breaker opened; timeout={self.timeout}s")
    
    def reset(self):
        """Reset circuit state."""
        self.failure_count = 0
        self.opened_at = None
```

---

## Backup & Recovery

### Automated Backups (MongoDB Atlas)

MongoDB Atlas provides automated daily backups:

```bash
# Configure in MongoDB Atlas console:
# 1. Enable "Continuous Backups"
# 2. Set retention to 35 days (default)
# 3. Configure backup snapshots (hourly, daily, weekly)
```

### Manual Backup (mongodump)

```bash
# Export entire database
mongodump \
  --uri="mongodb+srv://user:pass@cluster.mongodb.net/attendance_system" \
  --out=./backups/$(date +%Y-%m-%d)

# Restore from backup
mongorestore \
  --uri="mongodb+srv://user:pass@cluster.mongodb.net/attendance_system" \
  ./backups/2024-09-15
```

### Point-in-Time Recovery (Atlas)

MongoDB Atlas enables recovery to specific timestamps:

```bash
# Via MongoDB Atlas console:
# 1. Select database cluster
# 2. Backup → Restore
# 3. Choose point-in-time or backup snapshot
# 4. Specify target cluster
```

---

## Summary

AutoAttendance's MongoDB schema provides:

1. **Flexibility**: Handles 128-D and 512-D encodings transparently.
2. **Performance**: Strategic indexing; denormalization for fast queries.
3. **Integrity**: Unique indexes enforce data consistency.
4. **Resilience**: Circuit breaker handles connection failures.
5. **Scalability**: Sharding support for large institutions.

Next steps:
- See [BACKEND.md](BACKEND.md) for Flask API design.
- See [DEPLOYMENT.md](DEPLOYMENT.md) for MongoDB Atlas setup.

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0
