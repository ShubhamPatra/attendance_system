# Database Design: MongoDB Schema, Indexing, and Scalability

Complete technical explanation of MongoDB collections, schema design, binary embedding storage, indexing strategy, scalability considerations, and backup/recovery procedures.

---

## Overview

**Database**: MongoDB (NoSQL document-oriented)

**Purpose**: Store student profiles, embeddings, attendance records, and security logs.

**Scale**: 10K-100K students, millions of attendance records

**Key Advantage**: Schema flexibility + efficient binary encoding for embeddings

---

## Architecture Rationale: Why MongoDB?

### MongoDB vs PostgreSQL vs Redis

| Criteria | MongoDB | PostgreSQL | Redis |
|----------|---------|-----------|-------|
| **Schema Flexibility** | ✅ Dynamic | ❌ Fixed | N/A (cache) |
| **Binary Data** | ✅ Native BSON | ⚠️ BYTEA (slower) | ✅ Native |
| **Embedding Storage** | ✅ Efficient | ❌ PgVector overhead | ✅ Fast |
| **Scalability** | ✅ Horizontal | ⚠️ Vertical | ✅ Distributed |
| **Query Language** | ✅ Aggregation | ✅ SQL | ❌ Limited |
| **Transactions** | ⚠️ Multi-doc | ✅ Full ACID | ❌ None |
| **Cost** | Low | Medium | Low |

**Decision**: MongoDB for primary storage + Redis for caching

---

## MongoDB Collections Schema

### Collection 1: Students

**Purpose**: Store student profile information.

```javascript
db.students.insert({
    _id: ObjectId("..."),
    student_id: "STU2025001",           // Unique student ID
    name: "Shubham Patra",
    email: "shubham@school.edu",
    enrollment_date: ISODate("2025-01-15"),
    
    // Biometric data
    face_embeddings: [
        {
            embedding_id: ObjectId("..."),
            embedding: BinData(4, "..."),  // 512-D binary (2KB per embedding)
            timestamp: ISODate("2025-01-15T10:30:00Z"),
            quality_score: 0.95,           // Embedding quality (0-1)
            source: "enrollment"           // enrollment | attendance
        },
        // ... more embeddings (avg 3-5 per student)
    ],
    
    // Status
    is_active: true,
    enrollment_status: "enrolled",         // enrolled | graduated | inactive
    
    // Contact info
    phone: "+91-9876543210",
    guardian_email: "parent@email.com",
    
    // Metadata
    created_at: ISODate("2025-01-15T09:00:00Z"),
    updated_at: ISODate("2025-01-15T10:30:00Z")
})
```

**Document Size**: ~10KB per student (small embeddings can add ~2KB each)

**Rationale**:
- `face_embeddings`: Store all embeddings for accuracy comparison
- `embedding`: Binary-encoded 512-D vector (2KB vs 4KB as text)
- `quality_score`: Track embedding quality
- `timestamp`: Track when enrollment happened

### Collection 2: Attendance

**Purpose**: Record attendance marks.

```javascript
db.attendance.insert({
    _id: ObjectId("..."),
    student_id: "STU2025001",
    course_id: "CS101",
    session_id: ObjectId("..."),           // Link to attendance session
    
    // Attendance details
    marked_at: ISODate("2025-01-20T09:30:00Z"),
    status: "present",                     // present | absent | late | make_up
    
    // Recognition details
    recognized_embedding_id: ObjectId("..."),  // Which embedding matched
    recognition_confidence: 0.92,          // Confidence of match (0-1)
    liveness_confidence: 0.94,             // Confidence of liveness check
    
    // Face data
    face_metadata: {
        detection_bbox: [100, 50, 200, 300],   // Bounding box
        landmarks: [[110, 80], [190, 80], ...],  // 5 face landmarks
        alignment_angle: 5,                      // Rotation angle (degrees)
        image_quality: {
            blur: 250,
            brightness: 150,
            contrast: 45
        }
    },
    
    // Security & audit
    device_id: "CAM001",
    location: "Classroom_A",
    admin_verified: false,                 // Manual verification
    verified_by: null,                     // Admin user ID if verified
    verification_timestamp: null,
    
    notes: "Marked via automated system",
    
    created_at: ISODate("2025-01-20T09:30:00Z")
})
```

**Document Size**: ~5KB per record

**Rationale**:
- `recognized_embedding_id`: Link to enrollment embedding for traceability
- `face_metadata`: Store face details for auditing
- `liveness_confidence`: Separate from recognition for detailed analysis
- `admin_verified`: Audit trail for manual verification

### Collection 3: AttendanceSessions

**Purpose**: Group attendance marks (e.g., class on Jan 20, 2025, 9:30 AM).

```javascript
db.attendance_sessions.insert({
    _id: ObjectId("..."),
    course_id: "CS101",
    class_date: ISODate("2025-01-20"),
    start_time: "09:30:00",
    end_time: "10:30:00",
    
    // Session statistics
    total_students_expected: 45,
    total_students_marked: 42,
    total_present: 40,
    total_absent: 2,
    total_late: 0,
    
    // Security metrics
    total_faces_detected: 52,              // Including repeats, re-detections
    unique_faces: 42,                      // Unique students
    spoofing_attempts: 0,                  // Failed liveness checks
    failed_recognitions: 3,                // Couldn't match face to student
    
    // Metadata
    faculty_id: "FAC001",
    instructor_name: "Dr. Smith",
    session_type: "lecture",               // lecture | practical | tutorial
    
    // Status
    status: "completed",                   // in_progress | completed | cancelled
    created_at: ISODate("2025-01-20T09:00:00Z"),
    updated_at: ISODate("2025-01-20T10:35:00Z")
})
```

### Collection 4: Courses

**Purpose**: Course information.

```javascript
db.courses.insert({
    _id: ObjectId("..."),
    course_id: "CS101",
    course_name: "Data Structures",
    course_code: "CS101",
    semester: "Spring2025",
    
    faculty_id: "FAC001",
    enrolled_students: ["STU2025001", "STU2025002", ...],
    
    // Attendance policy
    min_attendance_required: 0.75,         // 75% minimum
    
    // Metadata
    credits: 3,
    created_at: ISODate("2024-12-01T00:00:00Z")
})
```

### Collection 5: Users (Admin/Faculty)

**Purpose**: Admin and faculty accounts.

```javascript
db.users.insert({
    _id: ObjectId("..."),
    user_id: "FAC001",
    name: "Dr. Smith",
    email: "smith@school.edu",
    role: "faculty",                       // faculty | admin | operator
    
    // Access control
    permissions: ["view_attendance", "edit_attendance", "view_reports"],
    
    // Account security
    password_hash: "$2b$12$...",           // bcrypt hash
    phone: "+91-9876543210",
    
    // Status
    is_active: true,
    last_login: ISODate("2025-01-20T14:30:00Z"),
    
    created_at: ISODate("2024-12-01T00:00:00Z"),
    updated_at: ISODate("2025-01-20T14:30:00Z")
})
```

### Collection 6: SecurityLogs

**Purpose**: Audit trail for all security-relevant events.

```javascript
db.security_logs.insert({
    _id: ObjectId("..."),
    event_type: "spoofing_attempt",        // Failed liveness check
    severity: "medium",                    // low | medium | high | critical
    
    // Details
    student_id: "STU2025001",
    device_id: "CAM001",
    timestamp: ISODate("2025-01-20T09:30:00Z"),
    
    details: {
        reason: "Face liveness check failed",
        liveness_score: 0.22,
        confidence: 0.78,
        attack_type: "print_attack"        // Suspected attack type
    },
    
    // Response
    action_taken: "request_retry",         // request_retry | alert_admin | block
    admin_notified: false,
    
    // Metadata
    user_agent: "attendance_system/1.0",
    ip_address: "192.168.1.100"
})
```

---

## Binary Encoding for Embeddings

### Problem: Text Storage vs Binary

**Text Encoding (JSON)**:
```
ArcFace embedding (512 floats):
[0.123, -0.456, 0.789, ...] 

JSON representation:
"[0.123,-0.456,0.789,...]"
Size: ~4KB per embedding
```

**Binary Encoding (BSON)**:
```
Same embedding:
BinData(4, "...")  (Float32 packed)
Size: 2KB per embedding (50% reduction)
Speed: 2× faster parsing
```

### Implementation

```python
import struct
import numpy as np

def embed_to_binary(embedding_array):
    """
    Convert 512-D embedding to binary BSON format
    
    Args:
        embedding_array: np.array of shape (512,)
    
    Returns:
        bytes: binary-encoded embedding
    """
    # Ensure float32
    embedding = embedding_array.astype(np.float32)
    
    # Pack as binary
    binary = struct.pack(f'{len(embedding)}f', *embedding)
    
    return binary

def binary_to_embed(binary_data):
    """
    Convert binary to embedding array
    """
    embedding = struct.unpack(f'{len(binary_data)//4}f', binary_data)
    return np.array(embedding)

# MongoDB storage
def store_embedding(student_id, embedding):
    """Store embedding in MongoDB"""
    binary = embed_to_binary(embedding)
    
    db.students.update_one(
        {'student_id': student_id},
        {'$push': {
            'face_embeddings': {
                'embedding_id': ObjectId(),
                'embedding': binary,  # Stored as BSON binary
                'timestamp': datetime.now(),
                'quality_score': 0.95
            }
        }}
    )

# Retrieval and comparison
def retrieve_embedding(student_id, embedding_idx=0):
    """Retrieve embedding and convert back"""
    doc = db.students.find_one(
        {'student_id': student_id},
        {'face_embeddings.embedding': 1}
    )
    
    binary_embedding = doc['face_embeddings'][embedding_idx]['embedding']
    embedding = binary_to_embed(binary_embedding)
    
    return embedding
```

---

## Indexing Strategy

### Index 1: Student ID (Primary)

```javascript
db.students.createIndex({
    student_id: 1          // Ascending order
}, {
    unique: true,          // Must be unique
    name: "idx_student_id"
})

// Usage: Fast lookup by student ID
db.students.findOne({ student_id: "STU2025001" })
```

### Index 2: Attendance Lookup (Compound)

```javascript
db.attendance.createIndex({
    student_id: 1,         // Primary
    course_id: 1,          // Secondary
    marked_at: -1          // Descending (recent first)
}, {
    name: "idx_attendance_student_course_date"
})

// Usage: Find all attendance for student in course, ordered by date
db.attendance.find({
    student_id: "STU2025001",
    course_id: "CS101"
}).sort({ marked_at: -1 })
```

### Index 3: Session Reports (Compound)

```javascript
db.attendance.createIndex({
    session_id: 1,
    student_id: 1
}, {
    name: "idx_attendance_session_student"
})

// Usage: Get attendance for specific session
db.attendance.find({
    session_id: ObjectId("..."),
    status: "present"
})
```

### Index 4: Security Log Query

```javascript
db.security_logs.createIndex({
    timestamp: -1          // Recent first
}, {
    name: "idx_security_logs_recent",
    expireAfterSeconds: 7776000  // TTL: 90 days
})

// Usage: Get recent security events
db.security_logs.find({
    timestamp: {
        $gte: ISODate("2025-01-01")
    }
}).sort({ timestamp: -1 })
```

### Index 5: Attendance Statistics (Partial)

```javascript
db.attendance.createIndex({
    course_id: 1,
    class_date: 1
}, {
    partialFilterExpression: {
        status: "present"   // Only index present attendance
    },
    name: "idx_attendance_present"
})

// Usage: Optimize "present" counting
db.attendance.find({
    course_id: "CS101",
    status: "present",
    class_date: ISODate("2025-01-20")
}).count()
```

### Index Statistics

| Index | Type | Size | Query Time | Usage |
|-------|------|------|-----------|-------|
| `student_id` | Unique | 100MB | <1ms | Frequent |
| `attendance (compound)` | Compound | 500MB | 1-5ms | Very frequent |
| `session_student` | Compound | 300MB | 1-3ms | Frequent |
| `security_logs (TTL)` | TTL | 50MB | 2-10ms | Occasional |
| **Total** | - | **950MB** | - | - |

---

## Query Optimization Examples

### Query 1: Get Today's Attendance for Student

```javascript
db.attendance.aggregate([
    {
        $match: {
            student_id: "STU2025001",
            marked_at: {
                $gte: ISODate("2025-01-20T00:00:00Z"),
                $lt: ISODate("2025-01-21T00:00:00Z")
            }
        }
    },
    {
        $lookup: {
            from: "attendance_sessions",
            localField: "session_id",
            foreignField: "_id",
            as: "session_info"
        }
    },
    {
        $project: {
            course_id: 1,
            status: 1,
            marked_at: 1,
            recognition_confidence: 1,
            "session_info.instructor_name": 1
        }
    }
])
```

**Index Used**: `idx_attendance_student_course_date`  
**Query Time**: 5-10ms

### Query 2: Attendance Report (by Course)

```javascript
db.attendance.aggregate([
    {
        $match: {
            course_id: "CS101",
            marked_at: {
                $gte: ISODate("2025-01-01"),
                $lt: ISODate("2025-02-01")
            }
        }
    },
    {
        $group: {
            _id: "$student_id",
            present_count: {
                $sum: { $cond: [{ $eq: ["$status", "present"] }, 1, 0] }
            },
            absent_count: {
                $sum: { $cond: [{ $eq: ["$status", "absent"] }, 1, 0] }
            },
            total_sessions: { $sum: 1 }
        }
    },
    {
        $project: {
            attendance_percentage: {
                $multiply: [
                    { $divide: ["$present_count", "$total_sessions"] },
                    100
                ]
            },
            present_count: 1,
            absent_count: 1,
            total_sessions: 1
        }
    },
    {
        $sort: { attendance_percentage: -1 }
    }
])
```

**Index Used**: `idx_attendance_present` (partial)  
**Query Time**: 50-100ms (for full course)

### Query 3: Security Audit

```javascript
db.security_logs.find({
    event_type: "spoofing_attempt",
    timestamp: {
        $gte: ISODate("2025-01-20")
    },
    severity: { $in: ["medium", "high", "critical"] }
}).sort({ timestamp: -1 }).limit(100)
```

**Index Used**: `idx_security_logs_recent` (TTL)  
**Query Time**: 10-20ms

---

## Scalability Strategy

### Horizontal Scaling: Sharding

**Problem**: 100K+ students × millions of records = single server can't handle.

**Solution**: Shard by `student_id`

```javascript
// Enable sharding
sh.enableSharding("attendance_db")

// Shard attendance collection
sh.shardCollection("attendance_db.attendance", {
    student_id: 1  // Shard key
})

// Shard students collection
sh.shardCollection("attendance_db.students", {
    student_id: 1
})
```

**How It Works**:
```
Cluster: 4 nodes (shards)

Shard 1: STU0001-STU0250
Shard 2: STU0251-STU0500
Shard 3: STU0501-STU0750
Shard 4: STU0751-STU1000

Query for STU0600:
Router → Hash("STU0600") → Shard 3 → Return result
```

**Benefits**:
- Distribute data across multiple servers
- Each shard handles ~25K students
- Query parallelization
- Write throughput increase

### Backup & Recovery

```bash
# Full backup
mongodump --db attendance_db --out /backups/attendance_db_$(date +%Y%m%d)

# Point-in-time recovery (with wiredTiger)
# First, enable oplog
rs.initiate()

# Backup oplog
mongodump --db local --collection oplog.rs --out /backups/oplog

# Restore to specific point in time
mongorestore --db attendance_db /backups/attendance_db_20250120
# Then restore oplog to specific timestamp
```

### Connection Pooling

```python
from pymongo import MongoClient
from pymongo.server_selectors import writable_server_selector

client = MongoClient(
    uri='mongodb://host1,host2,host3',
    maxPoolSize=50,        # Max connections per node
    minPoolSize=5,         # Min connections (always open)
    maxIdleTimeMS=30000,   # Close idle after 30s
    serverSelectionTimeoutMS=5000
)

db = client['attendance_db']

# Usage (connection automatically returned to pool)
result = db.students.find_one({'student_id': 'STU2025001'})
```

### Caching Strategy (Redis)

```python
import redis
from json import dumps, loads

redis_cache = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

def get_student_embeddings(student_id):
    """Get embeddings with caching"""
    
    # Try cache first
    cache_key = f"embeddings:{student_id}"
    cached = redis_cache.get(cache_key)
    if cached:
        return loads(cached)
    
    # Not in cache, query MongoDB
    doc = db.students.find_one(
        {'student_id': student_id},
        {'face_embeddings': 1}
    )
    
    embeddings = doc['face_embeddings']
    
    # Cache for 24 hours
    redis_cache.setex(
        cache_key,
        86400,  # 24 hours in seconds
        dumps(embeddings)
    )
    
    return embeddings
```

---

## Performance Characteristics

### Query Performance Targets

| Operation | Time | Notes |
|-----------|------|-------|
| **Find student by ID** | <1ms | Indexed |
| **Get student embeddings** | <5ms | Indexed + cached |
| **Mark attendance** | <10ms | Index write |
| **Get session attendance** | <20ms | Aggregation |
| **Generate report (100 students)** | 100-200ms | Aggregation |
| **Security audit (1000 events)** | 50-100ms | TTL index |

### Storage Calculations

```
10,000 students:
├─ Per student: ~10KB
└─ Total: ~100MB (base)

Per student: 5 embeddings × 2KB = 10KB
├─ Total embeddings: 50MB
└─ Total students collection: 150MB

1M attendance records:
├─ Per record: ~5KB
└─ Total: ~5GB

100K security logs:
├─ Per log: ~1KB
└─ Total: 100MB

Total database size: ~5.3GB (manageable, fits on single server)
Sharded across 4 nodes: ~1.3GB each
```

---

## Backup Strategy

### Daily Backups

```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/attendance_db_$DATE"

# Full database dump
mongodump --uri="mongodb://user:pass@localhost/attendance_db" \
          --out="$BACKUP_DIR"

# Compress
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"

# Upload to cloud (AWS S3)
aws s3 cp "$BACKUP_DIR.tar.gz" s3://backup-bucket/attendance_db/

# Cleanup old backups (keep last 30)
find /backups -name "attendance_db_*.tar.gz" -mtime +30 -delete
```

### Point-in-Time Recovery

```javascript
// Find all oplog entries after timestamp
db.getSiblingDB("local").oplog.rs.find({
    ts: { $gte: Timestamp(1641254400, 1) }
}).sort({ $natural: -1 })

// Restore to specific point
mongorestore --oplogReplay --oplogLimit 1641254400 /backup_path
```

---

## Conclusion

AutoAttendance Database Design achieves:

1. **Efficient Storage**: Binary embeddings save 50% space
2. **Fast Queries**: Strategic indexing for common operations
3. **Scalable Architecture**: Sharding for 100K+ students
4. **Secure Audit Trail**: All events logged to security_logs
5. **Resilient Backup**: Point-in-time recovery capability

**Key Innovation**: MongoDB's BSON binary storage + indexing strategy provides optimal balance between storage efficiency and query performance.

---

## References

1. MongoDB Official Documentation: https://docs.mongodb.com
2. MongoDB Indexing Strategy: https://docs.mongodb.com/manual/indexes/
3. MongoDB Sharding: https://docs.mongodb.com/manual/sharding/
4. BSON Specification: https://bsonspec.org/
