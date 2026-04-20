# REST API Reference: Complete Endpoint Documentation

Comprehensive API reference for AutoAttendance with all endpoints, request/response examples, error codes, and WebSocket events.

---

## Base URL

```
Development: http://localhost:5000
Production: https://api.attendance-system.school.edu
```

## Authentication

All endpoints (except `/health` and `/login`) require JWT token in header:

```
Authorization: Bearer <token>
```

**Get token** (login):
```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@school.edu","password":"password"}'
```

**Response**:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600,
  "user": {"id":"FAC001","name":"Dr. Smith","role":"faculty"}
}
```

---

## API Endpoints

### Authentication Endpoints

#### POST /api/auth/login
**Purpose**: User login

**Request**:
```json
{
  "email": "smith@school.edu",
  "password": "password123"
}
```

**Response (200)**:
```json
{
  "token": "eyJhbGc...",
  "user": {
    "id": "FAC001",
    "name": "Dr. Smith",
    "email": "smith@school.edu",
    "role": "faculty"
  },
  "expires_in": 3600
}
```

**Error (401)**:
```json
{
  "error": "Invalid credentials",
  "code": "AUTH_INVALID_CREDENTIALS"
}
```

#### POST /api/auth/logout
**Purpose**: User logout (revoke token)

**Request**:
```
Headers: Authorization: Bearer <token>
```

**Response (200)**:
```json
{
  "message": "Logged out successfully"
}
```

#### POST /api/auth/refresh
**Purpose**: Refresh JWT token

**Request**:
```
Headers: Authorization: Bearer <token>
```

**Response (200)**:
```json
{
  "token": "eyJhbGc...",
  "expires_in": 3600
}
```

---

### Student Management Endpoints

#### GET /api/students
**Purpose**: List all students

**Query Parameters**:
```
?limit=50&offset=0&active=true&search=name
```

**Response (200)**:
```json
{
  "students": [
    {
      "student_id": "STU2025001",
      "name": "Shubham Patra",
      "email": "shubham@school.edu",
      "enrollment_date": "2025-01-15T10:30:00Z",
      "is_active": true,
      "embedding_count": 5
    }
  ],
  "total": 100,
  "limit": 50,
  "offset": 0
}
```

#### GET /api/students/{student_id}
**Purpose**: Get student details

**Response (200)**:
```json
{
  "student_id": "STU2025001",
  "name": "Shubham Patra",
  "email": "shubham@school.edu",
  "phone": "+91-9876543210",
  "enrollment_date": "2025-01-15T10:30:00Z",
  "is_active": true,
  "embedding_count": 5,
  "last_marked_date": "2025-01-20T10:30:00Z",
  "courses": ["CS101", "CS102"]
}
```

**Error (404)**:
```json
{
  "error": "Student not found",
  "code": "STUDENT_NOT_FOUND"
}
```

#### POST /api/students
**Purpose**: Create new student

**Request**:
```json
{
  "student_id": "STU2025002",
  "name": "Jane Doe",
  "email": "jane@school.edu",
  "phone": "+91-9876543211",
  "enrollment_status": "enrolled"
}
```

**Response (201)**:
```json
{
  "student_id": "STU2025002",
  "created_at": "2025-01-20T15:30:00Z"
}
```

**Error (409)**:
```json
{
  "error": "Student already exists",
  "code": "STUDENT_EXISTS"
}
```

#### PUT /api/students/{student_id}
**Purpose**: Update student information

**Request**:
```json
{
  "phone": "+91-9876543212",
  "is_active": true
}
```

**Response (200)**:
```json
{
  "message": "Student updated",
  "student_id": "STU2025001"
}
```

---

### Enrollment Endpoints

#### POST /api/enroll
**Purpose**: Enroll student (face capture)

**Request** (multipart form):
```
- video: <video_file>
- student_id: STU2025001
```

**Process**:
1. Extract 5 keyframes from video
2. Detect faces in each frame
3. Generate embeddings
4. Store in database

**Response (200)**:
```json
{
  "student_id": "STU2025001",
  "embeddings_captured": 5,
  "quality_scores": [0.92, 0.95, 0.88, 0.91, 0.93],
  "mean_quality": 0.92,
  "status": "enrollment_successful",
  "enrollment_id": "ENR_123456"
}
```

**Error (400)**:
```json
{
  "error": "No faces detected in video",
  "code": "NO_FACES_DETECTED"
}
```

**Error (422)**:
```json
{
  "error": "Quality threshold not met",
  "code": "LOW_QUALITY_FACES",
  "mean_quality": 0.45,
  "threshold": 0.70
}
```

---

### Attendance Endpoints

#### POST /api/attendance/mark
**Purpose**: Mark attendance (primary endpoint)

**Request** (multipart form):
```
- video: <video_file>
- student_id: STU2025001
- course_id: CS101
```

**Process**:
1. Capture 5 frames from video
2. Detect face in each frame
3. Generate embedding
4. Verify liveness (multi-layer)
5. Match against enrollment
6. Record attendance

**Response (200)** - Matched:
```json
{
  "status": "present",
  "student_id": "STU2025001",
  "course_id": "CS101",
  "marked_at": "2025-01-20T09:30:00Z",
  "recognition_confidence": 0.92,
  "liveness_confidence": 0.94,
  "message": "Attendance marked"
}
```

**Response (200)** - Spoofing attack detected:
```json
{
  "status": "spoofing_detected",
  "student_id": "STU2025001",
  "liveness_confidence": 0.22,
  "attack_type": "print_attack",
  "message": "Face liveness check failed",
  "error_code": "LIVENESS_FAILED"
}
```

**Response (200)** - No match found:
```json
{
  "status": "not_recognized",
  "recognition_confidence": 0.28,
  "message": "Face not recognized in database",
  "error_code": "FACE_NOT_MATCHED"
}
```

**Error (400)**:
```json
{
  "error": "No face detected in video",
  "code": "NO_FACE_DETECTED"
}
```

#### GET /api/attendance/{course_id}
**Purpose**: Get attendance records for course

**Query Parameters**:
```
?date=2025-01-20&student_id=STU2025001&limit=100
```

**Response (200)**:
```json
{
  "course_id": "CS101",
  "records": [
    {
      "student_id": "STU2025001",
      "marked_at": "2025-01-20T09:30:00Z",
      "status": "present",
      "recognition_confidence": 0.92,
      "liveness_confidence": 0.94,
      "device_id": "CAM001"
    }
  ],
  "summary": {
    "total_marked": 42,
    "total_present": 40,
    "total_absent": 2,
    "attendance_percentage": 95.2
  }
}
```

#### GET /api/attendance/{course_id}/report
**Purpose**: Generate attendance report

**Query Parameters**:
```
?start_date=2025-01-01&end_date=2025-01-31&format=csv
```

**Response (200)**:
```json
{
  "course_id": "CS101",
  "start_date": "2025-01-01",
  "end_date": "2025-01-31",
  "total_classes": 22,
  "report": [
    {
      "student_id": "STU2025001",
      "name": "Shubham Patra",
      "present": 20,
      "absent": 2,
      "late": 0,
      "attendance_percentage": 90.9
    }
  ]
}
```

**Format CSV**:
```bash
curl -X GET "http://localhost:5000/api/attendance/CS101/report?format=csv" \
  -H "Authorization: Bearer <token>" \
  > attendance_report.csv
```

#### POST /api/attendance/{attendance_id}/verify
**Purpose**: Manual verification by admin

**Request**:
```json
{
  "verified": true,
  "notes": "Manual verification"
}
```

**Response (200)**:
```json
{
  "message": "Attendance verified",
  "attendance_id": "ATT_123456",
  "verified_by": "FAC001"
}
```

---

### Course Endpoints

#### GET /api/courses
**Purpose**: List all courses

**Response (200)**:
```json
{
  "courses": [
    {
      "course_id": "CS101",
      "course_name": "Data Structures",
      "course_code": "CS101",
      "faculty_id": "FAC001",
      "faculty_name": "Dr. Smith",
      "enrolled_students": 45,
      "min_attendance_required": 0.75,
      "semester": "Spring2025"
    }
  ],
  "total": 12
}
```

#### GET /api/courses/{course_id}
**Purpose**: Get course details

**Response (200)**:
```json
{
  "course_id": "CS101",
  "course_name": "Data Structures",
  "course_code": "CS101",
  "credits": 3,
  "faculty_id": "FAC001",
  "faculty_name": "Dr. Smith",
  "enrolled_students": 45,
  "min_attendance_required": 0.75,
  "semester": "Spring2025",
  "schedule": {
    "monday": "09:30-11:00",
    "wednesday": "14:00-15:30",
    "friday": "09:30-11:00"
  }
}
```

#### POST /api/courses
**Purpose**: Create new course

**Request**:
```json
{
  "course_id": "CS103",
  "course_name": "Database Systems",
  "course_code": "CS103",
  "faculty_id": "FAC001",
  "credits": 3,
  "semester": "Spring2025"
}
```

**Response (201)**:
```json
{
  "course_id": "CS103",
  "created_at": "2025-01-20T15:30:00Z"
}
```

---

### Security & Audit Endpoints

#### GET /api/security/logs
**Purpose**: Get security logs (spoofing attempts, errors)

**Query Parameters**:
```
?event_type=spoofing_attempt&severity=high&limit=50&start_date=2025-01-01
```

**Response (200)**:
```json
{
  "logs": [
    {
      "log_id": "LOG_123456",
      "event_type": "spoofing_attempt",
      "severity": "medium",
      "student_id": "STU2025001",
      "device_id": "CAM001",
      "timestamp": "2025-01-20T09:30:00Z",
      "details": {
        "reason": "Liveness check failed",
        "liveness_score": 0.22,
        "attack_type": "print_attack"
      }
    }
  ],
  "total": 3
}
```

#### GET /api/security/attacks
**Purpose**: Get list of detected attacks

**Response (200)**:
```json
{
  "attacks": [
    {
      "timestamp": "2025-01-20T09:30:00Z",
      "attack_type": "print_attack",
      "device_id": "CAM001",
      "liveness_confidence": 0.22,
      "action_taken": "request_retry"
    }
  ],
  "total_attacks": 3,
  "success_rate": 0.0
}
```

---

### Health & Status Endpoints

#### GET /health
**Purpose**: Application health check (no auth required)

**Response (200)**:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "database": "connected",
  "models": "loaded",
  "timestamp": "2025-01-20T15:30:00Z"
}
```

**Error (503)**:
```json
{
  "status": "unhealthy",
  "database": "disconnected",
  "models": "failed_to_load"
}
```

#### GET /api/status
**Purpose**: System status and metrics

**Response (200)**:
```json
{
  "status": "running",
  "uptime_seconds": 3600,
  "request_count": 1250,
  "average_latency_ms": 95,
  "active_cameras": 3,
  "database": {
    "connected": true,
    "latency_ms": 2,
    "size_gb": 5.3
  },
  "models": {
    "face_detection": "loaded",
    "face_recognition": "loaded",
    "anti_spoofing": "loaded"
  }
}
```

---

## Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `AUTH_INVALID_CREDENTIALS` | 401 | Login failed |
| `AUTH_TOKEN_EXPIRED` | 401 | JWT token expired |
| `AUTH_UNAUTHORIZED` | 403 | Insufficient permissions |
| `STUDENT_NOT_FOUND` | 404 | Student doesn't exist |
| `STUDENT_EXISTS` | 409 | Duplicate student |
| `NO_FACES_DETECTED` | 400 | No face in image/video |
| `FACE_NOT_MATCHED` | 200 | Face not in database |
| `LIVENESS_FAILED` | 200 | Spoofing detected |
| `LOW_QUALITY_FACES` | 422 | Image quality too low |
| `DATABASE_ERROR` | 500 | Database operation failed |
| `MODEL_ERROR` | 500 | ML model inference failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |

---

## Rate Limiting

**Limits**:
- **Default**: 100 requests/minute per user
- **Enrollment**: 10 requests/hour
- **Attendance marking**: 60 requests/hour

**Response (429)**:
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

**Headers**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1642775400
```

---

## WebSocket Events (SocketIO)

### Connection

```javascript
const socket = io('http://localhost:5000', {
  auth: {
    token: 'eyJhbGc...'
  }
});

socket.on('connect', () => {
  console.log('Connected');
});
```

### Enrollment Events

#### emit: start_enrollment
```javascript
socket.emit('start_enrollment', {
  student_id: 'STU2025001',
  target_frames: 5
});
```

#### on: enrollment_started
```javascript
socket.on('enrollment_started', (data) => {
  console.log('Enrollment started', data);
  // {status: 'ready', camera_id: 'CAM001'}
});
```

#### on: frame_captured
```javascript
socket.on('frame_captured', (data) => {
  console.log('Frame captured', data.frame_number);
  // {frame_number: 1, quality: 0.92, faces_detected: 1}
});
```

#### on: enrollment_complete
```javascript
socket.on('enrollment_complete', (data) => {
  console.log('Enrollment done', data);
  // {
  //   status: 'success',
  //   embeddings_captured: 5,
  //   mean_quality: 0.92
  // }
});
```

### Attendance Marking Events

#### emit: start_attendance
```javascript
socket.emit('start_attendance', {
  course_id: 'CS101',
  student_id: 'STU2025001'
});
```

#### on: attendance_result
```javascript
socket.on('attendance_result', (data) => {
  if (data.status === 'present') {
    console.log('Attendance marked!');
  } else if (data.status === 'spoofing_detected') {
    console.log('Attack detected!');
  }
});
```

### Live Feed Events

#### emit: start_camera
```javascript
socket.emit('start_camera', {
  camera_id: 0,
  resolution: '640x480'
});
```

#### on: frame_update
```javascript
socket.on('frame_update', (frame_data) => {
  // Update video feed in real-time
  // frame_data: {frame: base64, detections: [...], timestamp: ...}
});
```

---

## Request/Response Examples

### Example 1: Full Attendance Flow

**Step 1: Login**
```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"smith@school.edu","password":"pass"}'
```

**Response**:
```json
{"token":"eyJhbGc...","expires_in":3600}
```

**Step 2: Mark Attendance**
```bash
curl -X POST http://localhost:5000/api/attendance/mark \
  -H "Authorization: Bearer eyJhbGc..." \
  -F "video=@attendance_video.mp4" \
  -F "student_id=STU2025001" \
  -F "course_id=CS101"
```

**Response**:
```json
{
  "status":"present",
  "recognition_confidence":0.92,
  "liveness_confidence":0.94,
  "marked_at":"2025-01-20T09:30:00Z"
}
```

### Example 2: Generate Report

```bash
curl -X GET "http://localhost:5000/api/attendance/CS101/report?start_date=2025-01-01&end_date=2025-01-31" \
  -H "Authorization: Bearer <token>"
```

---

## References

1. Flask API Documentation: https://flask.palletsprojects.com
2. Flask-RESTful: https://flask-restful.readthedocs.io
3. JWT Authentication: https://tools.ietf.org/html/rfc7519
4. SocketIO Documentation: https://python-socketio.readthedocs.io
5. RESTful API Design: https://restfulapi.net
