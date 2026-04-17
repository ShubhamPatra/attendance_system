# REST API Reference

## Table of Contents

1. [Authentication](#authentication)
2. [Attendance Sessions](#attendance-sessions)
3. [Attendance Records](#attendance-records)
4. [Students](#students)
5. [Courses](#courses)
6. [Health & Diagnostics](#health--diagnostics)
7. [Error Handling](#error-handling)

---

## Authentication

### Session-Based (Admin)

All admin endpoints require an active Flask session.

```bash
# Login
curl -X POST http://localhost:5000/login \
  -d "email=REDACTED&password=admin123"

# Cookies automatically stored by browser/curl
# Subsequent requests include session cookie

# Logout
curl -X POST http://localhost:5000/logout
```

### Flask-Login (Student)

Student endpoints use Flask-Login session management.

```bash
# Student login
curl -X POST http://localhost:5001/login \
  -d "credential=CS21001&password=student123"

# Access protected student endpoint
curl http://localhost:5001/student/attendance \
  -H "Cookie: session=..."
```

---

## Attendance Sessions

### Create Session

**Endpoint**: `POST /api/attendance/sessions`

**Description**: Start a new attendance session for a course/camera.

**Request**:
```http
POST /api/attendance/sessions HTTP/1.1
Content-Type: application/json

{
  "camera_id": "lab-1",
  "course_id": "CS101",
  "instructor_id": "507f1f77bcf86cd799439011",  // optional
  "start_time": "2024-09-15T09:00:00Z",       // optional, default: now
  "metadata": {                                  // optional
    "location": "Building A, Room 101",
    "batch": "2024-A"
  }
}
```

**Response** (201):
```json
{
  "session_id": "507f1f77bcf86cd799439012",
  "camera_id": "lab-1",
  "course_id": "CS101",
  "status": "active",
  "created_at": "2024-09-15T09:00:00Z",
  "updated_at": "2024-09-15T09:00:00Z"
}
```

**Errors**:
```json
// 400 Bad Request
{
  "errors": ["camera_id is required", "course_id is required"]
}

// 409 Conflict (session already active for camera)
{
  "error": "Active session already exists for camera: lab-1"
}
```

---

### End Session

**Endpoint**: `POST /api/attendance/sessions/{session_id}/end`

**Description**: Close attendance session and calculate statistics.

**Response** (200):
```json
{
  "session_id": "507f1f77bcf86cd799439012",
  "status": "closed",
  "created_at": "2024-09-15T09:00:00Z",
  "closed_at": "2024-09-15T11:00:00Z",
  "duration_seconds": 7200,
  "attendance_marked": {
    "present": 42,
    "absent": 8,
    "total_students": 50
  }
}
```

---

### Get Session

**Endpoint**: `GET /api/attendance/sessions/{session_id}`

**Response** (200):
```json
{
  "session_id": "507f1f77bcf86cd799439012",
  "camera_id": "lab-1",
  "course_id": "CS101",
  "status": "active",
  "created_at": "2024-09-15T09:00:00Z",
  "start_time": "2024-09-15T09:00:00Z"
}
```

---

### List Sessions

**Endpoint**: `GET /api/attendance/sessions?status=active&course_id=CS101&limit=10&skip=0`

**Query Parameters**:
- `status`: "active" | "closed"
- `course_id`: Filter by course
- `date`: Filter by date (YYYY-MM-DD)
- `limit`: Max results (default: 20)
- `skip`: Pagination offset

**Response** (200):
```json
{
  "sessions": [
    {
      "session_id": "507f1f77bcf86cd799439012",
      "camera_id": "lab-1",
      "course_id": "CS101",
      "status": "active",
      "created_at": "2024-09-15T09:00:00Z"
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 20
}
```

---

## Attendance Records

### Get Attendance

**Endpoint**: `GET /api/attendance?date=2024-09-15&course_id=CS101&status=Present`

**Query Parameters**:
- `date`: Filter by date (required)
- `course_id`: Filter by course
- `status`: "Present" | "Absent" | "Late"
- `limit`: Max results (default: 100)

**Response** (200):
```json
{
  "attendance": [
    {
      "id": "507f1f77bcf86cd799439013",
      "student_id": "507f1f77bcf86cd799439014",
      "student_name": "John Doe",
      "registration_number": "CS21001",
      "date": "2024-09-15",
      "session_id": "507f1f77bcf86cd799439012",
      "status": "Present",
      "confidence": 0.94,
      "marked_at": "2024-09-15T09:05:30Z",
      "marked_by_camera": "lab-1"
    }
  ],
  "total": 45,
  "date": "2024-09-15"
}
```

---

### Bulk Upsert Attendance

**Endpoint**: `POST /api/attendance`

**Description**: Bulk insert/update attendance marks (for manual corrections).

**Request**:
```json
{
  "marks": [
    {
      "student_id": "507f1f77bcf86cd799439014",
      "date": "2024-09-15",
      "status": "Present",
      "confidence": 0.92,
      "notes": "Manual entry"
    }
  ]
}
```

**Response** (201):
```json
{
  "inserted": 5,
  "updated": 2,
  "total": 7
}
```

---

### Update Single Attendance

**Endpoint**: `PUT /api/attendance/{attendance_id}`

**Request**:
```json
{
  "status": "Absent",
  "notes": "Student requested correction"
}
```

**Response** (200):
```json
{
  "id": "507f1f77bcf86cd799439013",
  "status": "Absent",
  "updated_at": "2024-09-15T14:30:00Z"
}
```

---

### Delete Attendance

**Endpoint**: `DELETE /api/attendance/{attendance_id}`

**Response** (204): No content

---

## Students

### Create Student

**Endpoint**: `POST /api/students`

**Request**:
```json
{
  "registration_number": "CS21001",
  "name": "John Doe",
  "email": "REDACTED",
  "semester": 6,
  "course_ids": ["CS101", "CS102"]
}
```

**Response** (201):
```json
{
  "id": "507f1f77bcf86cd799439014",
  "registration_number": "CS21001",
  "name": "John Doe",
  "created_at": "2024-09-15T10:00:00Z"
}
```

---

### Get Student

**Endpoint**: `GET /api/students/{student_id}`

**Response** (200):
```json
{
  "id": "507f1f77bcf86cd799439014",
  "registration_number": "CS21001",
  "name": "John Doe",
  "email": "REDACTED",
  "semester": 6,
  "enrollment_status": "completed",
  "face_embedding": null,
  "created_at": "2024-09-15T10:00:00Z"
}
```

---

### List Students

**Endpoint**: `GET /api/students?semester=6&limit=50&skip=0`

**Query Parameters**:
- `semester`: Filter by semester
- `course_id`: Filter by course
- `enrollment_status`: "pending" | "completed" | "rejected"
- `limit`: Max results
- `skip`: Pagination

**Response** (200):
```json
{
  "students": [
    {
      "id": "507f1f77bcf86cd799439014",
      "registration_number": "CS21001",
      "name": "John Doe",
      "semester": 6,
      "enrollment_status": "completed"
    }
  ],
  "total": 250,
  "skip": 0,
  "limit": 50
}
```

---

### Update Student

**Endpoint**: `PUT /api/students/{student_id}`

**Request**:
```json
{
  "semester": 7,
  "course_ids": ["CS201", "CS202"]
}
```

**Response** (200): Updated student object

---

### Delete Student

**Endpoint**: `DELETE /api/students/{student_id}`

**Response** (204): No content

---

## Courses

### Create Course

**Endpoint**: `POST /api/courses`

**Request**:
```json
{
  "code": "CS101",
  "name": "Introduction to Computer Science",
  "credits": 3,
  "semester": 1,
  "instructor_id": "507f1f77bcf86cd799439015"
}
```

**Response** (201):
```json
{
  "id": "507f1f77bcf86cd799439016",
  "code": "CS101",
  "name": "Introduction to Computer Science",
  "credits": 3
}
```

---

### List Courses

**Endpoint**: `GET /api/courses?semester=1`

**Response** (200):
```json
{
  "courses": [
    {
      "id": "507f1f77bcf86cd799439016",
      "code": "CS101",
      "name": "Introduction to Computer Science",
      "credits": 3,
      "semester": 1,
      "enrolled_students": 120
    }
  ],
  "total": 5
}
```

---

## Health & Diagnostics

### Health Check

**Endpoint**: `GET /health`

**Description**: System health and diagnostic information.

**Response** (200):
```json
{
  "status": "healthy",
  "timestamp": "2024-09-15T14:30:00Z",
  "uptime_seconds": 86400,
  
  "components": {
    "database": {
      "status": "connected",
      "latency_ms": 5.2,
      "connection_pool_size": 35
    },
    "models": {
      "yunet": {
        "status": "loaded",
        "memory_mb": 15.3
      },
      "arcface": {
        "status": "loaded",
        "memory_mb": 502.1
      },
      "silent_face": {
        "status": "loaded",
        "memory_mb": 45.2
      }
    },
    "camera": {
      "active_sessions": 2,
      "active_tracks": 15,
      "fps": 28.4
    }
  },
  
  "system": {
    "cpu_percent": 42.3,
    "memory_percent": 65.1,
    "disk_percent": 25.4
  }
}
```

---

### System Metrics

**Endpoint**: `GET /metrics`

**Description**: Prometheus-compatible metrics endpoint.

**Response** (200, text/plain):
```
# HELP attendance_system_recognition_calls_total Total recognition calls
# TYPE attendance_system_recognition_calls_total counter
attendance_system_recognition_calls_total{status="match"} 1250
attendance_system_recognition_calls_total{status="no_match"} 450

# HELP attendance_system_recognition_latency_ms Recognition latency
# TYPE attendance_system_recognition_latency_ms histogram
attendance_system_recognition_latency_ms_bucket{le="10"} 580
attendance_system_recognition_latency_ms_bucket{le="50"} 1680
attendance_system_recognition_latency_ms_bucket{le="100"} 1700

# HELP attendance_system_database_latency_ms Database operation latency
# TYPE attendance_system_database_latency_ms histogram
attendance_system_database_latency_ms{operation="mark_attendance"} 8.2
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "string",
  "code": "ERROR_CODE",
  "details": {},
  "timestamp": "2024-09-15T14:30:00Z"
}
```

### Common Status Codes

| Code | Meaning | Example |
|---|---|---|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created |
| 204 | No Content | DELETE succeeded |
| 400 | Bad Request | Missing required fields |
| 401 | Unauthorized | Not logged in |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Duplicate mark, session exists |
| 429 | Too Many Requests | Rate limited |
| 500 | Server Error | Unexpected error |
| 503 | Service Unavailable | Database down |

### Error Examples

**Missing Required Field** (400):
```json
{
  "error": "Bad Request",
  "code": "VALIDATION_ERROR",
  "details": {
    "camera_id": "This field is required",
    "course_id": "This field is required"
  }
}
```

**Unauthorized** (401):
```json
{
  "error": "Unauthorized",
  "code": "AUTH_FAILED",
  "details": {
    "message": "Session expired or invalid"
  }
}
```

**Not Found** (404):
```json
{
  "error": "Not Found",
  "code": "RESOURCE_NOT_FOUND",
  "details": {
    "resource": "student",
    "id": "507f1f77bcf86cd799439014"
  }
}
```

---

## Rate Limiting

### Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1694711400
```

### Limits by Endpoint

| Endpoint | Limit | Window |
|---|---|---|
| `/api/attendance/sessions` | 100 | 1 hour |
| `/api/attendance` | 1000 | 1 hour |
| `/api/students` | 500 | 1 hour |
| `/health` | Unlimited | - |

### Rate Limit Response (429)

```json
{
  "error": "Too Many Requests",
  "code": "RATE_LIMIT_EXCEEDED",
  "details": {
    "retry_after": 3600
  },
  "headers": {
    "Retry-After": "3600"
  }
}
```

**Last Updated**: April 16, 2026 | **Version**: 2.0.0

