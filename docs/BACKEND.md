# Flask Backend & REST API Design

## Table of Contents

1. [Application Structure](#application-structure)
2. [Flask App Factory Pattern](#flask-app-factory-pattern)
3. [Blueprint & Route Organization](#blueprint--route-organization)
4. [Authentication & RBAC](#authentication--rbac)
5. [REST API Endpoints](#rest-api-endpoints)
6. [WebSocket (SocketIO) Events](#websocket-socketio-events)
7. [Error Handling & HTTP Status Codes](#error-handling--http-status-codes)
8. [Request Validation](#request-validation)
9. [Session Management](#session-management)

---

## Application Structure

AutoAttendance runs two separate Flask applications:

```
┌──────────────────────┐         ┌──────────────────────┐
│   Admin Application  │         │  Student Application │
│   (admin_app/app.py) │         │(student_app/app.py)  │
│   Port 5000          │         │   Port 5001          │
├──────────────────────┤         ├──────────────────────┤
│  SocketIO (real-time)        │ ✗ SocketIO (async)   │
│  Camera feeds                │ ✗ Light-weight       │
│  Analytics                   │  Registration       │
│  Admin dashboard             │  Attendance view    │
└──────────────────────┘         └──────────────────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
            ┌────────────┴────────────┐
            │  Shared Modules:        │
            │  - web/routes.py        │
            │  - core/database.py     │
            │  - core/models.py       │
            │  - vision/*             │
            └─────────────────────────┘
```

### Admin Application

**Entry Point**: [admin_app/app.py](../admin_app/app.py)

```python
def create_app(config_name='development'):
    """Admin application factory."""
    app = Flask(__name__)
    
    # Configuration
    app.config.from_object(get_config(config_name))
    
    # Initialize extensions
    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")
    login_manager.init_app(app)
    
    # Startup diagnostics
    with app.app_context():
        _startup_diagnostics()
    
    # Register blueprints
    from web.routes import register_all_routes
    bp = Blueprint('main', __name__)
    register_all_routes(bp)
    app.register_blueprint(bp)
    
    # SocketIO handlers
    @socketio.on('connect', namespace='/admin')
    def on_connect():
        logger.info(f"Admin client connected: {request.sid}")
    
    return app
```

### Student Application

**Entry Point**: [student_app/app.py](../student_app/app.py)

```python
def create_app(config_name='development'):
    """Student portal application factory."""
    app = Flask(__name__)
    
    # Configuration
    app.config.from_object(get_config(config_name))
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    
    # Initialize authentication
    from student_app.auth import init_auth
    init_auth(login_manager, app)
    
    # Register blueprints
    from student_app.routes import register_student_routes
    bp = Blueprint('main', __name__)
    register_student_routes(bp)
    app.register_blueprint(bp)
    
    return app
```

---

## Flask App Factory Pattern

### Benefits

```python
# Enables testing with different configurations
app_test = create_app('testing')
app_prod = create_app('production')

# Late binding of dependencies
db.init_app(app)
socketio.init_app(app)

# Easy configuration management
app.config.from_object(Config)
app.config.from_env_file('.env')
```

### Configuration Hierarchy

```python
# In core/config.py
class Config:
    """Base configuration."""
    DEBUG = False
    TESTING = False
    MONGO_URI = os.getenv('MONGO_URI')
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    MONGO_URI = 'mongomock://localhost'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    # Use environment variables for sensitive config
```

---

## Blueprint & Route Organization

### Central Coordinator

**File**: [web/routes.py](../web/routes.py)

```python
def register_all_routes(bp):
    """Register all sub-blueprints into main blueprint."""
    
    from web.attendance_routes import register_attendance_routes
    from web.camera_routes import register_camera_routes
    from web.registration_routes import register_registration_routes
    from web.student_routes import register_student_routes
    from web.report_routes import register_report_routes
    from web.auth_routes import register_auth_routes
    from web.health_routes import register_health_routes
    
    # Register each module's routes
    register_attendance_routes(bp)
    register_camera_routes(bp)
    register_registration_routes(bp)
    register_student_routes(bp)
    register_report_routes(bp)
    register_auth_routes(bp)
    register_health_routes(bp)
```

### Modular Route Definitions

**Example**: [web/attendance_routes.py](../web/attendance_routes.py)

```python
def register_attendance_routes(bp):
    """Register attendance-related endpoints."""
    
    @bp.route('/api/attendance/sessions', methods=['POST'])
    @require_login
    @require_roles('admin', 'teacher')
    def create_session():
        """Start new attendance session."""
        data = request.get_json()
        
        # Validation
        errors = _validate_session_creation(data)
        if errors:
            return {'errors': errors}, 400
        
        # Create session
        daos = _build_daos()
        session_id = daos['session'].create_session(
            camera_id=data['camera_id'],
            course_id=data['course_id']
        )
        
        return {'session_id': str(session_id)}, 201
    
    @bp.route('/api/attendance/sessions/<session_id>/end', methods=['POST'])
    @require_roles('admin', 'teacher')
    def end_session(session_id):
        """End attendance session."""
        daos = _build_daos()
        daos['session'].end_session(session_id)
        
        return {'status': 'ended'}, 200
```

### Blueprint Registration Pattern

```python
def register_attendance_routes(bp):
    """Register routes on existing blueprint."""
    # All routes use @bp.route, not @app.route
    @bp.route('/api/attendance', methods=['GET'])
    def get_attendance():
        pass
```

**Advantages**:
- Modular: Each route file focuses on one domain.
- Testable: Mock blueprint for unit tests.
- Maintainable: Easy to locate route definitions.

---

## Authentication & RBAC

### Session-Based Authentication (Admin)

```python
# In web/auth_routes.py
@bp.route('/login', methods=['GET', 'POST'])
def login():
    """Admin login endpoint."""
    
    if request.method == 'GET':
        return render_template('login.html')
    
    # POST: Process login
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Verify credentials
    user = database.get_user_by_username(username)
    
    if not user or not check_password(password, user['password_hash']):
        flash('Invalid credentials', 'error')
        return redirect(url_for('login'))
    
    # Create session
    session['user_id'] = str(user['_id'])
    session['username'] = user['username']
    session['role'] = user['role']
    
    return redirect(url_for('dashboard'))
```

### Flask-Login Integration (Student Portal)

```python
# In student_app/auth.py
from flask_login import UserMixin

class StudentUser(UserMixin):
    """Student user class for Flask-Login."""
    
    def __init__(self, student_id, reg_no, email):
        self.id = student_id
        self.reg_no = reg_no
        self.email = email
    
    def get_id(self):
        return self.id

def authenticate_student(credential, password):
    """Authenticate student by reg_no or email."""
    
    # Find student by reg_no or email
    student = database.get_student_by_credential(credential)
    
    if not student:
        return None
    
    # Verify password
    if not check_password(password, student['password_hash']):
        return None
    
    return StudentUser(student['_id'], student['reg_no'], student['email'])

def init_auth(login_manager, app):
    """Initialize Flask-Login for student portal."""
    
    login_manager.init_app(app)
    login_manager.login_view = 'student_login'
    
    @login_manager.user_loader
    def load_user(user_id):
        student = database.get_student_by_id(user_id)
        if student:
            return StudentUser(user_id, student['reg_no'], student['email'])
        return None
```

### RBAC Decorators (Current: No-Op)

**File**: [web/decorators.py](../web/decorators.py)

```python
from functools import wraps

def require_login(f):
    """Decorator to enforce login (currently no-op for compatibility)."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # TODO: Implement when ENABLE_RBAC=1
        # if 'user_id' not in session:
        #     return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def require_roles(*roles):
    """Decorator to enforce role-based access (currently no-op)."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # TODO: Implement when ENABLE_RBAC=1
            # user_role = session.get('role')
            # if user_role not in roles:
            #     return {'error': 'Unauthorized'}, 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator
```

**Note**: RBAC decorators are currently **no-op placeholders** for compatibility. Full role enforcement can be enabled by setting `ENABLE_RBAC=1` in `.env` and implementing the decorator logic.

---

## REST API Endpoints

### Admin API Reference

#### Attendance Management

**POST /api/attendance/sessions**

Start new attendance session.

```http
POST /api/attendance/sessions HTTP/1.1
Content-Type: application/json

{
  "camera_id": "lab-1",
  "course_id": "CS101"
}
```

**Response** (201 Created):

```json
{
  "session_id": "507f1f77bcf86cd799439031"
}
```

---

**POST /api/attendance/sessions/<session_id>/end**

End attendance session.

```http
POST /api/attendance/sessions/507f1f77bcf86cd799439031/end HTTP/1.1
```

**Response** (200 OK):

```json
{
  "status": "ended"
}
```

---

**GET /api/attendance/sessions/active**

Get active session for camera.

```http
GET /api/attendance/sessions/active?camera_id=lab-1 HTTP/1.1
```

**Response** (200 OK):

```json
{
  "_id": "507f1f77bcf86cd799439031",
  "camera_id": "lab-1",
  "course_id": "CS101",
  "session_start": "2024-09-15T09:00:00Z",
  "status": "active",
  "attendance_count": 45
}
```

---

#### Student Management

**POST /api/register**

Register single student.

```http
POST /api/register HTTP/1.1
Content-Type: application/json

{
  "reg_no": "CS21001",
  "name": "Alice Johnson",
  "email": "REDACTED",
  "semester": "6",
  "section": "A",
  "password": "secure-password"
}
```

**Response** (201 Created):

```json
{
  "student_id": "507f1f77bcf86cd799439011",
  "reg_no": "CS21001",
  "status": "pending"
}
```

---

**PATCH /api/admin/students/<reg_no>**

Update student profile.

```http
PATCH /api/admin/students/CS21001 HTTP/1.1
Content-Type: application/json

{
  "name": "Alice J. Johnson",
  "semester": "7"
}
```

**Response** (200 OK):

```json
{
  "reg_no": "CS21001",
  "updated_fields": ["name", "semester"]
}
```

---

#### Reporting

**GET /api/reports/csv**

Export attendance as CSV.

```http
GET /api/reports/csv?start_date=2024-09-01&end_date=2024-09-30&course_id=CS101 HTTP/1.1
```

**Response** (200 OK):

```
reg_no,name,date,status,confidence,time
CS21001,Alice Johnson,2024-09-15,Present,0.92,09:15:30
CS21002,Bob Smith,2024-09-15,Present,0.88,09:16:00
...
```

---

### Student Portal API Reference

**POST /register**

Student self-registration.

```http
POST /register HTTP/1.1
Content-Type: application/json

{
  "reg_no": "CS21001",
  "name": "Alice Johnson",
  "email": "REDACTED",
  "semester": "6",
  "password": "secure-password"
}
```

---

**POST /api/capture**

Submit face samples for verification.

```http
POST /api/capture HTTP/1.1
Content-Type: application/json

{
  "frames": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRg==...",
    "data:image/jpeg;base64,/9j/4AAQSkZJRg==...",
    "data:image/jpeg;base64,/9j/4AAQSkZJRg==..."
  ]
}
```

**Response** (200 OK):

```json
{
  "status": "auto_approved",
  "score": 87,
  "feedback": "Enrollment successful"
}
```

---

**GET /attendance**

View attendance history.

```http
GET /attendance?start_date=2024-09-01&end_date=2024-09-30 HTTP/1.1
```

**Response** (200 OK):

```json
[
  {
    "date": "2024-09-15",
    "status": "Present",
    "confidence": 0.92,
    "time": "09:15:30"
  },
  {
    "date": "2024-09-14",
    "status": "Absent",
    "confidence": 0.0,
    "time": null
  }
]
```

---

## WebSocket (SocketIO) Events

### Admin Dashboard (Real-Time Feed)

#### Server → Client (Emit)

**connect**

```javascript
// Client auto-connects; server confirms
socket.on('connect', function() {
  console.log('Connected to admin dashboard');
});
```

---

**frame**

MJPEG video stream frame.

```json
{
  "data": "base64-encoded-jpeg-frame",
  "timestamp": "2024-09-15T09:15:23.123Z",
  "tracked_faces": 3
}
```

---

**attendance_event**

Real-time attendance mark.

```json
{
  "student_id": "507f1f77bcf86cd799439011",
  "student_name": "Alice Johnson",
  "reg_no": "CS21001",
  "confidence": 0.92,
  "timestamp": "2024-09-15T09:15:30.123Z",
  "camera_id": "lab-1"
}
```

---

**camera_status**

Periodic status update.

```json
{
  "fps": 22.5,
  "tracked_faces": 3,
  "session_active": true,
  "attendance_count": 45,
  "unknown_faces": 2
}
```

---

#### Client → Server (On)

**camera_control**

Admin controls camera (pause, resume, adjust threshold).

```javascript
socket.emit('camera_control', {
  'action': 'set_threshold',
  'threshold': 0.40
});
```

---

## Error Handling & HTTP Status Codes

### Standard HTTP Responses

| Status | Meaning | Example |
|---|---|---|
| 200 OK | Success | GET /api/attendance |
| 201 Created | Resource created | POST /api/attendance/sessions |
| 400 Bad Request | Invalid input | Missing required field |
| 401 Unauthorized | Not authenticated | Missing login credentials |
| 403 Forbidden | Insufficient permissions | Wrong role |
| 404 Not Found | Resource not found | Session ID doesn't exist |
| 409 Conflict | Duplicate/conflict | Attendance already marked |
| 500 Internal Error | Server error | Database failure |

### Error Response Format

```json
{
  "error": "Descriptive error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "reason"
  }
}
```

**Example**:

```json
{
  "error": "Validation failed",
  "code": "INVALID_INPUT",
  "details": {
    "camera_id": "Required field",
    "course_id": "Invalid format"
  }
}
```

### Application Exception Handling

```python
# In admin_app/app.py
@app.errorhandler(ValueError)
def handle_value_error(error):
    return {
        'error': str(error),
        'code': 'INVALID_VALUE'
    }, 400

@app.errorhandler(Exception)
def handle_exception(error):
    logger.error(f"Unhandled exception: {error}")
    return {
        'error': 'Internal server error',
        'code': 'INTERNAL_ERROR'
    }, 500
```

---

## Request Validation

### Input Validation

```python
# In web/routes_helpers.py
def validate_session_creation(data):
    """Validate session creation request."""
    
    errors = {}
    
    # Check required fields
    if not data.get('camera_id'):
        errors['camera_id'] = 'Required'
    
    if not data.get('course_id'):
        errors['course_id'] = 'Required'
    
    # Validate format
    if data.get('camera_id') and len(data['camera_id']) > 100:
        errors['camera_id'] = 'Invalid format'
    
    return errors if errors else None
```

### Request Schema Validation (Optional: Marshmallow)

```python
from marshmallow import Schema, fields, validate

class SessionCreationSchema(Schema):
    """Schema for session creation request."""
    camera_id = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    course_id = fields.Str(required=True, validate=validate.Length(min=1, max=100))

# Usage
schema = SessionCreationSchema()

try:
    result = schema.load(request.get_json())
except ValidationError as e:
    return {'errors': e.messages}, 400
```

---

## Session Management

### Admin Session Configuration

```python
# In core/config.py
SESSION_COOKIE_SECURE = True        # HTTPS only
SESSION_COOKIE_HTTPONLY = True      # JavaScript cannot access
SESSION_COOKIE_SAMESITE = 'Lax'     # CSRF protection
PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
```

### Student Session (Flask-Login)

```python
# In student_app/routes.py
@app.route('/login', methods=['POST'])
def student_login():
    """Student login endpoint."""
    
    credential = request.form.get('credential')  # reg_no or email
    password = request.form.get('password')
    
    user = authenticate_student(credential, password)
    
    if user:
        login_user(user, remember=request.form.get('remember_me', False))
        return redirect(url_for('student_portal'))
    
    flash('Invalid credentials', 'error')
    return redirect(url_for('student_login'))
```

---

## Summary

AutoAttendance's Flask backend provides:

1. **Modularity**: Blueprint pattern for organized routes.
2. **Security**: Session-based + Flask-Login authentication, optional RBAC.
3. **Real-Time**: SocketIO for live camera feeds and attendance events.
4. **Robustness**: Comprehensive error handling and input validation.
5. **Scalability**: Stateless design compatible with Gunicorn worker pools.

Next steps:
- See [DEPLOYMENT.md](DEPLOYMENT.md) for production setup with Gunicorn + Nginx.
- See [TESTING.md](TESTING.md) for unit and integration tests.

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0

