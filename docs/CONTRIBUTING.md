# Contributing Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Standards](#code-standards)
4. [Testing Requirements](#testing-requirements)
5. [Pull Request Process](#pull-request-process)
6. [Reporting Issues](#reporting-issues)

---

## Getting Started

### Prerequisites

- Git
- Python 3.9 or higher
- MongoDB (local or Atlas)
- NVIDIA GPU (optional, for GPU development)

### Fork & Clone

```bash
# 1. Fork on GitHub (click "Fork" button)

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/attendance_system.git
cd attendance_system

# 3. Add upstream remote
git remote add upstream https://github.com/ShubhamPatra/attendance_system.git

# 4. Create development branch
git checkout -b feature/your-feature-name
```

### Setup Development Environment

```bash
# 1. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# 2. Install dependencies (including dev tools)
pip install -r requirements.txt
pip install -r requirements/dev.txt  # pytest, flake8, black

# 3. Download models
python scripts/download_models.py

# 4. Setup pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

---

## Development Workflow

### Creating a Feature Branch

```bash
# Create branch from main
git checkout -b feature/my-new-feature

# Or bug fix
git checkout -b fix/issue-123

# Branch naming conventions:
# - feature/description-of-feature
# - fix/issue-number-or-name
# - docs/documentation-improvement
# - test/test-suite-improvement
```

### Making Changes

```bash
# 1. Make changes to code
vim vision/pipeline.py  # Edit file

# 2. Run tests locally
pytest tests/

# 3. Check code style
flake8 vision/pipeline.py
black vision/pipeline.py  # Auto-format

# 4. Commit changes
git add vision/pipeline.py
git commit -m "Improve YuNet detection accuracy

- Add confidence filtering for low-quality detections
- Reduce false positive rate by 5%
- Add unit tests for quality gating

Fixes #123"
```

### Commit Message Conventions

```
[type]/[scope]: [subject]

[body]

[footer]

Types: feature, fix, docs, style, refactor, test, chore
Scope: pipeline, recognition, database, deployment, etc.
Subject: 50 chars max, imperative mood

Example:
feature/recognition: Add ArcFace confidence caching

- Cache embeddings for 2 seconds to reduce latency
- Skip redundant encodings for same face
- Add cache TTL configuration parameter

Improves FPS from 15 to 22 FPS with 5 tracked faces.
Fixes #156
```

### Before Pushing

```bash
# 1. Sync with upstream (get latest changes)
git fetch upstream
git rebase upstream/main

# 2. Run all tests
pytest --cov=.

# 3. Check code style
black --check .
flake8 .

# 4. Verify no merge conflicts
git status

# 5. Push to your fork
git push origin feature/my-new-feature
```

---

## Code Standards

### Python Style Guide

Follow **PEP 8** with these tools:

```bash
# 1. Format code automatically
black vision/pipeline.py

# 2. Check code style
flake8 vision/pipeline.py
# Max line length: 100 chars
# Ignored: E203, W503 (black compatibility)

# 3. Type hints (required for public functions)
def match_embeddings(embedding: np.ndarray, db: dict) -> tuple[str, float]:
    """Match embedding to database.
    
    Args:
        embedding: L2-normalized 512-D array
        db: Dictionary of {student_id: [embeddings]}
    
    Returns:
        (student_id, similarity) or (None, 0.0) if no match
    """
    ...
```

### Documentation Standards

**Docstrings** (Google style):

```python
def detect_faces_yunet(frame: np.ndarray) -> list[dict]:
    """Detect faces using YuNet ONNX model.
    
    Args:
        frame: Input image (H x W x 3, BGR)
    
    Returns:
        List of detections, each dict with:
        - bbox: [x1, y1, x2, y2]
        - confidence: float (0–1)
    
    Raises:
        RuntimeError: If YuNet model not loaded
    
    Example:
        >>> frame = cv2.imread('test.jpg')
        >>> dets = detect_faces_yunet(frame)
        >>> print(f"Found {len(dets)} faces")
    """
    ...
```

### Code Organization

```
core/
├── __init__.py
├── config.py          # Configuration parameters
├── auth.py           # Authentication & password hashing
├── database.py       # MongoDB interface
├── models.py         # Data Access Objects (DAO)
├── extensions.py     # Flask extensions (db, mail, etc.)
└── utils.py          # Utility functions

vision/
├── __init__.py
├── pipeline.py       # Detection & tracking
├── recognition.py    # Alignment & encoding
├── face_engine.py    # Embedding backend
├── anti_spoofing.py  # Liveness detection
└── tracker.py        # CSRT/KCF tracking (if separate)

camera/
├── __init__.py
├── camera.py         # Main camera class
└── utils.py          # Camera utilities

web/
├── __init__.py
├── routes.py         # Blueprint coordinator
├── auth_routes.py    # Authentication endpoints
├── attendance_routes.py  # Attendance endpoints
├── camera_routes.py  # Camera endpoints
├── student_routes.py # Student endpoints
└── ...
```

### Naming Conventions

```python
# Constants: SCREAMING_SNAKE_CASE
RECOGNITION_THRESHOLD = 0.38
MAX_TRACK_AGE = 30

# Classes: PascalCase
class FaceTrack:
    pass

class EmbeddingBackend:
    pass

# Functions: snake_case
def detect_faces_yunet():
    pass

def encode_face():
    pass

# Private: _leading_underscore
def _cleanup_stale_tracks():
    pass

class _InternalHelper:
    pass

# Avoid: Single letter variables (except loop indices)
#  Good
for track in tracks:
    process_track(track)

# ✗ Bad
for t in tracks:
    process_track(t)
```

---

## Testing Requirements

### Unit Tests

```python
# tests/test_recognition.py

import pytest
import numpy as np
from vision.recognition import match_embeddings

def test_match_embeddings_exact_match():
    """Test matching identical embeddings."""
    
    # Setup
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    
    database = {'student1': [embedding]}
    
    # Execute
    student_id, similarity = match_embeddings(embedding, database)
    
    # Assert
    assert student_id == 'student1'
    assert similarity > 0.99

def test_match_embeddings_no_match():
    """Test unrelated embeddings don't match."""
    
    # Setup
    emb1 = np.array([1, 0, 0, ...], dtype=np.float32)  # Unit vector
    emb2 = np.array([0, 1, 0, ...], dtype=np.float32)  # Orthogonal
    
    database = {'student1': [emb2]}
    
    # Execute
    student_id, similarity = match_embeddings(emb1, database)
    
    # Assert
    assert student_id is None
    assert similarity < 0.01

@pytest.mark.parametrize("threshold", [0.35, 0.38, 0.42])
def test_threshold_sensitivity(threshold):
    """Test system behavior at different thresholds."""
    # Parameterized test runs for each threshold
    ...
```

### Integration Tests

```bash
# tests/test_routes.py (test API endpoints)

def test_create_attendance_session(client):
    """Test session creation via API."""
    
    response = client.post('/api/attendance/sessions', json={
        'camera_id': 'lab-1',
        'course_id': 'CS101'
    })
    
    assert response.status_code == 201
    data = json.loads(response.data)
    assert 'session_id' in data
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_recognition.py

# Specific test
pytest tests/test_recognition.py::test_match_embeddings_exact_match

# With coverage
pytest --cov=. --cov-report=html

# Verbose
pytest -v --tb=short

# Fast tests only
pytest -m "not integration"
```

### Test Coverage Requirements

- **New features**: Minimum 80% coverage
- **Bug fixes**: Add test case for bug
- **Existing code**: Maintain current coverage level

---

## Pull Request Process

### 1. Create Pull Request

```bash
# Push your branch
git push origin feature/my-feature

# Create PR on GitHub (or use CLI)
# gh pr create --title "Add X feature" --body "Fixes #123"
```

### 2. PR Title & Description

```markdown
# Title
[Type] Brief description (max 50 chars)

Examples:
- feature: Add facial liveness challenge-response
- fix: Correct false positive rate calculation
- docs: Update deployment guide for Kubernetes
- test: Add edge case tests for enrollment

# Description
## What does this PR do?
Explain the feature/fix in detail.

## Why?
Motivation, context, or issue it fixes.

## Related Issues
Fixes #123
Related to #456

## Testing
How did you test this?
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Manual testing performed

## Checklist
- [ ] Code follows PEP 8 style
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No breaking changes
```

### 3. Code Review

Reviewers will check:
- **Functionality**: Does it work as intended?
- **Testing**: Sufficient test coverage?
- **Documentation**: Is it clear?
- **Performance**: Any regressions?
- **Security**: Any vulnerabilities?

### 4. Address Feedback

```bash
# Make requested changes
vim vision/pipeline.py

# Commit changes
git commit -m "Address review feedback: add performance check"

# Push (no force unless requested)
git push origin feature/my-feature
```

### 5. Merge

Once approved:
- PR will be squashed and merged to `main`
- Delete your branch (GitHub will suggest)

```bash
# Clean up locally
git checkout main
git pull upstream main
git branch -d feature/my-feature
```

---

## Reporting Issues

### Before Opening an Issue

```bash
# 1. Check existing issues

# 2. Try latest version
git pull upstream main
python scripts/verify_versions.py

# 3. Check documentation
grep -r "your-search-term" docs/

# 4. Try troubleshooting guide
# See TROUBLESHOOTING.md
```

### Issue Templates

**Bug Report**:
```markdown
# Environment
- OS: Ubuntu 20.04
- Python: 3.12
- Branch: main (as of commit abc123)

# Steps to Reproduce
1. ...
2. ...
3. ...

# Expected
...

# Actual
...

# Logs
```

**Feature Request**:
```markdown
# Description
What problem does this solve?

# Proposed Solution
How should this work?

# Alternatives Considered
Other approaches?

# Additional Context
Any other information?
```

---

## Resources

### Documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [PIPELINE.md](PIPELINE.md) - Face recognition pipeline
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production setup

### External
- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [GitHub Collaborative Development Model](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models)

---

## Code of Conduct

We are committed to providing a welcoming environment for all contributors. Please:

- Be respectful and inclusive
- No harassment or discrimination
- Constructive feedback only
- Questions are always welcome

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0

