# Project Setup Guide

## ⚠️ Critical Dependencies

This project requires **ML models** that are committed to the repository. These are essential for the system to function.

### Models Included in Repository

1. **YuNet Face Detection** (`models/face_detection_yunet_2023mar.onnx`)
   - Required for real-time face detection
   - ~232 KB (included in git)
   - Loaded on application startup

2. **Anti-Spoofing Models** (`models/anti_spoofing/`)
   - Silent-Face-Anti-Spoofing models for liveness detection
   - Located in `Silent-Face-Anti-Spoofing/resources/anti_spoof_models/`
   - Required for security features

3. **ArcFace Embeddings** (auto-downloaded)
   - InsightFace models cached in `~/.insightface/models/`
   - Auto-downloaded on first run
   - Requires internet connection on first startup

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git (for version control)
- MongoDB Atlas account (or local MongoDB)

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/ShubhamPatra/attendance_system.git
cd attendance_system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -e .

# 4. Configure environment
cp .env.example .env  # If available, or create .env with required vars

# 5. Run the project
python run.py
```

## ⚙️ Environment Configuration

Required environment variables (create `.env` file):

```env
# MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority
DB_NAME=attendance_db

# Flask
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=<generated_secret_key>

# Optional: Deployment
DEPLOY_ENV=local
WORKERS=2
```

## ✅ Verification

Verify the installation was successful:

```bash
python scripts/verify_versions.py
```

Expected output:
- ✓ Python version
- ✓ OpenCV with DNN module
- ✓ YuNet model found
- ✓ Anti-spoofing models found
- ✓ MongoDB connection

## 🔧 Troubleshooting

### Missing YuNet Model Error

**Error:**
```
ERROR: YuNet model missing: models/face_detection_yunet_2023mar.onnx
```

**Solution:**
- The model should be included after cloning the repository
- If missing, it's already in git: `git pull` or `git status`
- If gitignore still blocks it: `git check-ignore -v models/face_detection_yunet_2023mar.onnx`

### Models Tracking in Git

The `.gitignore` has been updated to include critical ML models:

```gitignore
# Models are now tracked (essential dependencies)
# models/*.onnx  <- COMMENTED OUT (do not ignore)
```

**Why are models in git?**
- Face detection (YuNet) is critical and small (~232 KB)
- Prevents "missing model" errors on fresh clones
- Ensures reproducibility across environments
- Alternative: Use Git LFS (Large File Storage) for large models

### Database Connection Issues

Ensure MongoDB connection string is correct in `.env`:
```env
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
```

### Camera Not Detected

- Ensure camera is connected and not in use by another application
- Linux users may need: `sudo usermod -a -G video $USER`

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md) - Detailed OS-specific setup
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## 🤝 Contributing

When adding new models or dependencies:
1. Keep models under 1 MB or use Git LFS
2. Update `.gitignore` appropriately
3. Document in this SETUP.md
4. Add verification checks in `scripts/verify_versions.py`

---

**Last Updated:** April 17, 2026
**Project:** AutoAttendance - Intelligent Face Recognition Attendance System
