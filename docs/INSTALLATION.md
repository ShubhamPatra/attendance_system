# Installation Guide

**AutoAttendance** supports Python 3.9+ across all major operating systems (Windows, macOS, Linux) and deployment methods (local development, Docker, Kubernetes).

## Quick Start

### Local Development (Python 3.9+)

**Prerequisites:**
- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- pip or conda
- Git

**Installation:**

```bash
# Clone repository
git clone https://github.com/ShubhamPatra/attendance_system.git
cd attendance_system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -e .

# Verify installation
python scripts/verify_versions.py

# Run locally
python run.py
```

### Docker (CPU)

```bash
# Build and run with default settings (Python 3.12, CPU)
docker compose up --build

# Run with Python 3.11 instead
PYTHON_VERSION=3.11 docker compose up --build

# Run with Python 3.10 (for older systems)
PYTHON_VERSION=3.10 docker compose up --build
```

### Docker (GPU - NVIDIA)

```bash
# Build with GPU support (requires NVIDIA Docker)
INSTALL_GPU=1 docker compose up --build

# Build with specific Python version and GPU
PYTHON_VERSION=3.11 INSTALL_GPU=1 docker compose up --build
```

---

## Detailed Setup by Python Version

### Python 3.9 (Broadest Compatibility)

**macOS:**
```bash
# Using Homebrew
brew install python@3.9
python3.9 -m venv venv
source venv/bin/activate
pip install -e .
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3.9 python3.9-venv python3.9-dev
python3.9 -m venv venv
source venv/bin/activate
pip install -e .
```

**Windows (PowerShell):**
```powershell
# Download from https://www.python.org/downloads/
python.exe -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .
```

### Python 3.10 (Recommended for Development)

**macOS:**
```bash
brew install python@3.10
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3.10 python3.10-venv python3.10-dev
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

**Windows (PowerShell):**
```powershell
python310.exe -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .
```

### Python 3.11 (Production Default)

**macOS:**
```bash
brew install python@3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3.11 python3.11-venv python3.11-dev
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

**Windows (PowerShell):**
```powershell
python311.exe -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .
```

### Python 3.12+ (Latest Features)

**macOS:**
```bash
brew install python@3.12
python3.12 -m venv venv
source venv/bin/activate
pip install -e .
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3.12 python3.12-venv python3.12-dev
python3.12 -m venv venv
source venv/bin/activate
pip install -e .
```

**Windows (PowerShell):**
```powershell
python312.exe -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .
```

---

## Installation Options

### Option 1: Core + CPU Runtime (Default)

```bash
# Local
pip install -e .

# Docker
docker compose up --build
```

Includes CPU-optimized ONNX Runtime for inference.

### Option 2: Core + Development Tools

```bash
# Local
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
black .
flake8 .
mypy .
```

### Option 3: Core + GPU Runtime

**Local (requires NVIDIA GPU + CUDA drivers):**

```bash
# Uninstall CPU variant
pip uninstall -y onnxruntime

# Install GPU variant
pip install -e ".[gpu]"

# Or directly:
pip install onnxruntime-gpu>=1.17,<2.0 torch>=2.0,<3.0 torchvision>=0.15,<1.0 --index-url https://download.pytorch.org/whl/cu121
```

**Docker (requires NVIDIA Docker Runtime):**

```bash
INSTALL_GPU=1 docker compose up --build
```

### Option 4: Modular Installation

```bash
# Install specific requirement files
pip install -r requirements/base.txt
pip install -r requirements/cpu.txt    # or gpu.txt
pip install -r requirements/dev.txt    # optional
```

---

## Verification

After installation, verify everything works:

```bash
# Automated verification
python scripts/verify_versions.py

# Expected output:
# ✓ Python version: 3.11.x
# ✓ All required packages installed
# ✓ PyTorch and torchvision available
# ✓ MongoDB connectivity check (if MONGO_URI set)
# ✓ All checks passed!
```

### Manual Verification

```bash
# Python version
python --version                    # Should be 3.9+

# Core imports
python -c "import flask; print('✓ Flask OK')"
python -c "import torch; print('✓ PyTorch OK')"
python -c "import pymongo; print('✓ PyMongo OK')"
python -c "import cv2; print('✓ OpenCV OK')"
python -c "import onnxruntime; print('✓ ONNX Runtime OK')"

# Run tests
pytest tests/ -v --tb=short
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'X'`

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade --force-reinstall -e .

# Or verify versions
python scripts/verify_versions.py
```

### Issue: Python Version Mismatch

**Verify current Python:**
```bash
python --version
which python        # macOS/Linux
where python        # Windows
```

**Ensure virtual environment is activated:**
```bash
# macOS/Linux
source venv/bin/activate

# Windows
.\venv\Scripts\Activate.ps1
```

### Issue: PyTorch Installation Fails

**Workaround for CPU:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Workaround for GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Docker Build Fails

**Try with Python 3.11 instead of default 3.12:**
```bash
PYTHON_VERSION=3.11 docker compose up --build
```

**Check Docker resources:**
```bash
docker system prune      # Free up space
docker compose logs -f   # Check error logs
```

### Issue: `torch` and `numpy` Compatibility

**The project pins `numpy>=1.24,<2.0` for insightface compatibility.**

If you see numpy version conflicts:
```bash
pip install 'numpy>=1.24,<2.0'
pip install 'torch>=2.0,<3.0'
```

### Issue: CUDA Driver / GPU Not Detected

**Verify GPU availability:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`:
1. Check NVIDIA drivers: `nvidia-smi`
2. Ensure NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi`
3. Use CPU variant instead: `INSTALL_GPU=0 docker compose up`

---

## System Requirements

### Minimum

- **Python:** 3.9+
- **RAM:** 4 GB (8 GB recommended)
- **Disk:** 2 GB for models + dependencies
- **CPU:** Dual-core 2 GHz+
- **OS:** Windows 7+, macOS 10.13+, Ubuntu 18.04+

### Recommended

- **Python:** 3.11 or 3.12
- **RAM:** 8 GB or more
- **GPU:** NVIDIA (CUDA 11.8+, CUDA 12.x)
- **Disk:** 4 GB or more
- **OS:** Windows 10+, macOS 12+, Ubuntu 20.04+

### For GPU Support

- **GPU:** NVIDIA (GTX 1060 or better)
- **VRAM:** 4 GB minimum, 8 GB recommended
- **NVIDIA Drivers:** Latest stable version
- **CUDA Toolkit:** 11.8+ or 12.x
- **cuDNN:** 8.x

---

## Environment Variables

Key environment variables for configuration:

```bash
# Database
MONGO_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/attendance_system

# Application
SECRET_KEY=your-secret-key-here
APP_HOST=0.0.0.0
APP_PORT=5000
APP_DEBUG=0

# Features
EMBEDDING_BACKEND=arcface
ENABLE_RBAC=0
ENABLE_RESTX_API=0

# Camera
CAMERA_INDICES=0
STARTUP_CAMERA_PROBE=0

# Recognition Thresholds
RECOGNITION_THRESHOLD=0.50
LIVENESS_CONFIDENCE_THRESHOLD=0.55

# See .env.example for all options
```

---

## Next Steps

1. ✅ Verify installation: `python scripts/verify_versions.py`
2. 📖 Review [README.md](../README.md) for usage
3. 🔧 Configure [.env](./.env.example) file
4. 🚀 Run: `python run.py` (local) or `docker compose up` (Docker)
5. 🧪 Test: `pytest tests/ -v`

---

## Support

- 📝 [GitHub Issues](https://github.com/ShubhamPatra/attendance_system/issues)
- 💬 [Discussions](https://github.com/ShubhamPatra/attendance_system/discussions)
- 📧 Contact: contact@shubhampatra.com
