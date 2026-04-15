#!/usr/bin/env python
"""
AutoAttendance - Version Verification Script
Validates Python version, dependencies, and system compatibility.
Run after installation: python scripts/verify_versions.py
"""

import sys
import importlib
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Color codes for terminal output
class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

def check_python_version():
    """Verify Python version is 3.9+"""
    version = sys.version_info
    print(f"\n{Color.BLUE}Python Version Check{Color.RESET}")
    print(f"  Current: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 9:
        print(f"  {Color.GREEN}✓ Python {version.major}.{version.minor}+ supported{Color.RESET}")
        return True
    else:
        print(f"  {Color.RED}✗ Requires Python 3.9+, found {version.major}.{version.minor}{Color.RESET}")
        return False

def check_module(module_name, min_version=None, max_version=None):
    """Check if a module is installed and optionally validate version."""
    try:
        module = importlib.import_module(module_name.split(".")[0])
        version = getattr(module, "__version__", None)
        
        status = f"{Color.GREEN}✓{Color.RESET}"
        msg = f"  {status} {module_name}"
        
        if version:
            msg += f" {version}"
            
            if min_version and not compare_versions(version, min_version, ">="):
                msg = f"  {Color.RED}✗{Color.RESET} {module_name} {version} (requires >={min_version})"
                return False, msg
            
            if max_version and not compare_versions(version, max_version, "<"):
                msg = f"  {Color.YELLOW}⚠{Color.RESET} {module_name} {version} (recommended <{max_version})"
                return True, msg
        
        return True, msg
    except ImportError:
        return False, f"  {Color.RED}✗{Color.RESET} {module_name} (not installed)"

def compare_versions(version_str, requirement_str, operator):
    """Simple version comparison."""
    try:
        from packaging import version
        v = version.parse(version_str)
        r = version.parse(requirement_str)
        
        if operator == ">=":
            return v >= r
        elif operator == "<":
            return v < r
        elif operator == "==":
            return v == r
    except (ImportError, Exception):
        # Fallback simple comparison
        pass
    return True

def check_dependencies():
    """Check all required dependencies."""
    print(f"\n{Color.BLUE}Core Dependencies{Color.RESET}")
    
    core_deps = [
        ("flask", None, None),
        ("flask_restx", None, None),
        ("flask_socketio", None, None),
        ("werkzeug", None, None),
        ("cv2", None, None),  # opencv
        ("numpy", None, None),
        ("PIL", None, None),  # Pillow
        ("pymongo", None, None),
        ("insightface", None, None),
        ("scipy", None, None),
    ]
    
    all_ok = True
    for module, min_ver, max_ver in core_deps:
        ok, msg = check_module(module, min_ver, max_ver)
        print(msg)
        if not ok:
            all_ok = False
    
    return all_ok

def check_ml_runtime():
    """Check PyTorch and ONNX Runtime."""
    print(f"\n{Color.BLUE}ML/AI Runtime{Color.RESET}")
    
    # PyTorch
    try:
        import torch
        import torchvision
        print(f"  {Color.GREEN}✓{Color.RESET} torch {torch.__version__}")
        print(f"  {Color.GREEN}✓{Color.RESET} torchvision {torchvision.__version__}")
        torch_ok = True
    except ImportError:
        print(f"  {Color.RED}✗{Color.RESET} torch (not installed)")
        torch_ok = False
    
    # ONNX Runtime
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        print(f"  {Color.GREEN}✓{Color.RESET} onnxruntime {onnxruntime.__version__}")
        
        # Check for GPU support
        if "TensorrtExecutionProvider" in providers or "CUDAExecutionProvider" in providers:
            print(f"    ├─ {Color.GREEN}GPU support detected{Color.RESET}")
        elif "CPUExecutionProvider" in providers:
            print(f"    └─ CPU mode (GPU not available)")
        
    except ImportError:
        print(f"  {Color.RED}✗{Color.RESET} onnxruntime (not installed)")
        torch_ok = False
    
    return torch_ok

def check_anti_spoofing():
    """Check anti-spoofing dependencies."""
    print(f"\n{Color.BLUE}Anti-Spoofing Runtime{Color.RESET}")
    
    try:
        import torch
        print(f"  {Color.GREEN}✓{Color.RESET} torch available for anti-spoofing models")
        return True
    except ImportError:
        print(f"  {Color.YELLOW}⚠{Color.RESET} torch required for anti-spoofing (optional)")
        return False

def check_dev_tools():
    """Check development tools if available."""
    print(f"\n{Color.BLUE}Development Tools (Optional){Color.RESET}")
    
    dev_tools = [
        "pytest",
        "black",
        "flake8",
        "mypy",
    ]
    
    for tool in dev_tools:
        ok, msg = check_module(tool)
        status = "installed" if ok else "not installed"
        symbol = f"{Color.GREEN}✓{Color.RESET}" if ok else f"{Color.YELLOW}○{Color.RESET}"
        print(f"  {symbol} {tool} ({status})")

def check_mongodb():
    """Check MongoDB connectivity if MONGO_URI is set."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    mongo_uri = os.environ.get("MONGO_URI")
    
    print(f"\n{Color.BLUE}Database Connectivity{Color.RESET}")
    
    if not mongo_uri:
        print(f"  {Color.YELLOW}○{Color.RESET} MONGO_URI not set (will be required at runtime)")
        return True
    
    try:
        from pymongo import MongoClient
        import socket
        
        # Quick connectivity test
        try:
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
            client.admin.command("ping")
            print(f"  {Color.GREEN}✓{Color.RESET} MongoDB connected")
            client.close()
            return True
        except Exception as e:
            print(f"  {Color.RED}✗{Color.RESET} MongoDB connection failed: {str(e)[:60]}")
            return False
    except ImportError:
        print(f"  {Color.RED}✗{Color.RESET} PyMongo not installed")
        return False

def check_system_info():
    """Display system information."""
    import platform
    import os
    
    print(f"\n{Color.BLUE}System Information{Color.RESET}")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  CPU Count: {os.cpu_count()}")
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  RAM: {mem.total / (1024**3):.1f} GB ({mem.percent}% used)")
    except ImportError:
        pass

def main():
    """Run all checks."""
    print(f"\n{Color.BLUE}{'='*50}")
    print(f"AutoAttendance - Version Verification")
    print(f"{'='*50}{Color.RESET}")
    
    checks = [
        ("Python Version", check_python_version),
        ("Core Dependencies", check_dependencies),
        ("ML/AI Runtime", check_ml_runtime),
        ("Anti-Spoofing", check_anti_spoofing),
        ("Database", check_mongodb),
        ("System Info", check_system_info),
        ("Dev Tools", check_dev_tools),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result if isinstance(result, bool) else True))
        except Exception as e:
            print(f"  {Color.YELLOW}⚠ {name} check failed: {str(e)[:40]}{Color.RESET}")
            results.append((name, False))
    
    # Summary
    print(f"\n{Color.BLUE}{'='*50}")
    print(f"Verification Summary{Color.RESET}")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        symbol = f"{Color.GREEN}✓{Color.RESET}" if result else f"{Color.RED}✗{Color.RESET}"
        print(f"  {symbol} {name}")
    
    print(f"\n{Color.BLUE}Result: {passed}/{total} checks passed{Color.RESET}\n")
    
    if passed == total:
        print(f"{Color.GREEN}✓ All checks passed! You're ready to go.{Color.RESET}\n")
        return 0
    else:
        print(f"{Color.YELLOW}⚠ Some checks failed. See above for details.{Color.RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
