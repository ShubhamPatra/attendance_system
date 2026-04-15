"""Verify key runtime imports and print their versions."""

from importlib import import_module


MODULES = [
    "flask",
    "cv2",
    "insightface",
    "onnxruntime",
    "torch",
    "torchvision",
    "pymongo",
    "celery",
    "redis",
    "scipy",
    "PIL",
    "eventlet",
    "bcrypt",
]


def module_version(mod):
    return getattr(mod, "__version__", "unknown")


def main() -> None:
    print("Dependency import check")
    print("=" * 40)
    for name in MODULES:
        module = import_module(name)
        print(f"{name:<15} OK  v{module_version(module)}")
    print("=" * 40)
    print("All imports OK")


if __name__ == "__main__":
    main()
