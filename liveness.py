"""
liveness.py – DEPRECATED

Blink-based liveness detection has been replaced by deep learning
anti-spoofing.  See ``anti_spoofing.py`` for the current implementation.

All EAR calculation, blink detection, and dlib landmark logic has been
removed.  Import ``check_liveness`` from ``anti_spoofing`` instead.
"""

from anti_spoofing import check_liveness, init_models  # noqa: F401 – re-export
