"""Basic smoke test for AutoAttendance deployments.

Usage:
    python scripts/smoke_test.py --base-url http://localhost:5000 --check-video
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _fetch(url: str, timeout: float = 5.0) -> tuple[int, bytes, dict]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        code = resp.getcode()
        body = resp.read()
        headers = dict(resp.headers.items())
    return code, body, headers


def _check_json_endpoint(base_url: str, path: str, expected: int = 200) -> bool:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        code, body, _ = _fetch(url)
        if code != expected:
            print(f"[FAIL] {path}: expected {expected}, got {code}")
            return False
        json.loads(body.decode("utf-8"))
        print(f"[ OK ] {path}")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {path}: {exc}")
        return False


def _check_video(base_url: str) -> bool:
    url = f"{base_url.rstrip('/')}/video_feed"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "multipart/x-mixed-replace" not in content_type:
                print(f"[FAIL] /video_feed: unexpected content type: {content_type}")
                return False
            chunk = resp.read(2048)
            if b"--frame" not in chunk:
                print("[FAIL] /video_feed: no MJPEG frame boundary detected")
                return False
            print("[ OK ] /video_feed")
            return True
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"[FAIL] /video_feed: {exc}")
        return False


def _check_stream_lifecycle(base_url: str) -> bool:
    """Open then close video stream, then verify diagnostics report no active viewers."""
    stream_url = f"{base_url.rstrip('/')}/video_feed"
    req = urllib.request.Request(stream_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            _ = resp.read(1024)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"[FAIL] stream lifecycle open/close: {exc}")
        return False

    diag_url = f"{base_url.rstrip('/')}/api/debug/diagnostics"
    try:
        code, body, _ = _fetch(diag_url)
        if code != 200:
            print(f"[WARN] diagnostics unavailable (status {code}); skipping lifecycle assertion")
            return True
        payload = json.loads(body.decode("utf-8"))
        viewers = payload.get("cameras", {}).get("viewers", {})
        active = sum(int(v) for v in viewers.values()) if viewers else 0
        if active != 0:
            print(f"[FAIL] stream lifecycle: expected 0 viewers after close, got {active}")
            return False
        print("[ OK ] stream lifecycle")
        return True
    except urllib.error.HTTPError as exc:
        # Diagnostics endpoint may be disabled when DEBUG_MODE=0.
        if exc.code == 404:
            print("[WARN] diagnostics endpoint disabled; lifecycle assertion skipped")
            return True
        print(f"[FAIL] diagnostics check: {exc}")
        return False
    except (urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] diagnostics check: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoAttendance smoke test")
    parser.add_argument("--base-url", default="http://localhost:5000")
    parser.add_argument("--check-video", action="store_true")
    args = parser.parse_args()

    checks = [
        _check_json_endpoint(args.base_url, "/health"),
        _check_json_endpoint(args.base_url, "/healthz"),
        _check_json_endpoint(args.base_url, "/api/metrics"),
        _check_json_endpoint(args.base_url, "/api/registration_numbers"),
    ]

    if args.check_video:
        checks.append(_check_video(args.base_url))
        checks.append(_check_stream_lifecycle(args.base_url))

    ok = all(checks)
    print("\nSmoke test:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
