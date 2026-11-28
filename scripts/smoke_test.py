"""Smoke-test a running serving container: fail unless every endpoint answers.

Waits until /health reports ok (the model takes a while to load), sends one test image
to /infer as base64 JSON (the contract of POST /invocations), and checks that /version
and /drift answer.

Usage:
    python scripts/smoke_test.py --base-url http://localhost:5001 \
        --images-dir data/raw/images/test
"""

import argparse
import base64
from pathlib import Path
import time

import requests

from vial_classifier.config import SERVING_PORT, TEST_DATA_DIR
from vial_classifier.modeling.predict import list_images


def wait_until_healthy(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "no answer"
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"/health: {response.text.strip()}")
                return
            last = f"HTTP {response.status_code} {response.text.strip()}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2)
    raise SystemExit(f"/health did not become ready within {timeout:.0f}s (last: {last})")


def check_infer(base_url: str, image: Path) -> None:
    payload = {"instances": [base64.b64encode(image.read_bytes()).decode()]}
    response = requests.post(f"{base_url}/infer", json=payload, timeout=60)
    if response.status_code != 200 or "predictions" not in response.text:
        raise SystemExit(f"/infer failed: HTTP {response.status_code} {response.text[:300]}")
    print(f"/infer ({image.name}): {response.text.strip()}")


def check_get(base_url: str, path: str) -> None:
    response = requests.get(f"{base_url}{path}", timeout=30)
    if response.status_code != 200:
        raise SystemExit(f"{path} failed: HTTP {response.status_code} {response.text[:300]}")
    print(f"{path}: {response.text.strip()[:300]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running serving container.")
    parser.add_argument("--base-url", default=f"http://localhost:{SERVING_PORT}")
    parser.add_argument("--images-dir", type=Path, default=TEST_DATA_DIR)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    wait_until_healthy(base_url, args.timeout)
    check_infer(base_url, args.images_dir / list_images(args.images_dir)[0])
    check_get(base_url, "/version")
    check_get(base_url, "/drift")
    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
