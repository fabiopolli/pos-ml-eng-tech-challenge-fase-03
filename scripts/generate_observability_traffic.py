"""Synthetic traffic generator for the Fase 2 observability stack.

Sends a controlled burst of ``/predict`` calls to the sklearn and ONNX
instances of the official API so the dashboard shows side-by-side
statistics without leaking any clinical text. The payloads are static,
non-clinical, and explicitly designed to never be stored server-side
(the official API rejects persistence by design).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import urllib.request

DEFAULT_SKL_URL = "http://127.0.0.1:8001"
DEFAULT_ONX_URL = "http://127.0.0.1:8002"


def _build_samples() -> list[str]:
    """Generate benign, varied samples so the route label distribution is healthy."""

    topics = ["cardiology", "oncology", "neurology", "respiratory", "general"]
    return [
        f"sample abstract number {i:04d} about {random.choice(topics)} case study"
        for i in range(32)
    ]


def _post_json(url: str, payload: dict[str, str], *, api_key: str | None = None) -> tuple[int, str]:
    body = urllib.request.Request(
        url,
        data=str(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **({"X-API-Key": api_key} if api_key else {})},
        method="POST",
    )
    with urllib.request.urlopen(body, timeout=10) as response:  # noqa: S310 - URLs are operator-supplied
        return response.status, response.read(1024).decode("utf-8", errors="replace")


def _scrape_metrics(url: str) -> str:
    with urllib.request.urlopen(url + "/metrics", timeout=5) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic traffic for observability dashboards."
    )
    parser.add_argument("--sklearn-url", default=os.environ.get("API_SKL_URL", DEFAULT_SKL_URL))
    parser.add_argument("--onnx-url", default=os.environ.get("API_ONX_URL", DEFAULT_ONX_URL))
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TRIAGE_ML_API_KEY_DOCTOR"),
        help="Doctor API key; falls back to env if omitted.",
    )
    parser.add_argument("--requests", type=int, default=20)
    args = parser.parse_args(argv)

    if not args.api_key:
        print("error: --api-key or TRIAGE_ML_API_KEY_DOCTOR is required", file=sys.stderr)
        return 2

    samples = _build_samples()
    failure_count = 0
    for target in (args.sklearn_url, args.onnx_url):
        for index in range(args.requests):
            text = samples[index % len(samples)]
            try:
                status, _ = _post_json(
                    f"{target}/predict",
                    payload={"text": text},
                    api_key=args.api_key,
                )
            except Exception as exc:  # noqa: BLE001 - operator-facing report
                print(f"warn: {target}/predict failed: {exc}", file=sys.stderr)
                failure_count += 1
                continue
            if status >= 400:
                failure_count += 1
        time.sleep(0.5)  # pause between variants

    # Final scrape to confirm metrics were emitted by both variants.
    for label, target in (("sklearn", args.sklearn_url), ("onnx", args.onnx_url)):
        try:
            payload = _scrape_metrics(target)
        except Exception as exc:  # noqa: BLE001 - operator-facing report
            print(f"warn: scrape {label} failed: {exc}", file=sys.stderr)
            continue
        samples_observed = sum(
            1
            for line in payload.splitlines()
            if line.startswith("triage_ml_request_latency_seconds_count")
        )
        print(f"observed {samples_observed} latency buckets on {label} ({target})")

    return failure_count


if __name__ == "__main__":
    raise SystemExit(main())
