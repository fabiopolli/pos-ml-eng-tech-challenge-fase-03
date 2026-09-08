"""Synthetic traffic generator for the Fase 2 observability stack.

Sends a controlled burst of ``/predict`` calls to the sklearn and ONNX
instances of the official API so the dashboard shows side-by-side
statistics without leaking any clinical text. The payloads are static,
non-clinical, and explicitly designed to never be stored server-side
(the official API rejects persistence by design).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from ipaddress import ip_address
from urllib.parse import urlparse

DEFAULT_SKL_URL = "http://127.0.0.1:8001"
DEFAULT_ONX_URL = "http://127.0.0.1:8002"


def _build_samples() -> list[str]:
    """Generate benign, varied samples so the route label distribution is healthy."""

    topics = ["cardiology", "oncology", "neurology", "respiratory", "general"]
    return [
        f"sample abstract number {i:04d} about {random.choice(topics)} case study"
        for i in range(32)
    ]


def _enforce_loopback(url: str, *, role: str) -> None:
    """Reject URLs pointing outside ``127.0.0.1``/``localhost`` to avoid SSRF drift."""

    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"{role} URL is missing a host: {url!r}")
    try:
        if ip_address(host).is_loopback:
            return
    except ValueError:
        if host.lower() in {"localhost", "127.0.0.1", "::1"}:
            return
        # Resolved URL to public IP: deny
        if host.lower() == "host.docker.internal":
            return
    if os.environ.get("TRIAGE_ALLOW_PUBLIC_OBSERVABILITY", "false").lower() != "true":
        raise ValueError(
            f"{role} URL must point to a loopback host; got {host!r}. "
            "Set TRIAGE_ALLOW_PUBLIC_OBSERVABILITY=true to allow (not recommended)."
        )


def _post_json(url: str, payload: dict[str, str], *, api_key: str | None = None) -> tuple[int, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    body = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(body, timeout=10) as response:  # noqa: S310 - URL is operator-supplied loopback
            return response.status, response.read(1024).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:  # noqa: PERF203 - operator-facing report
        return exc.code, exc.read(1024).decode("utf-8", errors="replace") if hasattr(
            exc, "read"
        ) else ""


def _scrape_metrics(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"unsupported metrics URL: {url!r}")
    with urllib.request.urlopen(url + "/metrics", timeout=5) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="replace")


def _resolve_api_key(args: argparse.Namespace) -> str:
    """Resolve the doctor key required by the production ``/predict`` RBAC."""

    for variable in (
        "TRIAGE_ML_API_KEY_DOCTOR",
        "TRIAGE_ML_TRAFFIC_API_KEY",
    ):
        value = os.environ.get(variable)
        if value:
            return value
    return args.api_key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic traffic for observability dashboards."
    )
    parser.add_argument("--sklearn-url", default=os.environ.get("API_SKL_URL", DEFAULT_SKL_URL))
    parser.add_argument("--onnx-url", default=os.environ.get("API_ONX_URL", DEFAULT_ONX_URL))
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TRIAGE_ML_TRAFFIC_API_KEY"),
        help="Doctor API key; TRIAGE_ML_API_KEY_DOCTOR/TRAFFIC_API_KEY env vars also accepted.",
    )
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Complete rounds over the 32 synthetic samples (ignored when --requests is set).",
    )
    args = parser.parse_args(argv)

    args.api_key = _resolve_api_key(args) or ""
    if not args.api_key:
        print("error: a doctor --api-key or TRIAGE_ML_API_KEY_DOCTOR is required", file=sys.stderr)
        return 2
    if args.iterations <= 0 or (args.requests is not None and args.requests <= 0):
        print("error: --iterations and --requests must be positive", file=sys.stderr)
        return 2

    _enforce_loopback(args.sklearn_url, role="sklearn")
    _enforce_loopback(args.onnx_url, role="onnx")

    samples = _build_samples()
    request_count = args.requests if args.requests is not None else args.iterations * len(samples)
    failure_count = 0
    for index in range(request_count):
        for target in (args.sklearn_url, args.onnx_url):
            text = samples[index % len(samples)]
            try:
                status, _ = _post_json(
                    f"{target}/predict",
                    payload={"text": text},
                    api_key=args.api_key,
                )
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                print(f"warn: {target}/predict failed: {exc}", file=sys.stderr)
                failure_count += 1
                continue
            if status >= 400:
                failure_count += 1
        time.sleep(0.01)

    # Final scrape to confirm metrics were emitted by both variants.
    for label, target in (("sklearn", args.sklearn_url), ("onnx", args.onnx_url)):
        try:
            payload = _scrape_metrics(target)
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            print(f"warn: scrape {label} failed: {exc}", file=sys.stderr)
            failure_count += 1
            continue
        observations = sum(
            float(line.rsplit(" ", 1)[1])
            for line in payload.splitlines()
            if line.startswith("triage_ml_request_latency_seconds_count")
            and 'route="/predict"' in line
        )
        if observations < request_count:
            print(
                f"warn: expected at least {request_count} prediction observations on {label}, "
                f"found {observations:g}",
                file=sys.stderr,
            )
            failure_count += 1
        print(f"observed {observations:g} prediction requests on {label} ({target})")

    return failure_count


if __name__ == "__main__":
    raise SystemExit(main())
