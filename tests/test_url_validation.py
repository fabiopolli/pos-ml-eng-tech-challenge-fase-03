"""Tests for the centralised SSRF guard in ``triage_ml.url_validation``."""

from __future__ import annotations

import pytest

from triage_ml.url_validation import validate_public_http_url


def test_accepts_public_literal_ip() -> None:
    """8.8.8.8 is a public Google DNS resolver and must be accepted."""

    assert validate_public_http_url("http://8.8.8.8:8000/", allow_loopback=False) == (
        "http://8.8.8.8:8000"
    )


def test_accepts_loopback_only_in_dev_mode() -> None:
    """127.0.0.1 is allowed when ``allow_loopback=True`` (dev dashboard)."""

    assert validate_public_http_url("http://127.0.0.1:8000/", allow_loopback=True) == (
        "http://127.0.0.1:8000"
    )


def test_rejects_loopback_in_production_mode() -> None:
    """The production dashboard must never accept a loopback URL."""

    with pytest.raises(ValueError, match="non-public address"):
        validate_public_http_url("http://127.0.0.1:8000", allow_loopback=False)


def test_rejects_link_local_169_254_169_254() -> None:
    """AWS/GCP/Azure metadata endpoint must be rejected by string."""

    with pytest.raises(ValueError, match="forbidden"):
        validate_public_http_url("http://169.254.169.254/latest/meta-data")


def test_rejects_rfc1918_10_dot() -> None:
    """RFC1918 10.0.0.0/8 is private and must be rejected."""

    with pytest.raises(ValueError, match="non-public address"):
        validate_public_http_url("http://10.0.0.1:8000", allow_loopback=False)


def test_rejects_ipv6_loopback_in_production() -> None:
    """``::1`` is the IPv6 loopback; production must reject it."""

    with pytest.raises(ValueError, match="non-public address"):
        validate_public_http_url("http://[::1]:8000", allow_loopback=False)


def test_rejects_credentials_in_url() -> None:
    """Userinfo in the URL is always rejected regardless of host."""

    with pytest.raises(ValueError, match="without credentials"):
        validate_public_http_url("http://user:pass@8.8.8.8:8000")


def test_rejects_query_and_fragment() -> None:
    """Query string and fragment are stripped/forbidden by the helper."""

    with pytest.raises(ValueError, match="without credentials"):
        validate_public_http_url("http://8.8.8.8:8000/?leak=1#frag")


def test_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="invalid port"):
        validate_public_http_url("http://8.8.8.8:notaport/")


def test_strips_trailing_slash() -> None:
    """The canonical URL must not keep a trailing slash."""

    assert validate_public_http_url("http://8.8.8.8:8000///", allow_loopback=False) == (
        "http://8.8.8.8:8000"
    )


def test_accepts_docker_rfc1918_with_flag() -> None:
    """Dashboards running inside a container must reach sibling services on Docker networks."""

    assert (
        validate_public_http_url("http://172.23.0.2:8000", allow_private_cidrs=True)
        == "http://172.23.0.2:8000"
    )
    assert (
        validate_public_http_url("http://10.0.0.5:8000", allow_private_cidrs=True)
        == "http://10.0.0.5:8000"
    )
    assert (
        validate_public_http_url("http://192.168.1.10:8000", allow_private_cidrs=True)
        == "http://192.168.1.10:8000"
    )


def test_rejects_rfc1918_without_flag() -> None:
    """Without ``allow_private_cidrs`` the historical SSRF guard still rejects RFC1918."""

    with pytest.raises(ValueError, match="non-public address"):
        validate_public_http_url("http://172.23.0.2:8000", allow_loopback=False)


def test_rejects_cloud_metadata_even_with_flag() -> None:
    """The cloud metadata endpoint stays forbidden even with ``allow_private_cidrs=True``."""

    with pytest.raises(ValueError, match="forbidden"):
        validate_public_http_url("http://169.254.169.254/latest", allow_private_cidrs=True)


def test_rejects_kubernetes_api_even_with_flag() -> None:
    """The Kubernetes API service is blocked by hostname blocklist."""

    with pytest.raises(ValueError, match="forbidden"):
        validate_public_http_url("http://kubernetes.default.svc", allow_private_cidrs=True)
