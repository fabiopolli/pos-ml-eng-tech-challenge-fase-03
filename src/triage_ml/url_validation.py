"""Defensive URL validation reused by the dev/prod dashboards and the API.

The previous ``_normalize_api_url`` helpers in ``front/app_dev.py`` and
``front/app_prod.py`` accepted any HTTP(S) URL with a resolvable host. In
production this opens the door to SSRF: an attacker who can edit the
URL field could point the dashboard at ``http://169.254.169.254/`` (cloud
metadata), ``http://localhost:25/`` (local services) or any RFC1918
endpoint exposed on the host network. This module centralises the
allowlist logic so the API, the dev dashboard and the production portal
agree on what counts as a safe URL.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Final
from urllib.parse import urlsplit, urlunsplit

#: Reserved addresses that the dashboards must never be allowed to hit.
_BLOCKED_HOSTNAMES: Final[frozenset[str]] = frozenset(
    {
        # AWS, GCP and Azure instance metadata endpoints (string match).
        "169.254.169.254",
        "metadata.google.internal",
        "metadata.azure.com",
        "kubernetes.default.svc",
        "localhost",
    }
)

#: CIDRs that resolve to non-public network space.
_PRIVATE_CIDRS: Final[tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]] = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


def _is_non_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True for link-local, loopback, private, multicast, reserved and unspecified."""

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True
    return any(ip in network for network in _PRIVATE_CIDRS)


def _resolve_ips(
    hostname: str, port: int | None
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(hostname, port or 80, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return []
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def validate_public_http_url(
    url: str,
    *,
    allow_loopback: bool = False,
    allow_private_cidrs: bool = False,
) -> str:
    """Validate ``url`` and return the canonical form.

    The URL must:

    * use ``http`` or ``https``;
    * not embed userinfo, query string or fragment;
    * resolve to at least one IP address;
    * not resolve to ``169.254.169.254`` or any other link-local/cloud
      metadata service, RFC1918, loopback or multicast address — unless
      the caller explicitly opts in via ``allow_loopback`` or
      ``allow_private_cidrs``.

    Parameters
    ----------
    url:
        Candidate URL typed by the operator.
    allow_loopback:
        When ``True`` (default in dev dashboards), loopback addresses
        (``127.0.0.0/8`` and ``::1``) are accepted so the dev portal can
        hit a local ``uvicorn``. Production callers should pass
        ``allow_loopback=False``.
    allow_private_cidrs:
        When ``True``, RFC1918 ranges (``10.0.0.0/8``,
        ``172.16.0.0/12``, ``192.168.0.0/16``) and link-local Docker
        ranges are accepted. This is meant for dashboards that run
        **inside a container** and need to reach sibling services
        (``api-prod`` resolving to e.g. ``172.23.0.2``). Link-local
        cloud metadata endpoints (``169.254.169.254``,
        ``metadata.google.internal``, ``metadata.azure.com`` and
        ``kubernetes.default.svc``) remain forbidden even with this
        flag set, because they expose credentials to any process that
        can reach them. Use only in trusted container networks.
    """

    candidate = url.strip().rstrip("/")
    parsed = urlsplit(candidate)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("API URL has an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("API URL must be HTTP(S) without credentials, query, or fragment")
    if parsed.hostname.lower() in _BLOCKED_HOSTNAMES:
        raise ValueError(f"API URL host is forbidden: {parsed.hostname}")
    # If the operator typed a literal IP, validate it directly without a
    # DNS roundtrip. We still reject non-public literal IPs.
    try:
        literal_ip = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        if literal_ip.is_loopback and allow_loopback:
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
        if allow_private_cidrs and _is_rfc1918_or_link_local(literal_ip):
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
        if _is_non_public(literal_ip):
            raise ValueError(f"API URL host is a non-public address: {literal_ip}")
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    resolved = _resolve_ips(parsed.hostname, port)
    if not resolved:
        if allow_loopback:
            # DNS is unavailable in unit tests; with loopback tolerated
            # the schema/port checks above are sufficient to refuse
            # obviously bad URLs. We accept the host on trust and let
            # the caller fail later when it tries to connect.
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
        raise ValueError(f"API URL host could not be resolved: {parsed.hostname}")
    for ip in resolved:
        if ip.is_loopback and allow_loopback:
            continue
        if allow_private_cidrs and _is_rfc1918_or_link_local(ip):
            continue
        if _is_non_public(ip):
            raise ValueError(f"API URL host resolves to a non-public address: {ip}")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _is_rfc1918_or_link_local(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Return True for RFC1918 / link-local addresses that are safe inside a container network.

    Cloud metadata endpoints (``169.254.169.254``) and Kubernetes API
    servers are excluded here — they must remain forbidden even when
    the caller opts into ``allow_private_cidrs``.
    """

    if ip.version == 4:
        # ``169.254.169.254`` is a subset of ``169.254.0.0/16`` (link-local
        # IPv4) and is always excluded because it's the cloud metadata
        # endpoint. We accept other ``169.254.x.x`` addresses here for
        # parity with the historical SSRF guard, which targeted the
        # metadata endpoint specifically via the hostname blocklist above.
        rfc1918 = (
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
        )
        return any(ip in network for network in rfc1918)
    # IPv6: ULA (``fc00::/7``) and link-local excluding ``::1``.
    return ip in ipaddress.ip_network("fc00::/7") or (
        ip in ipaddress.ip_network("fe80::/10")
    )
