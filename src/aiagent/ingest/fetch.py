"""Fetch a URL through the configured forward proxy (pipelock).

``httpx`` is imported lazily inside the function, matching the ``doctor``/
``models`` probe convention so importing this module stays cheap. TLS verifies
against the system trust store (which the devai lab image seeds with the
pipelock CA via ``SSL_CERT_FILE``), so no ``verify=False`` is ever needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from aiagent.config import Settings
from aiagent.exceptions import SourceError

_ALLOWED_SCHEMES = frozenset({"http", "https"})


@dataclass(frozen=True)
class FetchResult:
    """A fetched payload with its normalized (parameters-stripped) content type."""

    content: bytes
    content_type: str


def fetch_url(url: str, *, settings: Settings) -> FetchResult:
    """GET ``url`` via the configured proxy; raise :class:`SourceError` on failure."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise SourceError(
            f"unsupported URL scheme {scheme or '(none)'!r} (http/https only)"
        )

    import httpx  # local import keeps module load light

    proxy = settings.proxy_url or None
    try:
        with httpx.Client(
            timeout=settings.request_timeout_s,
            proxy=proxy,
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            raw_type = resp.headers.get("content-type", "")
            content_type = raw_type.split(";")[0].strip().lower()
            return FetchResult(content=resp.content, content_type=content_type)
    except httpx.HTTPError as exc:
        raise SourceError(f"failed to fetch {url}: {exc}") from exc
