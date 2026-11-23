"""Request guard: runs before any request body is read.

A face-recognition API on ``localhost`` is reachable by every web page the
operator has open, because browsers will happily send a cross-site multipart
``POST`` to ``http://127.0.0.1:8000`` without a preflight. This pure-ASGI
middleware closes that door and a few adjacent ones, and it does so before the
body is consumed, so nothing an unauthorised caller sends is ever buffered:

* **Host allow-list** (DNS-rebinding defence): the ``Host`` header must name
  this server.
* **Same-origin writes**: a state-changing request carrying an ``Origin`` (or a
  ``Sec-Fetch-Site`` of ``cross-site`` / ``same-site``) from anywhere else is
  refused. Non-browser clients send neither header and are unaffected.
* **API key** (when configured) for every ``/api/`` route except the few a room
  guest needs.
* **Body size**: declared ``Content-Length`` must be present and within bounds.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import re
from urllib.parse import urlsplit

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
# host[:port] or [ipv6][:port]; nothing else (no userinfo, lists or spaces).
_HOST_PATTERN = re.compile(r"^(?:[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?|\[[0-9A-Fa-f:.]+\])(?::\d{1,5})?$")
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def keys_equal(supplied: str | None, expected: str) -> bool:
    """Constant-time comparison that tolerates non-ASCII input."""
    return hmac.compare_digest((supplied or "").encode("utf-8"), expected.encode("utf-8"))


def host_without_port(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("["):  # [::1]:8000
        return value[1:].split("]", 1)[0]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class RequestGuard:
    def __init__(
        self,
        app,
        *,
        api_key: str | None,
        allowed_hosts: set[str] | None,
        max_body_bytes: int,
        open_paths: tuple[str, ...] = (),
        open_prefixes: tuple[str, ...] = (),
        trusted_origins: tuple[str, ...] = (),
    ) -> None:
        self.app = app
        self.api_key = api_key
        self.allowed_hosts = allowed_hosts  # None disables the check ("*")
        self.max_body_bytes = max_body_bytes
        self.open_paths = set(open_paths)
        self.open_prefixes = open_prefixes
        # Origins accepted in addition to "same as Host": the public address
        # when a TLS proxy in front does not forward the original Host header.
        self.trusted_origins = {o.rstrip("/").lower() for o in trusted_origins if o}

    @staticmethod
    async def _reject(scope, receive, send, status: int, detail: str) -> None:
        if scope["type"] == "websocket":
            # Closing before accept makes the handshake fail with HTTP 403.
            await send({"type": "websocket.close", "code": 4403})
            return
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"cache-control", b"no-store"),
                (b"connection", b"close"),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    def _host_allowed(self, host_header: str) -> bool:
        if not _HOST_PATTERN.match(host_header):
            return False
        if self.allowed_hosts is None:
            return True
        return host_without_port(host_header) in self.allowed_hosts

    def _is_open(self, method: str, path: str) -> bool:
        if path in self.open_paths:
            return True
        return method == "GET" and any(path.startswith(prefix) for prefix in self.open_prefixes)

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        host = headers.get("host", "")
        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()

        if not self._host_allowed(host):
            return await self._reject(
                scope, receive, send, 421,
                "this server does not answer for that Host header; set FAIRFACE_ALLOWED_HOSTS "
                "(comma-separated names, or *) if you reach it through a proxy or custom domain",
            )

        origin = headers.get("origin")
        cross_site = False
        if origin and origin != "null":
            cross_site = (
                urlsplit(origin).netloc.lower() != host.lower()
                and origin.rstrip("/").lower() not in self.trusted_origins
            )
        elif origin == "null":
            cross_site = True
        elif headers.get("sec-fetch-site") in ("cross-site", "same-site"):
            cross_site = True
        if cross_site and (scope["type"] == "websocket" or method in UNSAFE_METHODS):
            return await self._reject(scope, receive, send, 403, "cross-origin requests are not accepted")

        if scope["type"] == "websocket":
            return await self.app(scope, receive, send)

        if method in UNSAFE_METHODS:
            declared = headers.get("content-length")
            if declared is None:
                if headers.get("transfer-encoding"):
                    return await self._reject(scope, receive, send, 411, "a Content-Length header is required")
            elif not declared.isdigit() or int(declared) > self.max_body_bytes:
                return await self._reject(scope, receive, send, 413, "request body too large")

        if self.api_key and path.startswith("/api/") and not self._is_open(method, path):
            if not keys_equal(headers.get("x-api-key"), self.api_key):
                return await self._reject(scope, receive, send, 401, "missing or invalid X-API-Key")

        return await self.app(scope, receive, send)
