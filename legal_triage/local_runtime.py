"""Minimal Ollama HTTP client constrained to loopback and with proxy inheritance disabled.

Module-level side-effect
------------------------
Importing this module installs a socket-level egress guard (``install_egress_guard``) that
prevents the Python process from opening TCP connections to any address outside the loopback
range.  The guard is idempotent; it will not be applied twice if the module is re-imported.

Note: this guard covers the Python process only.  The Ollama *server* process has its own
isolation requirements (``OLLAMA_NO_CLOUD=1``, ``OLLAMA_HOST=127.0.0.1:11434``) that must be
configured separately before launching the server.
"""
from __future__ import annotations

import ipaddress
import math
import socket
from typing import Any

import numpy as np
import requests

from .config import SETTINGS, ConfigurationError

# ---------------------------------------------------------------------------
# Runtime egress guard
# ---------------------------------------------------------------------------

_EGRESS_GUARD_INSTALLED: bool = False
_LOOPBACK_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),   # IPv4 loopback block
    ipaddress.ip_network("::1/128"),         # IPv6 loopback
)


def _is_loopback(host: str) -> bool:
    """Return True when *host* resolves to a loopback address.

    Hostnames ``localhost`` and ``127.*`` are accepted without DNS resolution;
    all other strings are parsed as IP addresses.  If parsing fails, the host
    is treated as non-loopback (fail-closed).
    """
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        addr = ipaddress.ip_address(host)
        return any(addr in net for net in _LOOPBACK_NETWORKS)
    except ValueError:
        # Could not parse as an IP – do not resolve; treat as non-loopback.
        return False


def install_egress_guard() -> None:
    """Monkey-patch ``socket.socket.connect`` to block non-loopback destinations.

    After this function returns, any attempt by the Python process to open a
    TCP (or UDP) connection to a non-loopback address will raise a
    ``RuntimeError``.

    The patch is idempotent: calling it a second time is a no-op.
    """
    global _EGRESS_GUARD_INSTALLED  # noqa: PLW0603
    if _EGRESS_GUARD_INSTALLED:
        return

    _original_connect = socket.socket.connect

    def _guarded_connect(self: socket.socket, address: object) -> None:  # type: ignore[override]
        """Intercept connect() and raise for non-loopback destinations."""
        host: str = ""
        if isinstance(address, (tuple, list)) and len(address) >= 1:
            # AF_INET: (host, port); AF_INET6: (host, port, flow, scope)
            host = str(address[0])
        elif isinstance(address, str):
            # AF_UNIX path or bare string – allow (Unix socket is local by definition).
            return _original_connect(self, address)
        if host and not _is_loopback(host):
            raise RuntimeError(
                f"Egress blocked: the Python process attempted to connect to "
                f"non-loopback address '{host}'. "
                "All network access must stay on 127.0.0.1/localhost (Python process). "
                "Ensure OLLAMA_HOST=127.0.0.1:11434 is set for the Ollama server process."
            )
        return _original_connect(self, address)

    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    _EGRESS_GUARD_INSTALLED = True


# Install the guard immediately when this module is imported so that the
# Streamlit app is protected even before a LocalOllama instance is created.
install_egress_guard()


class LocalModelError(RuntimeError):
    """The local model runtime is unreachable, missing models, or returned invalid data."""


class LocalOllama:
    def __init__(self) -> None:
        # Strict configuration validation happens before any request is created.
        self.base_url = SETTINGS.ollama_base_url
        self.chat_model = SETTINGS.chat_model
        self.embedding_model = SETTINGS.embedding_model
        self.session = requests.Session()
        # Do not inherit HTTP(S)_PROXY variables from the environment.
        self.session.trust_env = False
        self.session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

    def _url(self, endpoint: str) -> str:
        # Defensive invariant: a request can only target our validated loopback base URL.
        if not endpoint.startswith("/") or "://" in endpoint:
            raise ConfigurationError("Blocked an invalid Ollama API path.")
        return f"{self.base_url}{endpoint}"

    def _request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self._url(endpoint)
        try:
            if method == "GET":
                response = self.session.get(url, timeout=(4, 15))
            else:
                response = self.session.post(
                    url,
                    json=payload or {},
                    timeout=(4, SETTINGS.request_timeout_seconds),
                )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise LocalModelError(f"Ollama returned an unexpected response from {endpoint}.")
            return result
        except requests.RequestException as exc:
            raise LocalModelError(
                f"Cannot reach local Ollama at {self.base_url}. Start Ollama, pull the required models, "
                f"and retry. Details: {type(exc).__name__}: {exc}"
            ) from exc
        except ValueError as exc:
            raise LocalModelError(f"Ollama returned non-JSON output from {endpoint}.") from exc

    def installed_models(self) -> set[str]:
        result = self._request("GET", "/api/tags")
        models = result.get("models", [])
        names = {str(item.get("name", "")) for item in models if isinstance(item, dict)}
        normalized = {name for name in names if name}
        # Ollama reports pulled untagged names as `name:latest` from /api/tags.
        for name in tuple(normalized):
            if name.endswith(":latest"):
                normalized.add(name[:-7])
            elif ":" not in name:
                normalized.add(f"{name}:latest")
        return normalized

    def verify_models(self) -> dict[str, str]:
        available = self.installed_models()
        missing = [name for name in (self.chat_model, self.embedding_model) if name not in available]
        if missing:
            raise LocalModelError(
                "Required local model(s) are not installed: " + ", ".join(missing) + ".\n"
                "Download them during setup with `ollama pull MODEL`, then rerun the check. "
                "No hosted-model fallback is used."
            )
        return {"chat_model": self.chat_model, "embedding_model": self.embedding_model, "runtime": "Ollama local API"}

    def embed(self, texts: list[str], batch_size: int = 24) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            result = self._request(
                "POST",
                "/api/embed",
                {"model": self.embedding_model, "input": batch},
            )
            embeddings = result.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise LocalModelError("The local embedding model returned an unexpected number of vectors.")
            for vector in embeddings:
                if not isinstance(vector, list) or not vector:
                    raise LocalModelError("The local embedding model returned an empty vector.")
                floats = [float(value) for value in vector]
                if not all(math.isfinite(value) for value in floats):
                    raise LocalModelError("The local embedding model returned non-finite vector values.")
                vectors.append(floats)
        dimensions = {len(row) for row in vectors}
        if len(dimensions) != 1:
            raise LocalModelError("The local embedding model returned vectors with inconsistent dimensions.")
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms

    def chat_json(self, system_prompt: str, user_prompt: str) -> str:
        result = self._request(
            "POST",
            "/api/chat",
            {
                "model": self.chat_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_ctx": SETTINGS.num_ctx,
                    "num_predict": SETTINGS.num_predict,
                },
                "keep_alive": "5m",
            },
        )
        message = result.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        if not isinstance(content, str) or not content.strip():
            raise LocalModelError("The local Gemma model returned an empty response.")
        return content
