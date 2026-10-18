"""Configuration for the local-only PS2 legal triage application."""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """Raised when the application is configured in a way that breaks its privacy boundary."""


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return value


def _score(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number between -1 and 1.") from exc
    if not -1.0 <= value <= 1.0:
        raise ConfigurationError(f"{name} must be between -1 and 1.")
    return value


def validate_local_ollama_url(value: str) -> str:
    """Allow only cleartext HTTP to loopback; reject cloud endpoints and proxy-like hosts."""
    parsed = urlparse(value.strip())
    allowed_hosts = {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "http" or parsed.hostname not in allowed_hosts:
        raise ConfigurationError(
            "Privacy guard blocked OLLAMA_BASE_URL. It must be local loopback HTTP, "
            "for example http://127.0.0.1:11434. Remote hosts and HTTPS endpoints are not allowed."
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigurationError("OLLAMA_BASE_URL must not contain credentials, query parameters, or fragments.")
    if parsed.path not in {"", "/"}:
        raise ConfigurationError("OLLAMA_BASE_URL must be a root URL without a path.")
    port = parsed.port or 80
    if not 1 <= port <= 65535:
        raise ConfigurationError("OLLAMA_BASE_URL contains an invalid port.")
    return f"http://{parsed.netloc}".rstrip("/")


def validate_local_model_name(name: str, env_name: str = "OLLAMA_MODEL") -> str:
    clean = name.strip()
    lowered = clean.lower()
    if not clean:
        raise ConfigurationError(f"{env_name} cannot be empty.")
    if lowered.endswith(":cloud") or lowered.endswith("-cloud") or ":cloud-" in lowered:
        raise ConfigurationError(f"{env_name} points to an Ollama cloud model. Only local models are allowed.")
    return clean


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k_question: int
    top_k_triage: int
    min_retrieval_similarity: float
    max_file_mb: int
    max_files: int
    max_pages: int
    max_chunks: int
    request_timeout_seconds: int
    num_ctx: int
    num_predict: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ollama_base_url=validate_local_ollama_url(
                os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            ),
            chat_model=validate_local_model_name(
                os.getenv("OLLAMA_MODEL", "gemma4:e2b"), "OLLAMA_MODEL"
            ),
            embedding_model=validate_local_model_name(
                os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"), "OLLAMA_EMBED_MODEL"
            ),
            chunk_size=_positive_int("CHUNK_SIZE", 1100),
            chunk_overlap=_positive_int("CHUNK_OVERLAP", 180),
            top_k_question=_positive_int("TOP_K_QUESTION", 7),
            top_k_triage=_positive_int("TOP_K_TRIAGE", 12),
            min_retrieval_similarity=_score("MIN_RETRIEVAL_SIMILARITY", 0.25),
            max_file_mb=_positive_int("MAX_FILE_MB", 20),
            max_files=_positive_int("MAX_FILES", 10),
            max_pages=_positive_int("MAX_PAGES", 1500),
            max_chunks=_positive_int("MAX_CHUNKS", 12000),
            request_timeout_seconds=_positive_int("REQUEST_TIMEOUT_SECONDS", 300),
            num_ctx=_positive_int("OLLAMA_NUM_CTX", 8192),
            num_predict=_positive_int("OLLAMA_NUM_PREDICT", 1600),
        )


SETTINGS = Settings.from_env()
