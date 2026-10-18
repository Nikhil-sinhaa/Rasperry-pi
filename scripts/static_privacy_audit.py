#!/usr/bin/env python3
"""Lightweight static checks for the source tree's intended privacy boundary.

This is not a substitute for dependency review, network monitoring, or an offline run.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = [ROOT / "app.py", *sorted((ROOT / "legal_triage").glob("*.py"))]
FORBIDDEN_IMPORT_ROOTS = {"openai", "langchain", "google", "boto3", "azure", "telemetry"}
EXTERNAL_URL = re.compile(r"https?://(?!127\.0\.0\.1(?::\d+)?\b|localhost(?::\d+)?\b|\[?::1\]?(?::\d+)?\b)", re.I)


def main() -> int:
    errors: list[str] = []
    for path in CODE_FILES:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            errors.append(f"Syntax error in {path.relative_to(ROOT)}:{exc.lineno}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".", 1)[0] for alias in node.names]
                for root in roots:
                    if root in FORBIDDEN_IMPORT_ROOTS:
                        errors.append(f"Unexpected hosted-service/telemetry import `{root}` in {path.relative_to(ROOT)}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    errors.append(f"Unexpected hosted-service/telemetry import `{root}` in {path.relative_to(ROOT)}")
        # All explicit URL string literals in app/runtime code must be local loopback examples.
        for line_number, line in enumerate(source.splitlines(), start=1):
            if ("http://" in line or "https://" in line) and "http://{parsed.netloc}" not in line:
                if EXTERNAL_URL.search(line):
                    errors.append(f"Non-loopback URL literal in {path.relative_to(ROOT)}:{line_number}")

    config_source = (ROOT / "legal_triage" / "config.py").read_text(encoding="utf-8")
    runtime_source = (ROOT / "legal_triage" / "local_runtime.py").read_text(encoding="utf-8")
    streamlit_config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    if "validate_local_ollama_url" not in config_source:
        errors.append("Loopback endpoint validator was not found.")
    if "self.session.trust_env = False" not in runtime_source:
        errors.append("Local HTTP client does not disable inherited proxy environment variables.")
    if "gatherUsageStats = false" not in streamlit_config:
        errors.append("Streamlit usage statistics are not disabled in config.toml.")
    if 'address = "127.0.0.1"' not in streamlit_config:
        errors.append("Streamlit is not configured to bind to 127.0.0.1.")
    # Egress guard checks (Member 2 requirement)
    if "def install_egress_guard" not in runtime_source:
        errors.append("Runtime egress guard function 'install_egress_guard' is missing from local_runtime.py.")
    if "install_egress_guard()" not in runtime_source:
        errors.append(
            "install_egress_guard() is not called at module level in local_runtime.py. "
            "It must run automatically on import before any network call is made."
        )
    if "socket.socket.connect" not in runtime_source:
        errors.append("Egress guard does not patch socket.socket.connect in local_runtime.py.")

    if errors:
        print("FAIL: static privacy checks found issue(s):")
        for error in errors:
            print(f" - {error}")
        return 1
    print("PASS: no external URL literals or hosted-service imports in app/runtime Python code.")
    print(
        "PASS: loopback Ollama guard present; inherited HTTP proxies disabled; "
        "Streamlit telemetry disabled; app bound to localhost; "
        "socket-level egress guard (install_egress_guard) present and installed at module level."
    )
    print("LIMIT: static inspection does not prove runtime behavior. Pair with OLLAMA_NO_CLOUD=1 and network capture/offline testing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
