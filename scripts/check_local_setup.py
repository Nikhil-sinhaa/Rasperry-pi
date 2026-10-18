#!/usr/bin/env python3
"""Check local Ollama/model readiness without opening or uploading user documents."""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from legal_triage.local_runtime import LocalOllama  # noqa: E402


def main() -> int:
    runtime = LocalOllama()
    try:
        details = runtime.verify_models()
        vectors = runtime.embed(["Local setup check. No private document is used."])
        response = runtime.chat_json(
            'You are a local runtime check. Return JSON only: {"status":"ok"}.',
            'Return exactly the JSON object {"status":"ok"}.',
        )
    except Exception as exc:  # only print exception type; do not print model output or prompts
        print(f"FAILED: {type(exc).__name__}: local runtime or model setup is not ready.")
        print("No hosted-model fallback was attempted. Check Ollama status and `ollama list`.")
        return 1

    print("PASS: loopback Ollama endpoint is reachable.")
    print(f"PASS: local chat model configured: {details['chat_model']}")
    print(f"PASS: local embedding model configured: {details['embedding_model']}")
    print(f"PASS: embedding dimension: {vectors.shape[1]}; local chat response received ({len(response)} chars).")
    print("NOTE: this readiness check does not prove all external traffic is blocked; run the network test too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
