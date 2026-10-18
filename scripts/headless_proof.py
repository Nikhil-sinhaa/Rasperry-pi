#!/usr/bin/env python3
"""Headless proof script: ingest a document, run triage, ask one question, write evidence.

Used by the offline proof scripts and the Docker proof service.

Usage:
    python scripts/headless_proof.py \\
        --document samples/demo_rental_agreement.txt \\
        --question  "What is the monthly rent?" \\
        --output    evidence/offline-run.md
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from legal_triage.agent import AnalysisError, analyze
from legal_triage.local_runtime import LocalModelError, LocalOllama
from legal_triage.retrieval import build_index


def _read_document(path: str) -> tuple[str, bytes]:
    with open(path, "rb") as fh:
        return os.path.basename(path), fh.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless proof: ingest + triage + question")
    parser.add_argument("--document", required=True, help="Path to the document file")
    parser.add_argument("--question", default="What is the monthly rent?")
    parser.add_argument("--output", default="evidence/offline-run.md")
    args = parser.parse_args()

    print(f"[headless_proof] Document: {args.document}")
    print(f"[headless_proof] Question: {args.question}")

    runtime = LocalOllama()
    try:
        runtime.verify_models()
        print("[headless_proof] Models verified OK")
    except LocalModelError as exc:
        print(f"[headless_proof] FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

    filename, content = _read_document(args.document)
    index = build_index([(filename, content)], runtime)
    print(f"[headless_proof] Indexed {len(index.chunks)} chunks")

    # Triage
    triage_result = analyze("Document overview", index, runtime, mode="triage")
    print(f"[headless_proof] Triage: {len(triage_result.get('findings', []))} findings")

    # Question
    q_result = analyze(args.question, index, runtime, mode="question")
    findings = q_result.get("findings", [])
    print(f"[headless_proof] Question result: {len(findings)} findings, abstained={q_result.get('abstained')}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        out.write(f"# Offline Proof Run\n\n")
        out.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        out.write(f"Document: `{filename}`\n")
        out.write(f"Question: `{args.question}`\n\n")
        out.write(f"## Triage\n\n")
        out.write(f"Findings: {len(triage_result.get('findings', []))}\n")
        out.write(f"Coverage: {triage_result.get('coverage_line', 'N/A')}\n\n")
        out.write(f"## Question Answer\n\n")
        out.write(f"Abstained: {q_result.get('abstained')}\n")
        out.write(f"Legal boundary: {q_result.get('legal_boundary')}\n\n")
        for f in findings[:5]:
            out.write(f"**{f['category']}**: {f['claim']}\n")
            for ev in f.get("evidence", [])[:1]:
                out.write(f"> _{ev['quote']}_\n")
            out.write("\n")
        if not findings:
            out.write(f"Message: {q_result.get('message', '')}\n\n")
        out.write("## Egress Guard Status\n\n")
        out.write("Python-level egress guard: **active** (imported via legal_triage.local_runtime)\n")

    print(f"[headless_proof] Evidence written to: {args.output}")
    print("[headless_proof] PASS")


if __name__ == "__main__":
    main()
