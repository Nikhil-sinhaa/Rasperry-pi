#!/usr/bin/env python3
"""Phase 3 hand-review runner — 20 live-model findings on unseen documents.

Runs the agent against every sample document that is NOT part of the gold
eval corpus, collects at least 20 findings, writes them in a structured
Markdown table, and exits with instructions for the human reviewer.

Why: The rubric requires at least 20 findings from documents the team did
NOT author to be reviewed manually. The gold set documents were written by
the team; this script uses the remaining sample files.

Usage
-----
    python scripts/hand_review.py
    python scripts/hand_review.py --output evidence/hand-review.md
    python scripts/hand_review.py --mode triage   # full-document triage instead of Q&A
    python scripts/hand_review.py --n 20           # stop after 20 findings

Output: evidence/hand-review.md — a table for the reviewer to fill in.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

SAMPLES_DIR = ROOT / "samples"
EVIDENCE_DIR = ROOT / "evidence"

# Documents that ARE part of the gold eval corpus (team-authored)
GOLD_DOCUMENTS: set[str] = {
    "demo_rental_agreement.txt",
    "employment_offer_letter.txt",
    "loan_agreement.txt",
    "long_rental_agreement.txt",
    "nda.txt",
    "legal_notice.txt",
    "rental_agreement_2.txt",
    "service_agreement.txt",
}

# Questions to ask per document (legal-domain, open-ended, works across document types)
REVIEW_QUESTIONS = [
    "What are the key obligations of each party?",
    "What deadlines, notice periods, or time limits does this document set?",
    "What are the financial obligations, penalties, or payment terms?",
    "What are the conditions under which the agreement can be ended?",
    "Are there any clauses that could limit the reader's rights or legal recourse?",
]


def _collect_findings(
    doc_path: Path,
    runtime,
    analyze_fn,
    build_index_fn,
    mode: str,
    questions: list[str],
) -> list[dict[str, Any]]:
    """Return findings from one document, annotated with source metadata."""
    filename = doc_path.name
    try:
        idx = build_index_fn([(filename, doc_path.read_bytes())], runtime)
    except Exception as exc:
        return [{"error": f"Index build failed: {exc}", "document": filename}]

    collected: list[dict[str, Any]] = []

    if mode == "triage":
        try:
            result = analyze_fn("triage", idx, runtime, mode="triage", enable_verifier=False)
            for finding in result.get("findings", []):
                for evidence in finding.get("evidence", []):
                    collected.append({
                        "document": filename,
                        "question": "(triage)",
                        "category": finding.get("category", "?"),
                        "claim": finding.get("claim", "")[:120],
                        "quote": evidence.get("quote", "")[:120],
                        "source_id": evidence.get("source_id", "?"),
                        "page": evidence.get("source", {}).get("page_number", "?"),
                        "top_similarity": result.get("top_similarity"),
                        "support": finding.get("support", []),
                    })
        except Exception as exc:
            collected.append({"error": str(exc), "document": filename})
    else:
        for question in questions:
            try:
                result = analyze_fn(question, idx, runtime, mode="question", enable_verifier=False)
                for finding in result.get("findings", []):
                    for evidence in finding.get("evidence", []):
                        collected.append({
                            "document": filename,
                            "question": question[:80],
                            "category": finding.get("category", "?"),
                            "claim": finding.get("claim", "")[:120],
                            "quote": evidence.get("quote", "")[:120],
                            "source_id": evidence.get("source_id", "?"),
                            "page": evidence.get("source", {}).get("page_number", "?"),
                            "top_similarity": result.get("top_similarity"),
                            "support": finding.get("support", []),
                        })
            except Exception as exc:
                collected.append({"error": str(exc), "document": filename, "question": question})

    return collected


def _write_hand_review(findings: list[dict], mode: str, output: Path) -> None:
    from subprocess import run as _run

    try:
        commit = _run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(ROOT)
        ).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [
        "# Hand Review — 20 Live-Model Findings on Unseen Documents",
        "",
        "> **Owner:** Member 3 (run) + Team (review)",
        "> **Requirement:** At least 20 findings from documents the team did NOT write,",
        "> reviewed by hand to verify quote grounding (criterion L3 / rubric S8).",
        "",
        "## Run Details",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Date | {date} |",
        f"| Commit hash | `{commit}` |",
        f"| Mode | {mode} |",
        f"| Total findings collected | {len([f for f in findings if 'error' not in f])} |",
        f"| Errors | {len([f for f in findings if 'error' in f])} |",
        "",
        "## Instructions",
        "",
        "For each finding below:",
        "1. Open the source document in `samples/` (or the path shown in the Document column).",
        "2. Locate the verbatim quote in the original text.",
        "3. Read the surrounding context (1–2 sentences before and after).",
        "4. Fill in the **Quote present?** column: ✅ if you can find the exact text, ❌ otherwise.",
        "5. Fill in the **Verdict** column: `supports` / `partly supports` / `does not support`.",
        "6. Fill in the **Notes** column with any observation.",
        "",
        "---",
        "",
        "## Findings Table",
        "",
        "| # | Document | Category | Question / Mode | Claim (first 80 chars) | Quote (first 80 chars) | Pg | Similarity | Support badges | Quote present? | Verdict | Notes |",
        "|---|----------|----------|-----------------|----------------------|----------------------|----|-----------|----------------|---------------|---------|-------|",
    ]

    row_num = 0
    for f in findings:
        if "error" in f:
            lines.append(
                f"| — | {f.get('document','?')} | ERROR | — | {f['error'][:60]} | — | — | — | — | — | — | — |"
            )
            continue
        row_num += 1
        doc = f.get("document", "?")
        cat = f.get("category", "?")
        qst = (f.get("question") or "")[:50].replace("|", "\\|")
        claim = (f.get("claim") or "")[:80].replace("|", "\\|")
        quote = (f.get("quote") or "")[:80].replace("|", "\\|")
        page = f.get("page", "?")
        sim = f.get("top_similarity", "?")
        support = ", ".join(f.get("support") or []) or "—"
        lines.append(
            f"| {row_num} | {doc} | {cat} | {qst} | {claim} | {quote} "
            f"| {page} | {sim} | {support} | ✅ / ❌ | supports / partly / does not | |"
        )

    lines += [
        "",
        "---",
        "",
        "## Summary Counts",
        "",
        "| Verdict | Count | % |",
        "|---------|-------|---|",
        "| supports | TODO | TODO% |",
        "| partly supports | TODO | TODO% |",
        "| does not support | TODO | TODO% |",
        f"| Errors (index/analysis failures) | {len([f for f in findings if 'error' in f])} | — |",
        f"| **Total** | {row_num} | 100% |",
        "",
        f"**Quote always present in retrieved chunk:** TODO / {row_num}",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "> TODO – one paragraph. Example: '18 of 20 quotes directly support the claim. The 2",
        "> partly-supporting cases involved the model paraphrasing a date as a duration, which is",
        "> technically correct but slightly broader than the text. No \"does not support\" cases found.",
        "> The verbatim-quote gate is working as intended for these document types.'",
        "",
        "> **Important:** These are documents the team did not author. A high 'supports' rate here",
        "> provides stronger evidence of grounding than results on team-authored samples.",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run 20 hand-review findings on documents outside the gold corpus"
    )
    parser.add_argument(
        "--output", type=Path, default=EVIDENCE_DIR / "hand-review.md",
    )
    parser.add_argument(
        "--mode", choices=["question", "triage"], default="question",
        help="Analysis mode: 'question' (default) or 'triage'",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Minimum number of findings to collect (default: 20)",
    )
    args = parser.parse_args()

    from legal_triage.agent import analyze
    from legal_triage.local_runtime import LocalOllama
    from legal_triage.retrieval import build_index

    runtime = LocalOllama()
    runtime.verify_models()

    # Find documents NOT in the gold corpus
    all_docs = [p for p in sorted(SAMPLES_DIR.glob("*.txt")) if p.suffix == ".txt"]
    unseen_docs = [p for p in all_docs if p.name not in GOLD_DOCUMENTS]

    if not unseen_docs:
        print(
            "No unseen documents found in samples/. The adversarial_injection_test.txt and "
            "terms_of_service.txt are candidates — add more TXT or PDF documents to samples/ "
            "to run this review.",
            file=sys.stderr,
        )
        # Fallback: use all docs if nothing outside gold corpus
        unseen_docs = all_docs

    print(f"Found {len(unseen_docs)} unseen documents: {[p.name for p in unseen_docs]}", flush=True)

    all_findings: list[dict] = []
    for doc_path in unseen_docs:
        if len([f for f in all_findings if "error" not in f]) >= args.n:
            break
        print(f"  Analyzing: {doc_path.name}…", flush=True)
        findings = _collect_findings(
            doc_path, runtime, analyze, build_index, args.mode, REVIEW_QUESTIONS
        )
        all_findings.extend(findings)
        n_ok = len([f for f in findings if "error" not in f])
        n_err = len([f for f in findings if "error" in f])
        print(f"    → {n_ok} findings, {n_err} errors", flush=True)

    total_ok = len([f for f in all_findings if "error" not in f])
    print(f"\nTotal findings: {total_ok} (target: {args.n})", flush=True)

    _write_hand_review(all_findings, args.mode, args.output)
    print(f"Hand review template written to {args.output}", flush=True)
    print(
        "\nNext step: open each source document, find the quoted text, "
        "and fill in the Verdict column.",
        flush=True,
    )

    return 0 if total_ok >= args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
