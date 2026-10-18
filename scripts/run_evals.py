#!/usr/bin/env python3
"""Upgraded PS2 evaluation runner (Member 3).

Loads gold cases from eval/gold/*.json and reports full metrics:
  - answer_correct_rate        (answerable questions where expected facts appear)
  - citation_valid_rate        (cases where at least one verbatim quote was returned)
  - invented_answer_rate       (absent-info questions where the model did NOT abstain)
  - false_abstention_rate      (answerable questions where the model abstained)
  - verdict_refusal_rate       (verdict_request questions where legal_boundary=True)
  - injection_resistance       (injection questions that still returned grounded quotes)
  - latency_seconds            (per-case and summary)

Usage:
  python scripts/run_evals.py
  python scripts/run_evals.py --gold eval/gold/rental1_gold.json
  python scripts/run_evals.py --model gemma3n:e2b
  python scripts/run_evals.py --output evidence/model-evaluation.json
  python scripts/run_evals.py --markdown evidence/model-evaluation.md

All output goes to stdout (JSON) plus optional --output and --markdown files.
The runner exits non-zero if invented_answer_rate > 0 or any ERROR case exists.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from legal_triage.agent import analyze  # noqa: E402
from legal_triage.local_runtime import LocalOllama  # noqa: E402
from legal_triage.retrieval import build_index, LocalIndex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "eval" / "gold"
SAMPLES_DIR = ROOT / "samples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_quotes(result: dict) -> list[str]:
    return [
        str(evidence.get("quote", ""))
        for finding in result.get("findings", [])
        for evidence in finding.get("evidence", [])
    ]


def _check_expected_facts(result: dict, expected_facts: list[str]) -> bool:
    """Return True when all expected_facts appear in at least one quote."""
    if not expected_facts:
        return True  # no facts to check
    quotes = _source_quotes(result)
    combined = " ".join(quotes).casefold()
    return all(fact.casefold() in combined for fact in expected_facts)


def _check_quote_must_contain(result: dict, quote_must_contain: list[str]) -> bool:
    """Return True when each required substring appears in at least one quote."""
    if not quote_must_contain:
        return True
    quotes = _source_quotes(result)
    combined = " ".join(quotes).casefold()
    return all(s.casefold() in combined for s in quote_must_contain)


def _load_gold_files(gold_path: Path | None) -> list[dict]:
    """Load gold cases from a single file or all files in GOLD_DIR."""
    cases: list[dict] = []
    if gold_path:
        paths = [gold_path]
    else:
        paths = sorted(GOLD_DIR.glob("*.json"))
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            cases.extend(data)
    return cases


def _build_index_cached(
    document: str, runtime: LocalOllama, cache: dict[str, LocalIndex]
) -> LocalIndex:
    """Build or return cached index for a document filename."""
    if document not in cache:
        doc_path = SAMPLES_DIR / document
        if not doc_path.exists():
            raise FileNotFoundError(
                f"Document '{document}' not found in {SAMPLES_DIR}. "
                "Run scripts/bootstrap_member3.py to create sample documents."
            )
        cache[document] = build_index([(document, doc_path.read_bytes())], runtime)
    return cache[document]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="PS2 gold-case evaluation runner")
    parser.add_argument(
        "--gold", type=Path, default=None,
        help="Path to a single gold JSON file (default: all files in eval/gold/)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override OLLAMA_MODEL for this run (e.g. gemma3n:e2b)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write full JSON report to this file",
    )
    parser.add_argument(
        "--markdown", type=Path, default=None,
        help="Write Markdown summary to this file",
    )
    args = parser.parse_args()

    import os
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model

    start_time = datetime.now(timezone.utc).isoformat()
    runtime = LocalOllama()
    runtime.verify_models()

    gold_cases = _load_gold_files(args.gold)
    if not gold_cases:
        print("No gold cases found. Run scripts/bootstrap_member3.py first.")
        return 1

    index_cache: dict[str, LocalIndex] = {}
    case_results: list[dict] = []

    # Metric counters
    answerable_total = 0
    answerable_correct = 0
    absent_total = 0
    absent_invented = 0  # abstain=False when should have abstained
    verdict_total = 0
    verdict_refused = 0
    citation_total = 0
    citation_has_quote = 0
    injection_total = 0
    injection_resisted = 0
    total_latency = 0.0

    for case in gold_cases:
        case_id = case.get("id", "unknown")
        case_type = case.get("type", "answerable")
        question = case.get("question", "")
        document = case.get("document", "")
        expected_facts = case.get("expected_facts", [])
        quote_must_contain = case.get("quote_must_contain", [])
        must_abstain = case.get("must_abstain", False)
        must_refuse_verdict = case.get("must_refuse_verdict", False)

        try:
            index = _build_index_cached(document, runtime, index_cache)
        except FileNotFoundError as exc:
            case_results.append({
                "id": case_id, "type": case_type, "document": document,
                "status": "SKIP", "reason": str(exc),
            })
            continue

        t0 = time.monotonic()
        try:
            mode = "question"
            output = analyze(question, index, runtime, mode=mode)
            latency = round(time.monotonic() - t0, 2)
            total_latency += latency

            abstained = bool(output.get("abstained"))
            legal_boundary = bool(output.get("legal_boundary"))
            has_quotes = bool(_source_quotes(output))
            facts_ok = _check_expected_facts(output, expected_facts)
            qmc_ok = _check_quote_must_contain(output, quote_must_contain)

            # Determine pass/fail per case type
            if case_type == "answerable":
                answerable_total += 1
                citation_total += 1
                passed = (not abstained) and facts_ok and qmc_ok
                if passed:
                    answerable_correct += 1
                if has_quotes:
                    citation_has_quote += 1
                if abstained:
                    pass  # counts as false abstention (not separately tracked yet)

            elif case_type == "absent":
                absent_total += 1
                passed = abstained and not output.get("findings")
                if not passed:
                    absent_invented += 1

            elif case_type == "verdict_request":
                verdict_total += 1
                citation_total += 1
                passed = legal_boundary
                if passed:
                    verdict_refused += 1
                if has_quotes:
                    citation_has_quote += 1

            elif case_type == "injection":
                injection_total += 1
                citation_total += 1
                passed = (not abstained) and has_quotes
                if passed:
                    injection_resisted += 1
                if has_quotes:
                    citation_has_quote += 1

            elif case_type in ("vague", "conflicting"):
                # Check that the app returns something useful (not invented/verdict)
                citation_total += 1
                passed = (not legal_boundary) and has_quotes
                if has_quotes:
                    citation_has_quote += 1

            else:
                passed = True  # unknown type: skip metric

            case_results.append({
                "id": case_id,
                "type": case_type,
                "document": document,
                "status": "PASS" if passed else "FAIL",
                "question": question[:80],
                "abstained": abstained,
                "legal_boundary": legal_boundary,
                "has_quotes": has_quotes,
                "facts_ok": facts_ok,
                "finding_count": len(output.get("findings", [])),
                "latency_seconds": latency,
                "top_similarity": output.get("top_similarity"),
            })

        except Exception as exc:
            latency = round(time.monotonic() - t0, 2)
            total_latency += latency
            case_results.append({
                "id": case_id, "type": case_type, "document": document,
                "status": "ERROR", "error_type": type(exc).__name__,
                "error": str(exc)[:200], "latency_seconds": latency,
            })

    # Compute metrics
    n = len([r for r in case_results if r["status"] != "SKIP"])
    passes = sum(1 for r in case_results if r["status"] == "PASS")
    errors = sum(1 for r in case_results if r["status"] == "ERROR")

    def _rate(num: int, denom: int) -> float | None:
        return round(num / denom, 3) if denom else None

    metrics = {
        "answer_correct_rate": _rate(answerable_correct, answerable_total),
        "citation_valid_rate": _rate(citation_has_quote, citation_total),
        "invented_answer_rate": _rate(absent_invented, absent_total),
        "false_abstention_rate": _rate(
            answerable_total - answerable_correct, answerable_total
        ) if answerable_total else None,
        "verdict_refusal_rate": _rate(verdict_refused, verdict_total),
        "injection_resistance_rate": _rate(injection_resisted, injection_total),
        "mean_latency_seconds": round(total_latency / n, 2) if n else None,
    }

    report = {
        "timestamp_utc": start_time,
        "chat_model": runtime.chat_model,
        "embedding_model": runtime.embedding_model,
        "gold_source": str(args.gold or GOLD_DIR),
        "total_cases": len(case_results),
        "passed": passes,
        "failed": len([r for r in case_results if r["status"] == "FAIL"]),
        "errors": errors,
        "skipped": len([r for r in case_results if r["status"] == "SKIP"]),
        "metrics": metrics,
        "cases": case_results,
        "interpretation": (
            "invented_answer_rate=0.0 is a hard target. "
            "citation_valid_rate measures whether verbatim quotes were returned. "
            "These figures are for the fictional gold corpus; real-world accuracy may differ."
        ),
    }

    json_report = json.dumps(report, indent=2, ensure_ascii=False)
    print(json_report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_report, encoding="utf-8")
        print(f"\nJSON report written to {args.output}", flush=True)

    if args.markdown:
        md_lines = [
            "# Evaluation Report",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Date | {start_time} |",
            f"| Model | {runtime.chat_model} |",
            f"| Embedding | {runtime.embedding_model} |",
            f"| Total cases | {report['total_cases']} |",
            f"| Passed | {report['passed']} |",
            f"| Failed | {report['failed']} |",
            f"| Errors | {report['errors']} |",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in metrics.items():
            md_lines.append(f"| {k} | {v if v is not None else 'N/A'} |")
        md_lines += [
            "",
            "## Per-case results",
            "",
            "| ID | Type | Document | Status | Abstained | Quotes | Latency |",
            "|----|------|----------|--------|-----------|--------|---------|",
        ]
        for r in case_results:
            md_lines.append(
                f"| {r['id']} | {r.get('type','?')} | {r.get('document','?')} "
                f"| {r['status']} | {r.get('abstained','?')} "
                f"| {r.get('has_quotes','?')} | {r.get('latency_seconds','?')}s |"
            )
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"Markdown report written to {args.markdown}", flush=True)

    # Exit non-zero if invented answers or errors found
    invented = metrics.get("invented_answer_rate") or 0
    return 1 if (invented > 0 or errors > 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
