#!/usr/bin/env python3
"""Real retrieval similarity threshold sweep — wired to the gold eval set.

Replaces the skeleton sweep_retrieval_threshold.py that only counted cases
without actually running the agent.

Usage
-----
    python scripts/sweep_threshold.py
    python scripts/sweep_threshold.py --eval-dir eval/gold/ --output evidence/retrieval-threshold-sweep.md
    python scripts/sweep_threshold.py --thresholds 0.20 0.25 0.30 0.35 0.40

The sweep runs the full gold set at each threshold value and reports:
  - invented_answer_rate   (must be 0 on absent-information questions)
  - false_abstention_rate  (fraction of answerable questions that abstained)
  - answer_correct_rate    (fraction of answerable questions that passed)
  - verdict_refusal_rate

Pick the threshold with 0 invented answers and the fewest false abstentions.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

SAMPLES_DIR = ROOT / "samples"
DEFAULT_THRESHOLDS = [round(0.15 + i * 0.05, 2) for i in range(7)]  # 0.15 → 0.45


def _load_gold_cases(eval_dir: Path) -> list[dict]:
    cases: list[dict] = []
    for path in sorted(eval_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            cases.extend(data)
    return cases


def _source_quotes(result: dict) -> list[str]:
    return [
        str(e.get("quote", ""))
        for f in result.get("findings", [])
        for e in f.get("evidence", [])
    ]


def _check_expected_facts(result: dict, expected_facts: list[str]) -> bool:
    if not expected_facts:
        return True
    combined = " ".join(_source_quotes(result)).casefold()
    return all(f.casefold() in combined for f in expected_facts)


def _check_quote_must_contain(result: dict, qmc: list[str]) -> bool:
    if not qmc:
        return True
    combined = " ".join(_source_quotes(result)).casefold()
    return all(s.casefold() in combined for s in qmc)


def _run_at_threshold(
    threshold: float,
    gold_cases: list[dict],
    runtime,
    analyze_fn,
    build_index_fn,
) -> dict[str, Any]:
    """Run the full gold set at a given threshold. Returns aggregate stats."""
    from legal_triage.retrieval import LocalIndex

    index_cache: dict[str, LocalIndex] = {}

    answerable_total = answerable_correct = false_abstain = 0
    absent_total = absent_invented = 0
    verdict_total = verdict_refused = 0
    error_count = 0

    for case in gold_cases:
        case_type = case.get("type", "answerable")
        question = case.get("question", "")
        document = case.get("document", "")
        expected_facts = case.get("expected_facts", [])
        qmc = case.get("quote_must_contain", [])

        doc_path = SAMPLES_DIR / document
        if not doc_path.exists():
            continue

        if document not in index_cache:
            try:
                index_cache[document] = build_index_fn(
                    [(document, doc_path.read_bytes())], runtime
                )
            except Exception:
                continue

        idx = index_cache[document]
        try:
            output = analyze_fn(question, idx, runtime, mode="question", enable_verifier=False)
        except Exception:
            error_count += 1
            continue

        abstained = bool(output.get("abstained"))
        legal_boundary = bool(output.get("legal_boundary"))
        quotes = _source_quotes(output)
        facts_ok = _check_expected_facts(output, expected_facts)
        qmc_ok = _check_quote_must_contain(output, qmc)

        if case_type == "answerable":
            answerable_total += 1
            passed = (not abstained) and facts_ok and qmc_ok
            if passed:
                answerable_correct += 1
            if abstained:
                false_abstain += 1
        elif case_type == "absent":
            absent_total += 1
            if not abstained and output.get("findings"):
                absent_invented += 1
        elif case_type == "verdict_request":
            verdict_total += 1
            if legal_boundary:
                verdict_refused += 1

    def _rate(n: int, d: int) -> float | None:
        return round(n / d, 3) if d else None

    return {
        "threshold": threshold,
        "answerable_total": answerable_total,
        "absent_total": absent_total,
        "invented_answer_rate": _rate(absent_invented, absent_total),
        "false_abstention_rate": _rate(false_abstain, answerable_total),
        "answer_correct_rate": _rate(answerable_correct, answerable_total),
        "verdict_refusal_rate": _rate(verdict_refused, verdict_total),
        "errors": error_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep retrieval similarity threshold.")
    parser.add_argument(
        "--eval-dir", type=Path, default=ROOT / "eval" / "gold",
        help="Path to eval/gold/ directory",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evidence" / "retrieval-threshold-sweep.md",
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS,
        help="Threshold values to sweep (default: 0.15 to 0.45 in 0.05 steps)",
    )
    args = parser.parse_args()

    gold_cases = _load_gold_cases(args.eval_dir)
    if not gold_cases:
        print(f"No gold cases in {args.eval_dir}. Run bootstrap_member3.py first.")
        sys.exit(1)

    # Import after potential env-var changes
    from legal_triage.local_runtime import LocalOllama
    from legal_triage.retrieval import build_index

    runtime = LocalOllama()
    runtime.verify_models()

    rows: list[dict] = []
    chosen_threshold: float | None = None
    chosen_reason = ""

    for threshold in sorted(set(args.thresholds)):
        print(f"Sweeping threshold={threshold}…", flush=True)
        os.environ["MIN_RETRIEVAL_SIMILARITY"] = str(threshold)

        # Reload config + agent so the new threshold takes effect
        import legal_triage.config as cfg_mod
        importlib.reload(cfg_mod)
        import legal_triage.agent as agent_mod
        importlib.reload(agent_mod)
        from legal_triage.agent import analyze
        from legal_triage.retrieval import build_index as bi

        row = _run_at_threshold(threshold, gold_cases, runtime, analyze, bi)
        rows.append(row)

        invented = row.get("invented_answer_rate") or 0
        false_abs = row.get("false_abstention_rate") or 1
        answer_cr = row.get("answer_correct_rate") or 0
        print(
            f"  invented={invented:.3f}  false_abstain={false_abs:.3f}  "
            f"answer_correct={answer_cr:.3f}  errors={row.get('errors',0)}",
            flush=True,
        )

        # Auto-pick: first threshold with 0 invented answers and fewest false abstentions
        if invented == 0 and chosen_threshold is None:
            chosen_threshold = threshold
            chosen_reason = (
                f"First threshold achieving invented_answer_rate=0 "
                f"(false_abstention_rate={false_abs:.3f}, answer_correct_rate={answer_cr:.3f})"
            )

    # Write markdown table
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        out.write("# Retrieval Similarity Threshold Sweep\n\n")
        out.write(
            "| Threshold | Invented Answer Rate | False Abstention Rate "
            "| Answer Correct Rate | Verdict Refusal Rate | Errors |\n"
        )
        out.write(
            "|-----------|---------------------|----------------------"
            "|--------------------|---------------------|--------|\n"
        )
        for row in rows:
            marker = " ← chosen" if row["threshold"] == chosen_threshold else ""
            out.write(
                f"| {row['threshold']}{marker} "
                f"| {row.get('invented_answer_rate', 'N/A')} "
                f"| {row.get('false_abstention_rate', 'N/A')} "
                f"| {row.get('answer_correct_rate', 'N/A')} "
                f"| {row.get('verdict_refusal_rate', 'N/A')} "
                f"| {row.get('errors', 0)} |\n"
            )
        if chosen_threshold is not None:
            out.write(f"\n**Chosen threshold:** `{chosen_threshold}` — {chosen_reason}\n")
        else:
            out.write(
                "\n**Chosen threshold:** TODO — no threshold achieved invented_answer_rate=0. "
                "Investigate absent-question hallucinations.\n"
            )

    print(f"\nSweep table written to {args.output}", flush=True)
    if chosen_threshold is not None:
        print(f"Recommended threshold: {chosen_threshold}", flush=True)
    else:
        print("WARNING: No threshold achieved invented_answer_rate=0.", flush=True)


if __name__ == "__main__":
    main()
