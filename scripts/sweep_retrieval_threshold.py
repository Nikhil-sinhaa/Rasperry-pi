"""Retrieval similarity threshold sweep script.

Run this against Member 3's eval set to pick MIN_RETRIEVAL_SIMILARITY.

Usage
-----
    python scripts/sweep_retrieval_threshold.py \
        --eval-dir eval/gold/ \
        --output evidence/retrieval-threshold-sweep.md

The script sweeps MIN_RETRIEVAL_SIMILARITY from 0.15 to 0.45 in steps of 0.05
and reports:
  - invented_answer_rate  (must be 0 on absent-information questions)
  - false_abstention_rate (lower is better)
  - answer_correct_rate

Pick the threshold with 0 invented answers and the fewest false abstentions.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

THRESHOLDS = [round(v * 0.05 + 0.15, 2) for v in range(7)]  # 0.15 → 0.45


def _run_eval_at_threshold(threshold: float, eval_dir: Path, runtime) -> dict:
    """Run the eval set at a given threshold. Returns aggregate stats."""
    os.environ["MIN_RETRIEVAL_SIMILARITY"] = str(threshold)

    # Re-import SETTINGS so the new env var takes effect
    import legal_triage.config as cfg_mod
    import importlib
    importlib.reload(cfg_mod)
    import legal_triage.agent as agent_mod
    importlib.reload(agent_mod)
    from legal_triage.agent import analyze

    gold_files = sorted(eval_dir.glob("*.json"))
    if not gold_files:
        raise FileNotFoundError(f"No gold JSON files found in {eval_dir}")

    total = absent_total = 0
    invented = false_abstain = correct = 0

    for gold_path in gold_files:
        with gold_path.open() as fh:
            cases = json.load(fh)
        for case in cases:
            q_type = case.get("type", "answerable")
            question = case.get("question", "")
            expected_facts = case.get("expected_facts", [])
            # Build a tiny index from the gold document if available
            # (In a real eval, you would load the actual indexed document.)
            # This script skeleton prints instructions; the team must wire
            # it to the actual indexed documents.
            total += 1
            if q_type == "absent":
                absent_total += 1

    return {
        "threshold": threshold,
        "total_cases": total,
        "absent_cases": absent_total,
        "invented_answer_rate": "FILL_IN",
        "false_abstention_rate": "FILL_IN",
        "answer_correct_rate": "FILL_IN",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep retrieval similarity threshold.")
    parser.add_argument("--eval-dir", required=True, type=Path, help="Path to eval/gold/ directory.")
    parser.add_argument("--output", default="evidence/retrieval-threshold-sweep.md", type=Path)
    args = parser.parse_args()

    rows: list[dict] = []
    for threshold in THRESHOLDS:
        print(f"Sweeping threshold={threshold}…")
        try:
            row = _run_eval_at_threshold(threshold, args.eval_dir, runtime=None)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            row = {"threshold": threshold, "error": str(exc)}
        rows.append(row)

    # Write markdown table
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        out.write("# Retrieval Similarity Threshold Sweep\n\n")
        out.write("| Threshold | Invented Answer Rate | False Abstention Rate | Answer Correct Rate |\n")
        out.write("|-----------|---------------------|----------------------|--------------------|\n")
        for row in rows:
            if "error" in row:
                out.write(f"| {row['threshold']} | ERROR | ERROR | ERROR |\n")
            else:
                out.write(
                    f"| {row['threshold']} "
                    f"| {row['invented_answer_rate']} "
                    f"| {row['false_abstention_rate']} "
                    f"| {row['answer_correct_rate']} |\n"
                )
        out.write("\n**Chosen threshold:** TODO – pick value with 0 invented answers and fewest false abstentions.\n")

    print(f"Sweep table written to {args.output}")


if __name__ == "__main__":
    main()
