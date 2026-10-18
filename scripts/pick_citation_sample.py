#!/usr/bin/env python3
"""Citation sample picker for criterion L3 (no systematic fabrication).

Picks 20 random findings from a benchmark JSON output file, formats them
for manual review, and writes a filled citation-review.md evidence template.

Usage
-----
    # From the gold eval output:
    python scripts/pick_citation_sample.py --source evidence/model-evaluation.json

    # From the CUAD eval output:
    python scripts/pick_citation_sample.py --source evidence/cuad-eval.json --type cuad

    # Deterministic seed (reproduce same sample):
    python scripts/pick_citation_sample.py --source evidence/model-evaluation.json --seed 42

    # Fewer samples (for quick checks):
    python scripts/pick_citation_sample.py --source evidence/model-evaluation.json --n 10

Output: evidence/citation-review.md (overwritten with a pre-filled table)

After running this script:
1. Open each source document in samples/ or data/cuad_eval/contracts/
2. Find the quoted text on the cited page
3. Fill in each row's Verdict column with one of:
     supports / partly supports / does not support
4. Fill in the Notes column with any observation
5. Update the Summary Counts table and Interpretation paragraph
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence"


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(ROOT),
        )
        return result.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _extract_findings_from_gold_eval(data: dict) -> list[dict[str, Any]]:
    """Extract (finding, source metadata) pairs from run_evals.py JSON output."""
    samples: list[dict[str, Any]] = []
    for case in data.get("cases", []):
        # run_evals.py JSON doesn't include the raw findings — only pass/fail
        # We synthesise a row from the case metadata for manual lookup guidance
        if case.get("status") in ("SKIP", "ERROR"):
            continue
        if not case.get("has_quotes"):
            continue
        samples.append({
            "source": "gold_eval",
            "case_id": case.get("id", "?"),
            "case_type": case.get("type", "?"),
            "document": case.get("document", "?"),
            "question": (case.get("question") or "")[:80],
            "status": case.get("status", "?"),
            "claim": "(look up in the agent output — run the question against the document again)",
            "quote": "(look up finding with source_id in the agent output)",
            "source_id": "?",
            "page": "?",
            "latency_s": case.get("latency_seconds", "?"),
        })
    return samples


def _extract_findings_from_cuad_eval(data: dict) -> list[dict[str, Any]]:
    """Extract finding rows from score_cuad.py JSON output."""
    samples: list[dict[str, Any]] = []
    for row in data.get("rows", []):
        if row.get("status") in ("SKIP",) or str(row.get("status", "")).startswith("ERROR"):
            continue
        if row.get("findings", 0) == 0:
            continue
        samples.append({
            "source": "cuad_eval",
            "case_id": row.get("id", "?"),
            "case_type": row.get("type", "?"),
            "category": row.get("category", "?"),
            "document": "(see gold file for contract filename)",
            "question": "(see gold file for question text)",
            "status": row.get("status", "?"),
            "claim": "(rerun the question against the contract to get the live finding)",
            "quote": "(rerun to retrieve verbatim quote)",
            "source_id": "?",
            "page": "?",
            "latency_s": row.get("seconds", "?"),
        })
    return samples


def _write_citation_review(samples: list[dict], source_path: Path, n: int) -> None:
    out_path = EVIDENCE_DIR / "citation-review.md"
    commit = _git_hash()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [
        "# Citation Review Evidence",
        "",
        "> **Owner:** Member 3 (run) / Member 4 (copy into README §S8)",
        "> **Requirement:** Pick 20 random findings, open the original page, and mark each as",
        "> `supports`, `partly supports`, or `does not support`. Save counts here.",
        "> This is the honest grounding figure for criterion L3.",
        "",
        "---",
        "",
        "## Method",
        "",
        "1. Run `make eval` or `python scripts/run_evals.py` and note the commit hash.",
        "2. Run `python scripts/pick_citation_sample.py --source evidence/model-evaluation.json`",
        "   to regenerate this file from live model output.",
        "3. For each finding, open the source document in `samples/` (or `data/cuad_eval/contracts/`)",
        "   and read the quoted text in its original context.",
        "4. Mark the finding with one of:",
        "   - **supports** – the quote directly and unambiguously supports the claim.",
        "   - **partly supports** – the quote is relevant but the claim slightly overstates or understates.",
        "   - **does not support** – the quote is present but does not logically support the claim.",
        "5. Record each finding in the table below.",
        "",
        "---",
        "",
        "## Run Details",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Date | {date} |",
        f"| Commit hash | `{commit}` |",
        f"| Source file | `{source_path}` |",
        f"| Reviewer | TODO — fill in reviewer name |",
        f"| Sample size | {len(samples)} of {n} requested |",
        "",
        "---",
        "",
        "## Findings (sample)",
        "",
        "**Instructions:** For each row, open the source document, locate the quote, and fill in:",
        "- **Quote present?** — ✅ if you can see the exact text on the cited page, ❌ otherwise",
        "- **Verdict** — supports / partly supports / does not support",
        "- **Notes** — any observation (e.g. quote is truncated, claim is a stretch)",
        "",
        "| # | Case ID | Type | Document | Question (first 60 chars) | Claim (first 60 chars) | Quote (first 60 chars) | Quote present? | Verdict | Notes |",
        "|---|---------|------|----------|--------------------------|----------------------|----------------------|---------------|---------|-------|",
    ]

    for i, s in enumerate(samples, start=1):
        case_id = s.get("case_id", "?")
        case_type = s.get("case_type", "?")
        category = s.get("category", "")
        type_str = f"{case_type}/{category}" if category else case_type
        doc = s.get("document", "?")
        question = (s.get("question") or "")[:60].replace("|", "\\|")
        claim = (s.get("claim") or "")[:60].replace("|", "\\|")
        quote = (s.get("quote") or "")[:60].replace("|", "\\|")
        lines.append(
            f"| {i} | {case_id} | {type_str} | {doc} | {question} "
            f"| {claim} | {quote} | ✅ / ❌ | supports / partly / does not | |"
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
        f"| **Total** | {len(samples)} | 100% |",
        "",
        f"**Quote always present in retrieved chunk:** TODO / {len(samples)}",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "> TODO – one paragraph. Example: '18 of 20 quotes directly support the claim; 2 were partly",
        "> supporting (the model's plain-language restatement slightly broadened the scope).",
        "> Zero \"does not support\" cases in this sample. The verbatim-quote check is working as intended.'",
        "",
        "> **Note:** Verbatim quote validation proves the quote exists in the retrieved source; it does not",
        "> automatically prove the interpretation logically follows. This manual review is the honest",
        "> figure for L3 (no systematic fabrication).",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Citation review template written to {out_path}")
    print(f"  Sample: {len(samples)} findings")
    print(f"  Next step: open each source document and fill in the Verdict column.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick citation sample for manual L3 review")
    parser.add_argument(
        "--source", type=Path, required=True,
        help="JSON report from run_evals.py (evidence/model-evaluation.json) or "
             "score_cuad.py (evidence/cuad-eval.json)",
    )
    parser.add_argument(
        "--type", choices=["gold", "cuad"], default="gold",
        help="Source format: 'gold' for run_evals output, 'cuad' for score_cuad output",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of findings to sample (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"ERROR: Source file not found: {args.source}", file=sys.stderr)
        print(
            "Run `python scripts/run_evals.py --output evidence/model-evaluation.json` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    data = json.loads(args.source.read_text(encoding="utf-8"))

    if args.type == "cuad":
        pool = _extract_findings_from_cuad_eval(data)
    else:
        pool = _extract_findings_from_gold_eval(data)

    if not pool:
        print(
            f"No eligible findings in {args.source}. "
            "Make sure the eval was run with at least some passing cases.",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))

    print(f"Pool size: {len(pool)} findings with quotes")
    print(f"Sample size: {len(sample)}")

    _write_citation_review(sample, args.source, args.n)


if __name__ == "__main__":
    main()
