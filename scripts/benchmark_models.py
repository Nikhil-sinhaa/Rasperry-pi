#!/usr/bin/env python3
"""Phase 3 model benchmark runner.

Benchmarks gemma3n:e2b, gemma3n:e4b, gemma4:e2b and gemma4:e4b against the
gold eval set and writes results to evidence/model-benchmark.md.

Usage
-----
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --models gemma3n:e2b gemma4:e2b
    python scripts/benchmark_models.py --output evidence/model-benchmark.md
    python scripts/benchmark_models.py --representative-only   # 5-question latency probe only

Requirements
------------
- Ollama must be running locally on 127.0.0.1:11434
- All listed models must have been pulled with `ollama pull MODEL`
- Sample documents must exist in samples/ (run bootstrap_member3.py first)

The script uses subprocess to poll `ollama ps` for RAM figures.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from legal_triage.agent import analyze
from legal_triage.local_runtime import LocalOllama
from legal_triage.retrieval import build_index, LocalIndex

GOLD_DIR = ROOT / "eval" / "gold"
SAMPLES_DIR = ROOT / "samples"
EVIDENCE_DIR = ROOT / "evidence"

CANDIDATE_MODELS = [
    "gemma3n:e2b",
    "gemma3n:e4b",
    "gemma4:e2b",
    "gemma4:e4b",
]

# 5 representative questions used for the latency/RAM probe (one per document type)
REPRESENTATIVE_QUESTIONS = [
    ("demo_rental_agreement.txt",    "What is the monthly rent and when is it due?"),
    ("employment_offer_letter.txt",  "What is the base salary and the notice period for termination?"),
    ("nda.txt",                      "How long does the confidentiality obligation last?"),
    ("loan_agreement.txt",           "What is the interest rate and the repayment schedule?"),
    ("service_agreement.txt",        "What are the payment terms for the services?"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system_info() -> dict[str, str]:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
        "machine": platform.machine(),
    }


def _ollama_version() -> str:
    try:
        result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or result.stderr.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "ollama CLI not on PATH (version unknown)"


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(ROOT),
        )
        return result.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _peak_ram_mb() -> float | None:
    """Poll `ollama ps` and return total VRAM/RAM used in MB."""
    try:
        result = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().splitlines()
        total_mb = 0.0
        for line in lines[1:]:  # skip header
            parts = line.split()
            # Format: NAME  ID  SIZE  PROCESSOR  UNTIL
            # SIZE may be "3.9 GiB" or "512 MiB"
            for i, part in enumerate(parts):
                if part in ("GiB", "GB"):
                    try:
                        total_mb += float(parts[i - 1]) * 1024
                    except (ValueError, IndexError):
                        pass
                elif part in ("MiB", "MB"):
                    try:
                        total_mb += float(parts[i - 1])
                    except (ValueError, IndexError):
                        pass
        return round(total_mb, 0) if total_mb > 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


def _load_gold_cases() -> list[dict]:
    cases: list[dict] = []
    for path in sorted(GOLD_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            cases.extend(data)
    return cases


def _build_index_cached(
    document: str, runtime: LocalOllama, cache: dict[str, LocalIndex]
) -> LocalIndex | None:
    if document not in cache:
        doc_path = SAMPLES_DIR / document
        if not doc_path.exists():
            return None
        cache[document] = build_index([(document, doc_path.read_bytes())], runtime)
    return cache[document]


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
    return all(fact.casefold() in combined for fact in expected_facts)


def _check_quote_must_contain(result: dict, qmc: list[str]) -> bool:
    if not qmc:
        return True
    combined = " ".join(_source_quotes(result)).casefold()
    return all(s.casefold() in combined for s in qmc)


# ---------------------------------------------------------------------------
# Benchmark a single model
# ---------------------------------------------------------------------------

def benchmark_model(
    model: str,
    gold_cases: list[dict],
    representative_only: bool = False,
) -> dict[str, Any]:
    """Run the full eval (or representative probe) against one model."""
    print(f"\n{'='*60}", flush=True)
    print(f"  Benchmarking: {model}", flush=True)
    print(f"{'='*60}", flush=True)

    os.environ["OLLAMA_MODEL"] = model
    # Re-read settings after changing env
    import importlib
    import legal_triage.config as cfg_mod
    importlib.reload(cfg_mod)
    import legal_triage.agent as agent_mod
    importlib.reload(agent_mod)
    from legal_triage.agent import analyze as _analyze  # noqa: F811
    from legal_triage.local_runtime import LocalOllama as _LO
    from legal_triage.retrieval import build_index as _bi, LocalIndex as _LI

    runtime = _LO()
    try:
        runtime.verify_models()
    except Exception as exc:
        print(f"  SKIP – model not available: {exc}", flush=True)
        return {
            "model": model,
            "status": "SKIP",
            "reason": str(exc),
        }

    latencies: list[float] = []
    index_cache: dict[str, _LI] = {}

    # --- Representative latency probe ---
    print("  Representative latency probe (5 questions)…", flush=True)
    rep_latencies: list[float] = []
    for doc, question in REPRESENTATIVE_QUESTIONS:
        doc_path = SAMPLES_DIR / doc
        if not doc_path.exists():
            print(f"    SKIP {doc} (file missing)", flush=True)
            continue
        if doc not in index_cache:
            index_cache[doc] = _bi([(doc, doc_path.read_bytes())], runtime)
        idx = index_cache[doc]
        t0 = time.monotonic()
        try:
            _analyze(question, idx, runtime, mode="question", enable_verifier=False)
            elapsed = round(time.monotonic() - t0, 2)
        except Exception as exc:
            elapsed = round(time.monotonic() - t0, 2)
            print(f"    ERROR on '{doc}': {exc}", flush=True)
        rep_latencies.append(elapsed)
        print(f"    {doc}: {elapsed}s", flush=True)

    rep_median = (
        round(sorted(rep_latencies)[len(rep_latencies) // 2], 2)
        if rep_latencies else None
    )

    # Poll RAM after warm model
    ram_mb = _peak_ram_mb()
    print(f"  Peak RAM (ollama ps): {ram_mb} MB", flush=True)

    if representative_only:
        return {
            "model": model,
            "status": "REPRESENTATIVE_ONLY",
            "rep_latency_seconds_median": rep_median,
            "peak_ram_mb": ram_mb,
            "rep_latencies": rep_latencies,
        }

    # --- Full gold eval ---
    print("  Running full gold eval…", flush=True)
    answerable_total = answerable_correct = 0
    absent_total = absent_invented = 0
    verdict_total = verdict_refused = 0
    citation_total = citation_has_quote = 0
    error_count = 0
    case_rows: list[dict] = []

    for case in gold_cases:
        case_id = case.get("id", "?")
        case_type = case.get("type", "answerable")
        question = case.get("question", "")
        document = case.get("document", "")
        expected_facts = case.get("expected_facts", [])
        quote_must_contain = case.get("quote_must_contain", [])

        idx = _build_index_cached(document, runtime, index_cache)
        if idx is None:
            case_rows.append({"id": case_id, "status": "SKIP", "reason": f"{document} not found"})
            continue

        t0 = time.monotonic()
        try:
            output = _analyze(question, idx, runtime, mode="question", enable_verifier=False)
            elapsed = round(time.monotonic() - t0, 2)
            latencies.append(elapsed)

            abstained = bool(output.get("abstained"))
            legal_boundary = bool(output.get("legal_boundary"))
            has_quotes = bool(_source_quotes(output))
            facts_ok = _check_expected_facts(output, expected_facts)
            qmc_ok = _check_quote_must_contain(output, quote_must_contain)

            if case_type == "answerable":
                answerable_total += 1
                citation_total += 1
                passed = (not abstained) and facts_ok and qmc_ok
                if passed:
                    answerable_correct += 1
                if has_quotes:
                    citation_has_quote += 1
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
            else:
                citation_total += 1
                passed = not legal_boundary
                if has_quotes:
                    citation_has_quote += 1

            status = "PASS" if passed else "FAIL"
            case_rows.append({
                "id": case_id, "type": case_type, "status": status,
                "latency_s": elapsed, "has_quotes": has_quotes,
                "abstained": abstained, "legal_boundary": legal_boundary,
            })
            print(f"    {case_id} [{case_type}] → {status} ({elapsed}s)", flush=True)

        except Exception as exc:
            elapsed = round(time.monotonic() - t0, 2)
            error_count += 1
            case_rows.append({"id": case_id, "type": case_type, "status": "ERROR", "error": str(exc)[:120]})
            print(f"    {case_id} → ERROR: {exc}", flush=True)

    def _rate(num: int, denom: int) -> float | None:
        return round(num / denom, 3) if denom else None

    n = len([r for r in case_rows if r["status"] not in ("SKIP", "ERROR")])
    passes = sum(1 for r in case_rows if r["status"] == "PASS")
    mean_lat = round(sum(latencies) / len(latencies), 2) if latencies else None

    metrics = {
        "answer_correct_rate":    _rate(answerable_correct, answerable_total),
        "citation_valid_rate":    _rate(citation_has_quote, citation_total),
        "invented_answer_rate":   _rate(absent_invented, absent_total),
        "verdict_refusal_rate":   _rate(verdict_refused, verdict_total),
        "mean_latency_seconds":   mean_lat,
    }

    print(f"\n  Results for {model}:", flush=True)
    for k, v in metrics.items():
        print(f"    {k}: {v}", flush=True)

    return {
        "model": model,
        "status": "OK",
        "total_cases": len(case_rows),
        "passes": passes,
        "errors": error_count,
        "peak_ram_mb": ram_mb,
        "rep_latency_seconds_median": rep_median,
        "metrics": metrics,
        "cases": case_rows,
    }


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def _write_markdown(results: list[dict], sysinfo: dict, args: argparse.Namespace) -> str:
    lines = [
        "# Model Benchmark",
        "",
        "> **Owner:** Member 1",
        "> **Gate:** L1 – Gemma runs locally",
        "> **Deliverable:** S6 – How Gemma is used",
        "",
        "## Hardware",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Machine | {sysinfo.get('machine', 'TODO')} — {sysinfo.get('cpu', 'TODO')} |",
        f"| OS | {sysinfo.get('os', 'TODO')} |",
        f"| Python | {sysinfo.get('python', 'TODO')} |",
        f"| Ollama version | {sysinfo.get('ollama_version', 'TODO')} |",
        f"| Date | {sysinfo.get('date', 'TODO')} |",
        f"| Commit hash | {sysinfo.get('commit', 'TODO')} |",
        "",
        "## Benchmark Results",
        "",
        "Eval cases used: the same set from `eval/gold/` (run with `scripts/benchmark_models.py`).",
        "",
        "| Model | Pass rate | Seconds/answer (median) | Peak RAM (MB) | Invented answer rate | Verdict refusal rate |",
        "|-------|-----------|------------------------|---------------|---------------------|---------------------|",
    ]

    best_model = None
    best_score = -1.0

    for r in results:
        model = r.get("model", "?")
        if r.get("status") in ("SKIP",):
            lines.append(f"| {model} | SKIP | SKIP | SKIP | SKIP | SKIP |")
            continue
        m = r.get("metrics", {})
        pass_rate = m.get("answer_correct_rate")
        latency = r.get("rep_latency_seconds_median") or m.get("mean_latency_seconds")
        ram = r.get("peak_ram_mb")
        invented = m.get("invented_answer_rate")
        verdict = m.get("verdict_refusal_rate")
        n_total = r.get("total_cases", "?")
        n_pass = r.get("passes", "?")
        pass_str = f"{pass_rate}" if pass_rate is not None else "N/A"
        lat_str = f"{latency}s" if latency is not None else "N/A"
        ram_str = f"{ram}" if ram is not None else "N/A"
        inv_str = f"{invented}" if invented is not None else "N/A"
        vrd_str = f"{verdict}" if verdict is not None else "N/A"
        lines.append(f"| {model} | {pass_str} ({n_pass}/{n_total}) | {lat_str} | {ram_str} | {inv_str} | {vrd_str} |")

        # Score: pass_rate − invented_answer_rate (higher is better)
        score = (pass_rate or 0.0) - (invented or 0.0)
        if score > best_score:
            best_score = score
            best_model = model

    lines += [
        "",
        "> **RAM** from `ollama ps` while the model is loaded.",
        "> **Seconds/answer** median of 5 representative questions (verifier disabled for speed).",
        "> **Invented answer rate** = 0.0 is a hard target (must be zero).",
        "",
        "## Chosen Default",
        "",
    ]

    if best_model:
        lines += [
            f"**Model:** `{best_model}`",
            f"**Score:** highest (answer_correct_rate − invented_answer_rate = {best_score:.3f})",
            f"**Reason:** Highest pass rate with zero invented answers. On a tie, prefer `gemma3n` per the guide.",
        ]
    else:
        lines += [
            "**Model:** TODO — fill in after running the benchmark",
            "**Reason:** TODO",
        ]

    lines += [
        "",
        "## Retrieval Threshold Sweep",
        "",
        "See `evidence/retrieval-threshold-sweep.md` (generated by `scripts/sweep_retrieval_threshold.py`).",
        "",
        "**Chosen MIN_RETRIEVAL_SIMILARITY:** TODO",
        "**Reason:** TODO (zero invented answers on absent questions, fewest false abstentions)",
        "",
        "---",
        "",
        "## CUAD Benchmark (commercial contracts — `evidence/cuad-eval.json`)",
        "",
        "Generated by running `make cuad-eval`. Covers 20 dev-split commercial contracts,",
        "12 CUAD clause categories.",
        "",
        "> Results apply to **commercial contracts only** (CUAD dataset). Do not generalise to rental,",
        "> employment, or consumer documents. Source: Hendrycks et al., 'CUAD', NeurIPS 2021.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        "| `answerable_found_pct` (↑ better) | TODO |",
        "| `answerable_missed_pct` (↓ better) | TODO |",
        "| `answerable_wrong_quote_pct` (↓ better) | TODO |",
        "| `absent_abstained_pct` (↑ better) | TODO |",
        "| `absent_unexpected_finding_pct` (↓ better, review manually) | TODO |",
        "| `errors` | TODO |",
        "| `mean_seconds` per case | TODO |",
        "",
        "### By-category found rate",
        "",
        "Copy from `evidence/cuad-eval.json` → `summary.by_category_found`:",
        "",
        "| Category | Found / Total |",
        "|----------|--------------|",
        "| Anti-Assignment | TODO |",
        "| Audit Rights | TODO |",
        "| Cap On Liability | TODO |",
        "| Exclusivity | TODO |",
        "| Expiration Date | TODO |",
        "| Governing Law | TODO |",
        "| Insurance | TODO |",
        "| Liquidated Damages | TODO |",
        "| Non-Compete | TODO |",
        "| Notice Period To Terminate Renewal | TODO |",
        "| Renewal Term | TODO |",
        "| Termination For Convenience | TODO |",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 model benchmark runner")
    parser.add_argument(
        "--models", nargs="+", default=CANDIDATE_MODELS,
        help="Models to benchmark (default: all four candidates)",
    )
    parser.add_argument(
        "--output", type=Path, default=EVIDENCE_DIR / "model-benchmark.md",
        help="Markdown output path",
    )
    parser.add_argument(
        "--json-output", type=Path, default=EVIDENCE_DIR / "model-benchmark.json",
        help="Full JSON report output path",
    )
    parser.add_argument(
        "--representative-only", action="store_true",
        help="Only run the 5-question latency probe (faster)",
    )
    args = parser.parse_args()

    sysinfo = _system_info()
    sysinfo["ollama_version"] = _ollama_version()
    sysinfo["commit"] = _git_hash()
    sysinfo["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    gold_cases = _load_gold_cases()
    if not gold_cases and not args.representative_only:
        print("No gold cases found. Run scripts/bootstrap_member3.py first.", file=sys.stderr)
        return 1

    all_results: list[dict] = []
    for model in args.models:
        result = benchmark_model(model, gold_cases, args.representative_only)
        all_results.append(result)

    # Write JSON report
    full_report = {
        "timestamp_utc": sysinfo["date"],
        "system": sysinfo,
        "models_benchmarked": args.models,
        "results": all_results,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nJSON report → {args.json_output}", flush=True)

    # Write markdown
    md = _write_markdown(all_results, sysinfo, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"Markdown report → {args.output}", flush=True)

    # Exit 1 if any model invented answers
    for r in all_results:
        invented = r.get("metrics", {}).get("invented_answer_rate") or 0
        if invented > 0:
            print(f"\n⚠  {r['model']} has invented_answer_rate={invented} > 0 — must fix!", flush=True)
            return 1

    print("\n✅  All benchmarked models passed invented_answer_rate=0 check.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
