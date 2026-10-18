#!/usr/bin/env python3
"""Score the PS2 agent against CUAD gold cases (see cuad_to_gold.py).

Per case:
  answerable -> PASS if the agent did not abstain AND at least one validated quote matches an
                expert span; 'missed' if it abstained; 'wrong_quote' if it cited something else.
  absent     -> PASS if it abstained / returned no findings; otherwise 'unexpected_finding'
                (must be hand-reviewed: the model may legitimately cite related text, so this is
                not automatically a hallucination, but a high rate needs investigation).
Reports are honest smoke metrics on COMMERCIAL contracts, not proof for consumer documents.

  python scripts/score_cuad.py --gold eval/gold/cuad_gold.json --out evidence/cuad-eval
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


def span_matches(quote: str, span: str) -> bool:
    """True if a quote and an expert span substantially overlap (containment or token overlap)."""
    q, s = _tokens(quote), _tokens(span)
    if not q or not s:
        return False
    qs, ss = " ".join(q), " ".join(s)
    if qs in ss or ss in qs:
        return True
    common = sum((Counter(q) & Counter(s)).values())
    return len(s) >= 3 and (common / len(s) >= 0.7 or common / len(q) >= 0.7)


def quotes_of(result: dict) -> list[str]:
    return [e.get("quote", "") for f in result.get("findings", []) for e in f.get("evidence", [])]


def score_case(case: dict, result: dict) -> str:
    abstained = bool(result.get("abstained")) or not result.get("findings")
    if case["type"] == "absent":
        return "PASS" if abstained else "unexpected_finding"
    if abstained:
        return "missed"
    hit = any(span_matches(q, s) for q in quotes_of(result) for s in case["gold_spans"])
    return "PASS" if hit else "wrong_quote"


def evaluate(cases, contracts_dir: Path, analyze_fn, build_index_fn, runtime, limit=None):
    rows, indexes = [], {}
    for case in cases[:limit]:
        fname = case["contract_file"]
        if fname not in indexes:
            indexes[fname] = build_index_fn([(fname, (contracts_dir / fname).read_bytes())], runtime)
        t0 = time.time()
        try:
            result = analyze_fn(case["question"], indexes[fname], runtime, mode="question")
            status = score_case(case, result)
            n_find = len(result.get("findings", []))
        except Exception as exc:  # record, never hide
            status, n_find = f"ERROR:{type(exc).__name__}", 0
        rows.append({"id": case["id"], "category": case["category"], "type": case["type"],
                     "status": status, "findings": n_find, "seconds": round(time.time() - t0, 1)})
    return rows


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] == "answerable"]
    ab = [r for r in rows if r["type"] == "absent"]
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    return {
        "cases": len(rows),
        "answerable": len(ans),
        "answerable_found_pct": pct(sum(r["status"] == "PASS" for r in ans), len(ans)),
        "answerable_missed_pct": pct(sum(r["status"] == "missed" for r in ans), len(ans)),
        "answerable_wrong_quote_pct": pct(sum(r["status"] == "wrong_quote" for r in ans), len(ans)),
        "absent": len(ab),
        "absent_abstained_pct": pct(sum(r["status"] == "PASS" for r in ab), len(ab)),
        "absent_unexpected_finding_pct": pct(sum(r["status"] == "unexpected_finding" for r in ab), len(ab)),
        "errors": sum(r["status"].startswith("ERROR") for r in rows),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1) if rows else None,
        "by_category_found": {
            cat: f"{sum(r['status']=='PASS' for r in ans if r['category']==cat)}/{sum(r['category']==cat for r in ans)}"
            for cat in sorted({r['category'] for r in ans})},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("eval/gold/cuad_gold.json"))
    ap.add_argument("--contracts-dir", type=Path, default=Path("data/cuad_eval/contracts"))
    ap.add_argument("--out", type=Path, default=Path("evidence/cuad-eval"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    from legal_triage.agent import analyze
    from legal_triage.local_runtime import LocalOllama
    from legal_triage.retrieval import build_index
    runtime = LocalOllama()
    runtime.verify_models()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    rows = evaluate(gold["cases"], args.contracts_dir, analyze, build_index, runtime, args.limit)
    report = {"chat_model": runtime.chat_model, "embedding_model": runtime.embedding_model,
              "gold_source": gold["source"], "subset": gold["subset"],
              "summary": summarize(rows), "rows": rows,
              "caveat": "Commercial contracts (CUAD); smoke metrics, not legal correctness."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
