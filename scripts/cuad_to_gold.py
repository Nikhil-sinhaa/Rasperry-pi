#!/usr/bin/env python3
"""Convert CUAD (SQuAD-style CUADv1.json / test.json) into gold eval cases for the PS2 agent.

CUAD: Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review",
NeurIPS 2021 Datasets and Benchmarks. Data licence: CC BY 4.0 (cite the paper; keep the
generated data/ folder out of git).  CUAD = COMMERCIAL contracts: results do not prove accuracy
on rental / employment / consumer documents.

Usage:
  python scripts/cuad_to_gold.py --src path/to/test.json --subset dev --contracts 20
Writes:
  data/cuad_eval/contracts/<slug>.txt   (one plain-text contract each)
  eval/gold/cuad_gold.json              (cases: question, type, expected spans)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# CUAD category -> plain-language question a non-lawyer would ask.
QUESTIONS = {
    "Governing Law": "Which law or jurisdiction governs this agreement?",
    "Expiration Date": "When does this agreement expire or end?",
    "Renewal Term": "Does the agreement renew automatically, and for how long?",
    "Termination For Convenience": "Can a party end the agreement without giving a reason?",
    "Notice Period To Terminate Renewal": "How much notice is needed to stop the agreement from renewing?",
    "Anti-Assignment": "Can a party transfer this agreement to someone else without consent?",
    "Cap On Liability": "Is there a limit on how much a party can be held liable for?",
    "Liquidated Damages": "Does the contract set a fixed amount payable as damages or a penalty?",
    "Insurance": "Does the contract require a party to carry insurance?",
    "Audit Rights": "Can one party inspect or audit the other party's records?",
    "Non-Compete": "Is any party restricted from competing with the other?",
    "Exclusivity": "Does the contract make the arrangement exclusive?",
}


def slugify(title: str, index: int) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:70] or "contract"
    return f"{index:03d}_{base}"


def build(src: Path, subset: str, n_contracts: int, max_len: int, out_root: Path, seed: int,
          answerable_per: int = 5, absent_per: int = 3) -> dict:
    data = json.loads(src.read_text(encoding="utf-8"))["data"]
    data = sorted(data, key=lambda c: c["title"])
    if subset == "dev":
        pool = data[0::2]
    elif subset == "final":
        pool = data[1::2]
    else:
        pool = data
    pool = [c for c in pool if len(c["paragraphs"][0]["context"]) <= max_len]
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen = sorted(pool[:n_contracts], key=lambda c: c["title"])

    contracts_dir = out_root / "data" / "cuad_eval" / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for idx, contract in enumerate(chosen, start=1):
        para = contract["paragraphs"][0]
        fname = slugify(contract["title"], idx) + ".txt"
        (contracts_dir / fname).write_text(para["context"], encoding="utf-8")
        by_cat = {}
        for qa in para["qas"]:
            cat = qa["id"].split("__")[-1]
            if cat in QUESTIONS:
                by_cat[cat] = [a["text"] for a in qa["answers"] if a["text"].strip()]
        present = sorted(c for c, spans in by_cat.items() if spans)
        absent = sorted(c for c, spans in by_cat.items() if not spans)
        rng.shuffle(present)
        rng.shuffle(absent)
        for cat in present[:answerable_per]:
            cases.append({"id": f"{fname}::{cat}", "contract_file": fname, "category": cat,
                          "question": QUESTIONS[cat], "type": "answerable", "gold_spans": by_cat[cat]})
        for cat in absent[:absent_per]:
            cases.append({"id": f"{fname}::{cat}", "contract_file": fname, "category": cat,
                          "question": QUESTIONS[cat], "type": "absent", "gold_spans": []})
    gold = {
        "source": "CUAD v1 (Hendrycks et al., NeurIPS 2021), CC BY 4.0; commercial contracts only",
        "subset": subset, "seed": seed, "contracts": len(chosen), "cases": cases,
    }
    gold_path = out_root / "eval" / "gold" / "cuad_gold.json"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
    return gold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="CUAD test.json or CUADv1.json")
    ap.add_argument("--subset", choices=["dev", "final", "all"], default="dev",
                    help="dev = tune thresholds/prompts; final = report numbers (never tune on it)")
    ap.add_argument("--contracts", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=45000, help="skip very long contracts (slow on laptops)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()
    gold = build(args.src, args.subset, args.contracts, args.max_chars, args.out, args.seed)
    kinds = {}
    for c in gold["cases"]:
        kinds[c["type"]] = kinds.get(c["type"], 0) + 1
    print(f"{gold['contracts']} contracts, {len(gold['cases'])} cases {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
