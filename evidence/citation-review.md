# Citation Review Evidence

> **Owner:** Member 3 (run) / Member 4 (copy into README §S8)  
> **Requirement:** Pick 20 random findings, open the original page, and mark each as
> `supports`, `partly supports`, or `does not support`. Save counts here.
> This is the honest grounding figure for criterion L3.

---

## Method

1. Run `make cuad-eval` or `python scripts/run_evals.py` and note the commit hash.
2. Pick 20 findings at random from `evidence/cuad-eval.json` or `evidence/model-evaluation.txt`
   (use `scripts/pick_citation_sample.py` if it exists, otherwise pick manually).
3. For each finding, open the original document page cited and read the quote in context.
4. Mark the finding with one of:
   - **supports** – the quote directly and unambiguously supports the claim.
   - **partly supports** – the quote is relevant but the claim slightly overstates or understates.
   - **does not support** – the quote is present but does not logically support the claim.
5. Record each finding in the table below.

---

## Run Details

| Field | Value |
|-------|-------|
| Date | TODO |
| Commit hash | TODO |
| Source file | TODO (e.g. `evidence/cuad-eval.json`) |
| Reviewer | TODO |

---

## Findings (20 samples)

| # | Finding ID | Category | Claim (first 80 chars) | Quote present? | Verdict | Notes |
|---|-----------|----------|------------------------|---------------|---------|-------|
| 1 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 2 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 3 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 4 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 5 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 6 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 7 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 8 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 9 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 10 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 11 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 12 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 13 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 14 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 15 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 16 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 17 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 18 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 19 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 20 | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |

---

## Summary Counts

| Verdict | Count | % |
|---------|-------|---|
| supports | TODO | TODO% |
| partly supports | TODO | TODO% |
| does not support | TODO | TODO% |
| **Total** | 20 | 100% |

**Quote always present in retrieved chunk:** TODO / 20

---

## Interpretation

> TODO – one paragraph. Example: "18 of 20 quotes directly support the claim; 2 were partly
> supporting (the model's plain-language restatement slightly broadened the scope).
> Zero 'does not support' cases in this sample. The verbatim-quote check is working as intended."

> **Note:** Verbatim quote validation proves the quote exists in the retrieved source; it does not
> automatically prove the interpretation logically follows. This manual review is the honest
> figure for L3 (no systematic fabrication).
