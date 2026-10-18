# Hand Review — Live-Model Findings on Unseen Documents

> **Owner:** Member 3 (run) + Team (review)
> **Requirement:** At least 20 findings from documents the team did NOT write,
> reviewed by hand to verify quote grounding (rubric criterion L3 / S8).

> **How to regenerate:** `make hand-review`  
> Then open each source document in `samples/` and fill in the Verdict column.

---

## Run Details

| Field | Value |
|-------|-------|
| Date | TODO — fill after running `make hand-review` |
| Commit hash | TODO — `git rev-parse --short HEAD` |
| Mode | question |
| Total findings collected | TODO |
| Errors | TODO |

---

## Instructions

For each finding below:
1. Open the source document in `samples/`.
2. Locate the verbatim quote in the original text.
3. Read the surrounding context (1–2 sentences before and after).
4. Fill in the **Quote present?** column: ✅ if you can find the exact text, ❌ otherwise.
5. Fill in the **Verdict** column: `supports` / `partly supports` / `does not support`.
6. Fill in the **Notes** column with any observation.

---

## Findings Table

> Run `make hand-review` to auto-populate this table from live model output.

| # | Document | Category | Question (first 60 chars) | Claim (first 80 chars) | Quote (first 80 chars) | Pg | Similarity | Support badges | Quote present? | Verdict | Notes |
|---|----------|----------|--------------------------|----------------------|----------------------|----|-----------|----------------|---------------|---------|-------|
| 1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 3 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 5 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 6 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 7 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 9 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 10 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 11 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 12 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 13 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 14 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 15 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 16 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 17 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 18 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 19 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |
| 20 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | ✅ / ❌ | supports / partly / does not | |

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

> TODO — one paragraph. Example: '18 of 20 quotes directly support the claim. The 2 partly-supporting
> cases involved the model paraphrasing a financial figure as a range. No "does not support" cases
> were found. The verbatim-quote gate is working as intended on documents outside the team corpus.'

> **Important:** These are documents the team did not author. A high 'supports' rate here provides
> stronger evidence of grounding than results on team-authored samples.
