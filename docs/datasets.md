# Dataset Sources, Licences, and Preparation

> **Owner:** Member 3 (content) / Member 4 (gate G4)
> **Gate:** G4 – dataset sources and licences disclosed; no private data committed.

---

## Bundled Sample Files

These files are committed to the repository and used in automated tests and the demo.

| File | Type | Source | Licence | Notes |
|------|------|--------|---------|-------|
| `samples/demo_rental_agreement.txt` | Synthetic fictional rental agreement | Written by project team | MIT (same as repo) | No real parties or addresses |
| `samples/adversarial_injection_test.txt` | Synthetic adversarial injection test | Written by project team | MIT | Contains AI-instruction text in document body for security testing |

These files contain no personal data, real legal documents, or non-redistributable text.

---

## CUAD v1 (Optional Benchmark Dataset)

| Field | Value |
|-------|-------|
| Name | CUAD: Contract Understanding Atticus Dataset |
| Authors | Hendrycks et al., NeurIPS 2021 Datasets and Benchmarks Track |
| Paper | https://arxiv.org/abs/2103.06268 |
| Source | https://www.atticusprojectai.org/cuad |
| Licence | **CC BY 4.0** — attribution required; no restrictions on use or redistribution with attribution |
| Attribution | "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review", Hendrycks et al., 2021 |
| What it contains | 510 commercial contracts from EDGAR (SEC public filings) with 13 000+ expert annotations |
| File in this repo | `scripts/data.zip` (includes `CUADv1.json`, `test.json`, `train_separate_questions.json`) |
| Generated output | `data/cuad_eval/contracts/` — **excluded from git** via `.gitignore` |

> **Important:** CUAD covers commercial contracts only. Results from the CUAD benchmark do not prove
> accuracy on rental, employment, or consumer documents.

### How to use

```bash
make cuad-eval
# Extracts CUADv1.json, generates gold cases, scores the agent, writes evidence/cuad-eval.json
```

See [README.md §E](../README.md) for full instructions.

---

## Potential Future Datasets (Not Currently Used)

The PS2 guide mentions these optional references. **None are committed to this repository.**
Verify their licence and access terms independently before use.

| Dataset | URL | Licence status |
|---------|-----|----------------|
| India Code | https://www.indiacode.nic.in | Bulk-use/redistribution terms not verified in the guide — check before using |
| Open India Law | Various sources | Large mixed-source dataset; some source terms unknown — check before using |

---

## No Private Data Committed

A secret scan was run with `git grep` before each push. No private documents, personal data,
API keys, tokens, or `.env` files are present in the repository. The `.gitignore` explicitly
excludes `.env` and `data/cuad_eval/`.
