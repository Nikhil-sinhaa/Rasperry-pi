# Private Legal Document Triage Agent — ZeroDay OSS Hacktoberfest 2026 PS2

A local-first, evidence-first document assistant for preliminary legal-document understanding. It ingests selectable-text PDFs and UTF-8 text files, builds embeddings with a locally running Ollama embedding model, retrieves passages from an in-memory local vector index, and asks an eligible local Gemma edge model to classify and explain text. The UI only displays model findings that pass source-ID and verbatim-quote checks.

**This is an implementation candidate, not a legal product or a completed privacy audit.** The unit tests cover deterministic code paths. You must run the model-backed evaluation and network-isolation procedure on your own laptop and record the actual results before claiming those checks passed.

## PS2 requirement coverage

The participant guide's mandatory requirements are L1–L10. This project is designed for those requirements as follows:

| PS2 requirement | Implementation | Evidence still required from your machine |
|---|---|---|
| L1: local ingestion of PDF or text | `legal_triage/ingestion.py`; PDF and UTF-8 TXT accepted | Demo upload works on your OS |
| L2: on-device extraction and interpretation | pypdf runs in the app process; embedding and chat requests go only to a validated loopback Ollama endpoint | Offline end-to-end run and network capture |
| L3: plain-language summaries | Whole-document triage action and question mode | Human review of output on unseen documents |
| L4: obligations, rights, deadlines, penalties, termination, concerns | Multi-query local retrieval and structured finding types | Evaluation across more than the bundled fictional sample |
| L5: identifiable supporting passages | Source filename/page plus required verbatim source quote; invalid source IDs/non-verbatim quotes are rejected | Inspect quotes against original pages |
| L6: explicit statement vs. possible concern | Separate `explicit_statement`, `potential_concern`, `ambiguity`, and `jurisdiction_dependency` categories | Review semantic correctness; exact quotes alone cannot prove an interpretation is logically sound |
| L7: ambiguity, insufficient evidence, jurisdiction | Dedicated categories, retrieval threshold, abstention when there is no validated evidence | Calibrate `MIN_RETRIEVAL_SIMILARITY` with eval results |
| L8: limit unsupported conclusions and escalate | Deterministic legal-verdict refusal for common requests, quote validation, professional-review questions | Adversarial tests and human review |
| L9: privacy | Loopback-only endpoint validation, HTTP proxy environment ignored by app client, no hosted API, in-memory index, Streamlit usage statistics disabled | Configure Ollama local-only mode and capture network evidence |
| L10: meaningful Gemma and reusable OSS | Gemma chat model configured by `.env`; dependencies, docs, tests, synthetic sample, MIT licence included | Verify the exact model variant/licence, record model runtime/hardware, publish the repo |

PS2's guide prefers a Gemma 3n-class edge model and also accepts Gemma 4 E2B/E4B edge variants. This starter defaults to `gemma4:e2b`, an accepted variant. If you use `gemma3n:e2b`, change `OLLAMA_MODEL` and pull that model. Confirm the exact variant and licence on its official model card before submission.

## What the app does

- Accepts one or more selectable-text PDFs or UTF-8 `.txt` files.
- Extracts text in memory, preserves filename/page metadata, and reports pages with no extractable text.
- Splits page text into overlapping chunks and asks local Ollama for embeddings.
- Ranks chunks locally using cosine similarity over normalized NumPy vectors. No hosted vector database is used.
- Runs local Gemma to return structured findings, each with a source ID and a contiguous exact quote.
- Checks that source IDs exist and quotes appear verbatim (whitespace-normalized) in their source chunk; a finding without valid evidence is dropped.
- Shows exact quotes and page references next to each finding.
- Includes a whole-document triage mode for obligations, rights, deadlines, fees/penalties, termination wording, ambiguity, and possible concerns.
- Rejects common requests for a definitive legal verdict (for example, enforceability/validity or whether to sign) and shows relevant retrieved excerpts instead.
- Does not save uploaded documents or indexes to disk. “Clear documents, vectors, and chat” drops app session state.

## Deliberate limits

- **No OCR:** image-only/scanned pages have no selectable text; those pages are flagged and can be missed. Do not claim scanned-PDF support.
- **No legal database:** the tool explains uploaded text; it does not look up current statutes, cases, or jurisdiction-specific law.
- **No definitive legal advice:** it cannot determine whether a clause is legal or enforceable, predict a court outcome, or tell the user whether to sign.
- **No proof of semantic correctness:** verbatim quote validation proves a quote exists in the retrieved source, not that the interpretation necessarily follows from it. Review each finding against the original page.
- **Retrieval threshold is a tunable baseline:** cosine scores are model- and corpus-dependent. `MIN_RETRIEVAL_SIMILARITY=0.25` is a starting point, not a universal confidence value. Use the model-backed evaluation to tune it; too high may cause false abstentions, too low may allow irrelevant passages into analysis.
- **In-memory only:** app state is temporary, but RAM is not an encrypted vault. Clear the session after use and do not run this app on an untrusted/shared machine.
- **Localhost only:** this code deliberately blocks non-loopback Ollama addresses. Do not change that privacy guard or expose Streamlit/Ollama over a network.

## Architecture

```text
Browser on the same computer
           |
           v
Streamlit app (127.0.0.1:8501; telemetry disabled)
           |
           +--> PDF/TXT parser (pypdf / UTF-8; memory only)
           |        |
           |        v
           |    page-aware chunker and metadata
           |        |
           |        v
           |    Ollama /api/embed on 127.0.0.1:11434
           |        |
           |        v
           |    normalized NumPy vectors in session memory
           |
           +--> local cosine retrieval / multi-query clause retrieval
                    |
                    v
              locally running Gemma via Ollama /api/chat
                    |
                    v
              JSON finding validation
              - known source ID
              - quote is verbatim in that chunk
              - reject unqualified legal-verdict claims
                    |
                    v
              UI: finding + exact quote + filename + page
```

There are no cloud RAG, hosted embeddings, external search tools, telemetry SDKs, or third-party APIs in the document-processing code. `requests.Session.trust_env = False` prevents the app's HTTP client from inheriting proxy environment variables, and the config validator rejects non-loopback Ollama URLs. The separate Ollama service must also be configured to disable its cloud features; do not assume that setting an app environment variable changes an already-running Ollama daemon.

## Prerequisites

- Linux recommended for the demonstrated setup (other platforms may work but have not been tested here).
- Python 3.11 or 3.12 recommended.
- Ollama installed and able to run the selected local models.
- Enough RAM/VRAM for the exact model/runtime/context. There is no claim that a model fits or runs at a particular speed until you record your own test.
- Model and package downloads require internet during setup. Once installed, ordinary document processing is intended to work offline.

## Installation on Linux

### 1. Configure Ollama local-only mode

For a standard Linux systemd installation, run:

```bash
sudo systemctl edit ollama.service
```

Add this override:

```ini
[Service]
Environment="OLLAMA_NO_CLOUD=1"
Environment="OLLAMA_HOST=127.0.0.1:11434"
```

Then apply it:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

If you start Ollama manually instead, set `OLLAMA_NO_CLOUD=1` and bind it to loopback in the environment used to start `ollama serve`. The environment must be present in the **Ollama server process**, not only in Streamlit. Confirm your Ollama version supports this setting and inspect its logs/configuration. Do not use any model tag labelled `cloud`.

Official Ollama reference: https://docs.ollama.com/faq (see “How do I disable Ollama's cloud features?”). Ollama states locally run prompts/data are not sent to Ollama.com, but this project's submission should still demonstrate the actual machine's network behavior.

### 2. Download the local models (one-time setup, while online)

```bash
ollama pull gemma4:e2b
ollama pull nomic-embed-text
ollama list
```

`gemma4:e2b` is the starter choice accepted by the PS2 guide. To try the guide-preferred Gemma 3n edge family instead:

```bash
ollama pull gemma3n:e2b
```

Then change `.env` to `OLLAMA_MODEL=gemma3n:e2b`. Use only a model tag present in `ollama list`, and do not use a cloud tag. Model quantization and size can change; record `ollama --version`, `ollama list`, and `ollama ps` in your evidence file.

Model/licence references to check before submission:
- Gemma 4 E2B instruction model card: https://huggingface.co/google/gemma-4-E2B-it
- Ollama Gemma 4 tags: https://ollama.com/library/gemma4/tags
- Ollama Gemma 3n tags: https://ollama.com/library/gemma3n/tags
- Ollama local embedding model: https://ollama.com/library/nomic-embed-text

Model terms are separate from this repository's MIT licence. Keep the model's own licence attribution/terms and do not commit model weights to the repository.

### 3. Install Python dependencies

```bash
cd Rasperry-pi
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # installs test tools too
cp .env.example .env
```

If you already have a `.env`, do not overwrite it; edit the model name directly. The `.env` file is ignored by git.

### 4. Check local models, run unit tests, then launch

```bash
python scripts/check_local_setup.py
pytest -q
streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open the local address printed by Streamlit, normally `http://127.0.0.1:8501`. Do not use a public tunnel or bind to `0.0.0.0`. Test the bundled fictional sample at `samples/demo_rental_agreement.txt` first.

## Evaluation plan (run this; don't assume it passed)

### A. Deterministic tests

```bash
pytest -q
```

These test configuration guards, PDF/text ingestion behavior, legal-verdict phrase detection, quote validation, and source-ID rejection. They do not prove that a particular Gemma model answers every question correctly.

### B. Model-backed evaluation

With Ollama running and models installed:

```bash
python scripts/run_evals.py | tee evidence/model-evaluation.txt
```

The runner uses only bundled fictional examples. It checks known facts, a deliberately absent field, a legal-verdict request, and a prompt-injection scenario. Save the output only after actually running it. If relevant questions abstain, review the extracted chunks and tune `MIN_RETRIEVAL_SIMILARITY`; if unsupported questions are answered, do not submit until the cause is understood.

### C. Privacy/network evidence for judges

1. Pull all required models and Python dependencies while online.
2. Verify the Ollama service is local-only and `ollama list` contains local model tags rather than cloud tags.
3. Disconnect Wi-Fi (or place the actual app and Ollama service in a network-disabled environment that still permits their loopback communication).
4. Start the app and run a fresh test from upload through a grounded answer and an absent-information question.
5. For Linux packet evidence, start this in another terminal before the demo:

   ```bash
   sudo tcpdump -i any -nn 'not net 127.0.0.0/8 and not host ::1' | tee evidence/network-monitor.txt
   ```

   Run the demo, then stop capture with `Ctrl+C`. The filter omits loopback traffic and records non-loopback traffic; when Wi-Fi is disabled and no other network is active, it should show no outbound document-processing traffic. Be precise in your report about exactly what was monitored and for how long. Packet capture, static code review, model configuration, and a true offline run together provide stronger evidence than a screenshot alone.

6. Record versions/hardware and fill `evidence/system-info-template.md`. Do not state that Wi-Fi-off/offline or network monitoring passed unless you personally ran it and saved the output.

### D. Manual adversarial checks

Use `tests/ADVERSARIAL_TESTS.md`. Try questions that are not in the document; ask whether a clause is enforceable; ask the model to ignore the source; and upload the bundled malicious-instruction sample. Inspect every claim and quote. If your run produces an unsupported finding, document it as a failure and improve/retest before submission.

### E. CUAD benchmark (real commercial contracts)

`scripts/data.zip` contains CUAD v1 (`CUADv1.json`), a CC BY 4.0 dataset of annotated commercial contracts
(Hendrycks et al., NeurIPS 2021). The two-step pipeline below converts it into gold eval cases and scores the
local agent against them. Results land in `evidence/cuad-eval.json` and are the numbers to fill into
`evidence/model-benchmark.md`.

> **Important:** CUAD covers commercial contracts only. A good score does *not* prove accuracy on rental,
> employment, or consumer documents. Always cite the CUAD paper and keep the `data/cuad_eval/` folder out of
> git (it is already `.gitignore`d).

**Step 1 — generate gold cases** (runs once; repeatable with the same `--seed`):

```bash
unzip -o -d /tmp/cuad_extract scripts/data.zip CUADv1.json
python scripts/cuad_to_gold.py \
  --src /tmp/cuad_extract/CUADv1.json \
  --subset dev --contracts 20 --out .
rm -rf /tmp/cuad_extract
```

This writes:
- `data/cuad_eval/contracts/<slug>.txt` — plain-text contracts (excluded from git)
- `eval/gold/cuad_gold.json` — cases with expert-annotated gold spans

**Step 2 — run the scorer** (requires Ollama running with models loaded):

```bash
python scripts/score_cuad.py \
  --gold eval/gold/cuad_gold.json \
  --contracts-dir data/cuad_eval/contracts \
  --out evidence/cuad-eval
```

Or use the combined `make` shortcut:

```bash
make cuad-eval
```

The scorer prints a JSON summary to stdout and writes `evidence/cuad-eval.json`. Key metrics:

| Metric | Meaning |
|---|---|
| `answerable_found_pct` | % of clauses that exist where the agent found and cited a matching span |
| `answerable_missed_pct` | % where the agent abstained even though a clause was present |
| `absent_abstained_pct` | % of absent clauses where the agent correctly returned no findings |
| `absent_unexpected_finding_pct` | % where the agent cited something; needs manual review (not automatically a hallucination) |

Copy the summary values into `evidence/model-benchmark.md`. Tune `MIN_RETRIEVAL_SIMILARITY` until
`absent_abstained_pct` is high and `answerable_missed_pct` is acceptably low, then re-run to confirm.

## Submission checklist mapped to the participant guide

- [ ] Public repository, MIT or other compatible open-source licence, no secrets.
- [ ] README, pinned dependencies, OS/Python/Ollama versions, model tag/quantization, model licence, and actual RAM/VRAM/speed recorded.
- [ ] Architecture and local-processing explanation included here.
- [ ] Dataset/sample source and licence disclosed. The bundled demo files are synthetic and do not require downloading or redistributing a dataset. If you add CUAD/India Code/Open India Law data, independently verify licence/access terms and cite the source; none is mandatory for PS2.
- [ ] Unit tests pass and model-backed evaluation results are included from a real run.
- [ ] Sample outputs show exact citations and an unsupported-question abstention.
- [ ] Privacy evidence: offline run or network-disabled run plus packet/connection-monitoring output, and explanation of remaining local connections.
- [ ] Working demo recording, recommended length up to around five minutes, shows real app behavior and limitations.
- [ ] Scope and known limitations are stated honestly; do not call output legal advice.

## Dataset note

PS2 does not require a dataset. Use synthetic/consented documents for a safe demo. Optional references from the guide include CUAD (commercial contracts; CC BY 4.0 with attribution), India Code (read terms because bulk-use/redistribution terms were not verified in the guide), and Open India Law (large, mixed-source dataset with some source terms unknown). Do not put private user documents or non-redistributable data in a public repository.

## Troubleshooting

- **Model missing:** `ollama list`; then `ollama pull gemma4:e2b` and `ollama pull nomic-embed-text` while online.
- **Cannot reach local Ollama:** check `systemctl status ollama`, `ollama --version`, and that the server binds to `127.0.0.1:11434`.
- **Loopback privacy guard error:** do not point the app to a remote IP or hosted service. Restore `http://127.0.0.1:11434`.
- **Out of memory/slow:** consider `gemma4:e2b`, lower `OLLAMA_NUM_CTX` in `.env`, and restart the app. Record the change in performance evidence.
- **Unsupported question abstains:** check extraction, inspect “All retrieved passages”, and tune threshold against relevant and irrelevant evaluation prompts. Do not lower it merely to make a demo answer.
- **Scanned PDF:** this version reports pages with no text. OCR is not implemented. Convert/extract text locally, then use a local `.txt` file; do not upload the legal PDF to a cloud OCR service.
- **Citation dropped:** the model's quote was not long enough or was not verbatim in the retrieved chunk, or the source ID was invalid. This is an intentional fail-closed behavior; retry a narrower question and inspect the original source.


## Maintainer and review files

- `docs/privacy-threat-model.md` explains data flows, controls, remaining threats, and a reproducible evidence protocol.
- `CONTRIBUTING.md` documents privacy-preserving contribution rules and required checks.
- `SECURITY.md` describes safe vulnerability reporting without sensitive legal documents.
- `.github/workflows/tests.yml` runs deterministic tests, static privacy checks, and compilation on Python 3.11.
