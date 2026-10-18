# Contributing

Contributions that improve grounded answers, uncertainty handling, privacy, test coverage, or usability are welcome.

## Development setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
pytest -q
python scripts/static_privacy_audit.py
python -m compileall -q app.py legal_triage scripts tests
```

Model-backed tests need a local Ollama instance and locally downloaded models. Do not make pull requests depend on a hosted LLM, remote embeddings, cloud OCR, or API credentials.

## Rules for changes

- Keep private-document processing local and preserve the loopback-only runtime guard.
- Do not add telemetry or logging that records prompts, extracted text, embeddings, model output, filenames from user uploads, or source passages.
- Any new factual finding in the UI must carry a known source ID and a quote verified against the extracted text. Explain limitations of any new validation.
- Add deterministic tests for parser/guardrail changes and add a documented adversarial test for model-facing changes.
- Do not commit user contracts, uploaded PDFs, model weights, environment files, API keys, or evaluation documents that you do not have permission to redistribute.
- Keep scope honest: no OCR, legal-database, jurisdiction-specific legal-research, or legal-advice claim unless the functionality and its tests are truly added.

## Pull-request checklist

- [ ] `pytest -q` passes.
- [ ] `python scripts/static_privacy_audit.py` passes.
- [ ] `python -m compileall -q app.py legal_triage scripts tests` passes.
- [ ] Relevant manual/adversarial tests are documented.
- [ ] README, dependencies, threat model, and limitations are updated.
