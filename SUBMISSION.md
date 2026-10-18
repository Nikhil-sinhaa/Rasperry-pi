# SUBMISSION.md — PS2 Deliverable Map

This file maps each required deliverable (S1–S10) and gate (G1–G5, L1–L4) to the exact file or link.
Check every link opens in a private window before submitting. Tag `v1.0` after verifying all links.

---

## Deliverables

| ID | Deliverable | File / Link |
|----|-------------|-------------|
| S1 | Public repo with licence | [GitHub repo](https://github.com/Nirbhay404/Rasperry-pi) · [LICENSE](LICENSE) |
| S2 | README: overview, setup, run, configuration, troubleshooting | [README.md](README.md) |
| S3 | Pinned requirements, tool versions, OS and hardware tested, model sizes | [requirements.txt](requirements.txt) · [evidence/system-info-template.md](evidence/system-info-template.md) |
| S4 | Architecture diagram and explanation | [README.md#architecture](README.md) |
| S5 | Dataset sources, licences, preparation | [docs/datasets.md](docs/datasets.md) |
| S6 | How Gemma is used | [README.md](README.md) · [evidence/model-benchmark.md](evidence/model-benchmark.md) |
| S7 | Working demo recording (~5 minutes) | TODO: add link after recording |
| S8 | Test results and known limitations | [evidence/model-evaluation.txt](evidence/model-evaluation.txt) · [docs/known-limitations.md](docs/known-limitations.md) |
| S9 | Steps to reproduce the demo | [README.md#evaluation-plan](README.md) · see §Demo Reproduction below |
| S10 | Statement of main technical contribution | See §Main Technical Contribution below |

---

## Gates

| Gate | What judges check | Owner | Evidence |
|------|-------------------|-------|----------|
| G1 | Public repo with a licence file | 4 | [GitHub](https://github.com/Nirbhay404/Rasperry-pi) · [LICENSE](LICENSE) |
| G2 | A judge can run the demo with no paid service or hidden credential | 4, 2 | [evidence/fresh-install.md](evidence/fresh-install.md) |
| G3 | Demo shows the real system, not mock-ups | 4 | Demo video link (TODO) |
| G4 | Dataset sources and licences disclosed; no private data committed | 3 | [docs/datasets.md](docs/datasets.md) |
| G5 | No keys, tokens or passwords in the repo | 4 | Secret-scan output (TODO: run gitleaks) |
| L1 | Gemma runs locally | 2, 1 | [docs/model-acquisition.md](docs/model-acquisition.md) · [evidence/model-benchmark.md](evidence/model-benchmark.md) |
| L2 | No private content leaves the device | 2 | [evidence/network-monitor.txt](evidence/network-monitor.txt) |
| L3 | No systematic fabrication of clauses or citations | 3, 1 | [evidence/model-evaluation.txt](evidence/model-evaluation.txt) · [evidence/citation-review.md](evidence/citation-review.md) |
| L4 | Not presented as legal advice; safe escalation shown | 1, 4 | Refusal examples in demo · UI disclaimer text |

---

## Demo Reproduction (S9)

1. Clone the repo on a clean machine (no previous install).
2. Follow `README.md` §Installation — configure Ollama local-only, pull models, install deps, copy `.env.example → .env`.
3. Run `python scripts/check_local_setup.py` and `pytest -q` — both must pass.
4. Disconnect Wi-Fi. Start `tcpdump` capture.
5. Run `streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false`.
6. Upload `samples/demo_rental_agreement.txt`.
7. Click **Run whole-document triage** → inspect findings, citations, key-facts card.
8. Ask "What is the tenant's insurance policy number?" → confirm abstention.
9. Ask "Is this agreement enforceable?" → confirm refusal with next-step.
10. Upload `samples/adversarial_injection_test.txt` → confirm injection is ignored.
11. Stop `tcpdump`. Verify no non-loopback outbound traffic.

---

## Main Technical Contribution (S10)

The project delivers a **fully local, evidence-first legal document triage agent** built on open-source
components (Streamlit, Ollama, Gemma, pypdf, NumPy). Its distinguishing properties:

- **Zero data egress**: all text, embeddings, and model calls stay on-device via a validated loopback Ollama
  endpoint; proxy environment variables are ignored by design.
- **Fail-closed citation validation**: every model finding must carry a source ID and a verbatim contiguous
  quote; findings that fail either check are silently dropped before display.
- **Deterministic legal-boundary guard**: a regex + Gemma disambiguation pass prevents the model from
  returning legal verdicts (enforceability, validity, outcome predictions) without any probabilistic
  confidence threshold.
- **Per-topic retrieval for long documents**: `retrieve_triage_per_topic()` runs one embedding search per
  semantic clause category so important sections are not crowded out in long contracts.
- **CUAD-backed smoke metrics**: a reproducible benchmark against real commercial contracts (CUAD v1,
  CC BY 4.0) gives a quantitative baseline for clause-finding recall and false-abstention rate.

The combination of local privacy, grounding enforcement, and quantified evaluation is the core technical
contribution over a naïve LLM chat wrapper.
