# Fresh-Install Test Evidence

> **Owner:** Member 4 (gate G2)  
> **Rule:** A teammate who **did not write the code** installs from the README on a clean user account
> or VM. Fix every snag. Save timing and notes here.

---

## Environment

| Field | Value |
|-------|-------|
| Tester (not the author) | TODO |
| Machine | TODO (e.g. MacBook Air M1, 8 GB RAM) |
| OS | TODO |
| Python version | TODO (`python3 --version`) |
| Ollama version | TODO (`ollama --version`) |
| Date | TODO |
| Repo commit | TODO (`git rev-parse --short HEAD`) |
| Starting state | Clean user account / fresh VM / Docker container |

---

## Step-by-Step Timing

| Step | README section | Time (mm:ss) | Result | Notes |
|------|---------------|-------------|--------|-------|
| Clone repo | — | TODO | ✅ / ❌ | |
| Configure Ollama local-only | §1 | TODO | ✅ / ❌ | |
| `ollama pull gemma4:e2b` | §2 | TODO | ✅ / ❌ | |
| `ollama pull nomic-embed-text` | §2 | TODO | ✅ / ❌ | |
| `python3 -m venv .venv && pip install -r requirements.txt` | §3 | TODO | ✅ / ❌ | |
| `cp .env.example .env` | §3 | TODO | ✅ / ❌ | |
| `python scripts/check_local_setup.py` | §4 | TODO | ✅ / ❌ | |
| `pytest -q` | §4 | TODO | ✅ / ❌ | |
| `streamlit run app.py …` | §4 | TODO | ✅ / ❌ | |
| Upload `samples/demo_rental_agreement.txt` | demo | TODO | ✅ / ❌ | |
| Run whole-document triage | demo | TODO | ✅ / ❌ | |
| Ask absent-info question | demo | TODO | ✅ / ❌ | |
| Ask enforceability question | demo | TODO | ✅ / ❌ | |

**Total time from zero to working demo:** TODO

---

## Issues Found and Fixes

| # | Step | Issue | Fix applied | Commit |
|---|------|-------|-------------|--------|
| 1 | — | TODO | TODO | — |

---

## Verdict

- [ ] All steps completed without undocumented manual intervention
- [ ] Total time is reasonable (target < 30 minutes excluding model download)
- [ ] No secrets, model weights or private documents were needed
- [ ] G2 gate: **PASS / FAIL**

> TODO – one paragraph summary
