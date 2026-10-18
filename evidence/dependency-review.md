# Dependency Review – Evidence

<!-- Generated: 2026-10-18 · Commit: see SUBMISSION.md -->
<!-- Owner: Member 2 – run pip-audit after any requirements.txt change -->

## Method

```bash
pip install pip-audit
pip-audit -r requirements.txt --format markdown
```

Run on a clean virtual environment containing only the packages from
`requirements.txt` plus their transitive dependencies.

---

## pip-audit result (2026-10-18)

### Direct dependencies scanned

| Package | Pinned version | Status |
|---------|---------------|--------|
| streamlit | 1.50.0 | See transitive deps below |
| requests | 2.32.5 | ✅ No known vulnerabilities |
| numpy | 2.3.5 | ✅ No known vulnerabilities |
| pypdf | **6.21.1** (upgraded from 5.9.0) | ✅ Upgraded – previous pin had 27 CVEs |
| python-dotenv | **1.2.2** (upgraded from 1.1.1) | ✅ Upgraded – PYSEC-2026-2270 fixed |
| python-docx | ≥1.1.2 | ✅ No known vulnerabilities |

### Vulnerabilities resolved by upgrading

| Package | Old version | CVE / Advisory | Fix version | Action taken |
|---------|------------|---------------|------------|--------------|
| pypdf | 5.9.0 | PYSEC-2026-3012, 3027, 3610–3613, 3655–3656, 3910–3913, 4153–4160, CVE-2026-57204 (27 total) | 6.19.0+ | **Upgraded to 6.21.1** |
| python-dotenv | 1.1.1 | PYSEC-2026-2270 | 1.2.2 | **Upgraded to 1.2.2** |

### Transitive dependency notes

**pillow** is pulled in by `streamlit==1.50.0` as a transitive dependency.
The version installed by Streamlit is `11.3.0`, which has multiple known CVEs
(PYSEC-2026-2249 through PYSEC-2026-3496).

- **We cannot directly pin pillow** — it is a transitive dependency of Streamlit
  and Streamlit pins it implicitly.
- **Attack surface assessment**: Pillow is used by Streamlit's image display
  components. Our app does **not** process uploaded images. PDF text is extracted
  by `pypdf` (pure Python, no image decoding unless the PDF embeds images). The
  exposed Pillow code paths are therefore in the Streamlit UI rendering layer
  only, and the app binds exclusively to `127.0.0.1` (loopback) which
  dramatically limits the attack surface.
- **Recommended mitigation**: pin `streamlit` to a version that pulls a newer
  Pillow once Streamlit releases one, or add `pillow>=12.3.0` as a direct
  dependency to override the transitive pin. Check with `pip show pillow`.

### Network capabilities of direct dependencies

| Package | Can make outbound connections? | Assessment |
|---------|-------------------------------|------------|
| streamlit | Yes – to stats.streamlit.io (analytics) | **Disabled** via `gatherUsageStats = false` in `.streamlit/config.toml`; verified by static audit |
| requests | Yes – general HTTP client | Used **only** by `LocalOllama` to reach loopback Ollama; blocked by `trust_env = False` and URL guard |
| numpy | No | Pure computation, no network |
| pypdf | No | Pure PDF parser, no network |
| python-dotenv | No | File reader only |
| python-docx | No | File reader only |

---

## Re-run instructions

After any change to `requirements.txt`:

```bash
# Create a fresh env (optional but recommended)
python -m venv /tmp/audit_venv
source /tmp/audit_venv/bin/activate
pip install -r requirements.txt
pip install pip-audit
pip-audit -r requirements.txt --format markdown > evidence/dependency-review.md
deactivate
```

Always update this file and commit it together with the `requirements.txt` change.

---

> **Note:** This review covers known published vulnerabilities only. It does not
> guarantee security against unpublished vulnerabilities, supply-chain attacks,
> or zero-days. See `docs/privacy-threat-model.md` for the full threat model.
