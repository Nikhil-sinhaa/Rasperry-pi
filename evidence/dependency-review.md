# Dependency Review
<!-- Member 2 evidence – fill after running pip-audit and manual grep -->

> **Instructions:**
> 1. While online (setup phase only), run `pip-audit -r requirements.txt` and paste below.
> 2. For each direct dependency, note what network access it can make.
> 3. Grep for telemetry patterns: `grep -r "telemetry\|analytics\|sentry\|amplitude\|mixpanel\|beacon" ~/.local/lib/python*/site-packages/{streamlit,requests,numpy,pypdf,dotenv} 2>/dev/null | head -60`
> 4. Never write PASSED for a check you did not run.

| Field        | Value |
|--------------|-------|
| Date/time    | _(fill in)_ |
| Commit hash  | _(fill in)_ |
| Machine / OS | _(fill in)_ |
| pip-audit version | _(fill in: `pip-audit --version`)_ |

---

## Direct dependencies (from requirements.txt)

| Package | Version | Can call out to network? | Telemetry / analytics? | Notes |
|---------|---------|--------------------------|------------------------|-------|
| streamlit | 1.50.0 | Yes – dev server, asset CDN at startup; **disabled** via `gatherUsageStats = false` in `.streamlit/config.toml` and `address = "127.0.0.1"`. Streamlit may check for updates; disable with `STREAMLIT_SERVER_HEADLESS=true` or the config key `client.showErrorDetails`. | `gatherUsageStats` set to `false` in config. | Binds to 127.0.0.1 only per `.streamlit/config.toml`. |
| requests | 2.32.5 | Yes – HTTP client. Used exclusively via `LocalOllama` which has `trust_env = False` and is constrained to loopback URLs by `config.py`. No other code imports requests directly. | None known. | |
| numpy | 2.3.5 | No – pure computation. | None. | |
| pypdf | 5.9.0 | No – PDF parsing only; uses `BytesIO` in-memory. | None. | |
| python-dotenv | 1.1.1 | No – reads `.env` files only. | None. | |

## Transitive dependencies

Run `pip show streamlit requests numpy pypdf python-dotenv | grep Requires` and list here.

```
(paste output here)
```

## pip-audit output

```
(paste full pip-audit output here – run while online during setup)
```

**Known CVEs affecting this project:** _(Yes / No – list if Yes)_

## Telemetry grep

```
(paste grep output here)
```

**Telemetry patterns found:** _(Yes / No – explain if Yes)_

## Sub-dependency capabilities

> List any sub-dependency that has the *capability* to make outbound network connections
> and explain why it does not do so in this application.

| Sub-package | Can phone home? | Reason it does not in this app |
|-------------|-----------------|-------------------------------|
| urllib3 (via requests) | Yes (HTTP) | Only called by `LocalOllama`; `trust_env=False`; loopback-only URL validated at config time |
| certifi (via requests) | Yes (reads cert bundle) | Cert bundle read at import; no outbound connection made | 
| charset-normalizer | No | Encoding detection only |
| idna | No | IDNA encoding only |

## Result

- [ ] No known CVEs in direct dependencies (or CVEs documented and assessed as non-exploitable in this context)
- [ ] No telemetry calls found in direct dependencies
- [ ] All network-capable sub-dependencies explained above

_Completed by:_ _(name)_  
_Date:_ _(date)_
