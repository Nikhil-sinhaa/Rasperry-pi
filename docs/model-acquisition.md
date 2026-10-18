# Model Acquisition Instructions
<!-- Member 2 deliverable – docs/model-acquisition.md -->
<!-- Maps to submission item "Extra: local-processing evidence, model acquisition instructions" -->

This document covers how to obtain the Gemma model and the embedding model required
to run the PS2 Legal Triage application **fully offline**.  It separates the
**setup phase** (requires internet) from the **processing phase** (must be offline).

---

## Separation of phases

| Phase | Network required? | What happens |
|-------|------------------|--------------|
| **Setup** (once, on first install) | **Yes** | Pull model weights from Ollama registry |
| **Processing** (every run) | **No** | All inference is local; no cloud call is made |

> The application's `config.py` enforces a loopback-only Ollama URL and rejects cloud model
> tags at startup.  The `local_runtime.py` module adds a socket-level egress guard so the
> Python process cannot open connections to non-loopback hosts at runtime.

---

## Prerequisites

- **Ollama** installed: <https://ollama.com/download> (desktop app or CLI).
  - Linux one-liner: `curl -fsSL https://ollama.com/install.sh | sh`
  - Windows/macOS: use the installer from the link above.
- A machine with enough RAM/VRAM to hold the chosen model (see table below).
- Python ≥ 3.11 with the project's `requirements.txt` installed.

---

## Step 1 – Configure Ollama for offline use (REQUIRED before pulling models)

Set the following environment variables **in the shell that starts the Ollama server**,
not just in the app shell:

```bash
# Prevent Ollama from contacting cloud endpoints
export OLLAMA_NO_CLOUD=1

# Bind the Ollama API to loopback only (default port 11434)
export OLLAMA_HOST=127.0.0.1:11434

# Disable debug logging (reduces noise and avoids inadvertent content logging)
export OLLAMA_DEBUG=0
```

On **Windows** (PowerShell):
```powershell
$env:OLLAMA_NO_CLOUD = "1"
$env:OLLAMA_HOST     = "127.0.0.1:11434"
$env:OLLAMA_DEBUG    = "0"
```

Verify that Ollama listens on loopback only after starting:
```bash
# Linux
ss -tlnp | grep 11434
# Expected: LISTEN 127.0.0.1:11434

# macOS
lsof -i :11434 | grep LISTEN

# Windows PowerShell
netstat -an | Select-String "11434"
```

> **Ollama desktop app update checks:** The Ollama desktop application may check for
> updates at launch.  This carries no document content, but it does make an outbound
> connection.  Either disable auto-update in the Ollama settings, start Ollama from the
> CLI with the environment variables above (which suppresses cloud features), or document
> this known connection explicitly in your evidence.

---

## Step 2 – Pull the chat model (online, setup phase only)

The application defaults to `gemma4:e4b`.  The guide also supports the Gemma 3n class
(preferred) and Gemma 4 E4B.  Pull whichever fits your hardware:

```bash
# Preferred: Gemma 3n E2B (smallest, ~2 B effective parameters)
ollama pull gemma3n:e2b

# Alternative: Gemma 3n E4B (~4 B effective parameters, more capable)
ollama pull gemma3n:e4b

# Alternative: Gemma 4 E2B
ollama pull gemma4:e2b

# Alternative: Gemma 4 E4B (largest, requires more RAM)
ollama pull gemma4:e4b
```

### Licence

Gemma models are distributed under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).
You must read and accept these terms before using any Gemma model.
The `ollama pull` command will prompt you to accept the licence on first pull.

### Expected sizes and RAM requirements

| Tag | Approx. download | Min. system RAM (no GPU) | Min. VRAM (GPU) |
|-----|-----------------|--------------------------|-----------------|
| `gemma3n:e2b` | ~1.5 GB | 8 GB | 4 GB |
| `gemma3n:e4b` | ~2.5 GB | 12 GB | 6 GB |
| `gemma4:e2b`  | ~1.5 GB | 8 GB | 4 GB |
| `gemma4:e4b`  | ~2.5 GB | 12 GB | 6 GB |

> These are approximate figures.  Actual usage depends on quantization and context window
> (`OLLAMA_NUM_CTX`).  Record exact figures from `ollama ps` after a model call in
> `evidence/system-info-template.md`.

### Verify the digest

After pulling, record the exact digest to pin the model version:
```bash
ollama show gemma3n:e2b --modelfile | grep -i digest
# or
ollama list
```
Paste the digest into `evidence/model-benchmark.md`.

---

## Step 3 – Pull the embedding model (online, setup phase only)

```bash
ollama pull nomic-embed-text
```

### Licence

`nomic-embed-text` is released under
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
No special terms to accept; the licence allows commercial and non-commercial use with
attribution.

### Expected size

~274 MB.  Requires ~1 GB RAM to run.

---

## Step 4 – Confirm no cloud tags

```bash
ollama list
```

Ensure no model tag ends in `:cloud` or contains `-cloud-`.  The application's
`config.py` rejects such tags at startup, but a visual check is required for the
evidence record.

---

## Step 5 – Set the model environment variables for the app

In the shell that runs `streamlit run app.py`:

```bash
# Use gemma3n:e2b (replace with your chosen tag)
export OLLAMA_MODEL=gemma3n:e2b
export OLLAMA_EMBED_MODEL=nomic-embed-text

# Optional tuning
export OLLAMA_NUM_CTX=8192     # context window; increase for long documents if RAM allows
export OLLAMA_NUM_PREDICT=1600 # max tokens per response
```

---

## Step 6 – Go offline and verify

1. Disconnect Wi-Fi or disable the network interface.
2. Start Ollama: `ollama serve` (it will use already-downloaded weights).
3. Start the app: `streamlit run app.py`
4. Confirm Ollama responds: `curl http://127.0.0.1:11434/api/tags`
5. Run `bash scripts/offline_proof.sh` to capture the evidence.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `LocalModelError: Cannot reach local Ollama` | Ollama server not running | `ollama serve` in a separate terminal |
| `Required local model(s) are not installed` | Model not pulled | `ollama pull MODEL` (requires internet) |
| `ConfigurationError: … cloud model` | Model tag ends in `:cloud` | Use a non-cloud tag |
| `ConfigurationError: … loopback` | `OLLAMA_BASE_URL` points to remote host | Set `OLLAMA_BASE_URL=http://127.0.0.1:11434` |
| `RuntimeError: Egress blocked` | App tried to reach non-loopback address | Check `OLLAMA_HOST` is set; restart Ollama with `OLLAMA_NO_CLOUD=1` |
| Slow inference | Model too large for available RAM | Use a smaller tag (e.g. `gemma3n:e2b`) or increase swap |

---

## Reference: key environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API URL (app-side) |
| `OLLAMA_MODEL` | `gemma4:e4b` | Chat model tag |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model tag |
| `OLLAMA_NO_CLOUD` | _(unset)_ | Set to `1` in Ollama server process to disable cloud features |
| `OLLAMA_HOST` | _(Ollama default)_ | Set to `127.0.0.1:11434` in Ollama server process to bind loopback only |
| `OLLAMA_DEBUG` | _(unset)_ | Set to `0` to disable debug logging |
| `OLLAMA_NUM_CTX` | `8192` | Context window size (tokens) |
| `OLLAMA_NUM_PREDICT` | `1600` | Max tokens per model response |
