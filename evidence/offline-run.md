# Offline Run – Evidence

> **How to fill this file:** Run `make offline-proof SAMPLE=samples/demo_rental_agreement.txt`
> (requires sudo for pfctl on macOS).  The script writes `evidence/network-monitor.txt`
> with the tcpdump output and prints PASS/FAIL.  Then fill in the fields below
> from the terminal output.  Never write "PASS" for a run you did not perform.

---

## Run details

| Field | Value |
|-------|-------|
| Date | _(fill after run)_ |
| Commit hash | _(git rev-parse --short HEAD)_ |
| Machine | _(e.g. MacBook Pro M2, 16 GB / Raspberry Pi 4, 8 GB)_ |
| OS | _(e.g. macOS 15.3 / Raspberry Pi OS Bookworm 64-bit)_ |
| Python | _(python --version)_ |
| Ollama version | _(ollama --version)_ |
| Chat model | _(e.g. gemma3n:e2b · sha256:…)_ |
| Embedding model | _(e.g. nomic-embed-text · sha256:…)_ |
| Network isolation method | macOS pfctl anchor `legal_triage_offline_proof` (blocks non-loopback TCP/UDP) |
| Sample document | samples/demo_rental_agreement.txt |

---

## Step 1 – Pre-flight: Ollama reachable on loopback

```
[fill: paste the curl output confirming Ollama is running on 127.0.0.1:11434]
```

## Step 2 – pfctl anchor applied

```
[fill: paste the pfctl confirmation line]
```

## Step 3 – Headless probe output

```
[fill: paste the terminal output ending with OFFLINE_PROOF: PASS or FAIL]
```

Expected output (example):
```
[probe] Loading document: samples/demo_rental_agreement.txt
[probe] Building in-memory index …
[probe] Asking: 'What is the monthly rent and when is it due?'
[probe] Response received in 8.3s
[probe] Got 2 finding(s).
  [1] explicit_statement: The monthly rent is INR 15,000 payable on or before…
       quote: The Tenant shall pay a monthly rent of INR 15,000 on or before the fifth…
OFFLINE_PROOF: PASS
```

## Step 4 – tcpdump evidence

See `evidence/network-monitor.txt` for the full tcpdump output.

Summary:
```
[fill: paste the STATUS line from network-monitor.txt, e.g.]
Non-loopback TCP/UDP packets captured: 0
STATUS: PASS – no external packets during offline test.
```

## Step 5 – Canary disk search

Run `bash scripts/canary_disk_search.sh` after the demo.
See `evidence/temp-file-check.md` for the result.

Summary: _(fill: PASS / FAIL)_

---

## Interpretation

_(fill: one paragraph. Example:)_

> The application completed a full embed + retrieve + generate cycle answering a
> question about the rental agreement while all non-loopback TCP/UDP traffic was
> blocked by pfctl.  tcpdump captured zero packets on external interfaces during
> the 45-second test window.  The canary phrase was not found on disk.
> This demonstrates that the Python process does not require external network
> access at inference time and does not write document content to disk.

---

## Residual risks acknowledged

- OS swap may page process memory to disk (documented in `docs/privacy-threat-model.md`).
- The Ollama server process has its own isolation requirements (`OLLAMA_NO_CLOUD=1`);
  this proof covers only the Python/Streamlit process.
- The socket egress guard does not cover `connect_ex`, UDP, or DNS (defence-in-depth only).
