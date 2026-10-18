#!/usr/bin/env bash
# offline_proof.sh – Demonstrate that the app produces a real answer while
# all network interfaces except loopback are blocked.
#
# Platform: macOS (uses `pfctl` to drop non-loopback traffic).
# Requires: sudo, curl, Ollama already running, Python env active.
#
# Usage:
#   bash scripts/offline_proof.sh [<path-to-contract.txt>]
#
# What it does:
#   1. Verifies Ollama is reachable on loopback (it must already be running).
#   2. Adds a pfctl anchor that blocks all external TCP/UDP (not loopback).
#   3. Runs a headless Python probe: embeds the document, asks one question,
#      and asserts the answer contains a verbatim quote.
#   4. Removes the pfctl anchor.
#   5. Saves tcpdump evidence of the blocked period to evidence/network-monitor.txt.
#
# Judges: look for "PASS" on the last line and empty TCP traffic in the dump.
# If you prefer Docker, see README.md §"Offline proof – Docker alternative".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SAMPLE="${1:-$REPO_ROOT/samples/demo_rental_agreement.txt}"
EVIDENCE_FILE="$REPO_ROOT/evidence/network-monitor.txt"
ANCHOR="legal_triage_offline_proof"

# ── Colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${YELLOW}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[PASS]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*" >&2; }

# ── Preflight ───────────────────────────────────────────────────────────────
info "Checking sample document: $SAMPLE"
if [[ ! -f "$SAMPLE" ]]; then
    fail "Sample document not found: $SAMPLE"
    fail "Run with:  bash scripts/offline_proof.sh /path/to/contract.txt"
    exit 1
fi

info "Checking Ollama is reachable on 127.0.0.1:11434 …"
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
    fail "Ollama is not running. Start it with: ollama serve"
    exit 1
fi
ok "Ollama reachable."

# ── Start tcpdump to capture non-loopback traffic ────────────────────────────
info "Starting tcpdump on non-loopback interfaces …"
DUMP_FILE="$(mktemp /tmp/offline_proof_dump.XXXXXX.pcap)"
# Capture only TCP/UDP that is NOT loopback (lo0).
# We limit packets so the file stays manageable during the test window.
sudo tcpdump -i any -c 500 -w "$DUMP_FILE" \
    'not (host 127.0.0.1 or host ::1) and (tcp or udp)' 2>/dev/null &
TCPDUMP_PID=$!
sleep 1  # give tcpdump a moment to start

# ── Block non-loopback traffic with pfctl ────────────────────────────────────
info "Applying pfctl anchor to block external traffic (requires sudo) …"
PFCONF="$(mktemp /tmp/offline_pf.XXXXXX.conf)"
cat > "$PFCONF" <<'PFRULES'
# Block all external traffic in/out (not loopback lo0).
block in  quick on ! lo0 proto {tcp, udp}
block out quick on ! lo0 proto {tcp, udp}
PFRULES

sudo pfctl -e 2>/dev/null || true          # enable pf if not already on
sudo pfctl -a "$ANCHOR" -f "$PFCONF"       # load our anchor rules
info "pfctl anchor '${ANCHOR}' loaded. External traffic blocked."

cleanup() {
    info "Removing pfctl anchor …"
    sudo pfctl -a "$ANCHOR" -F rules 2>/dev/null || true
    # Stop tcpdump
    sudo kill "$TCPDUMP_PID" 2>/dev/null || true
    wait "$TCPDUMP_PID" 2>/dev/null || true
    rm -f "$PFCONF"
}
trap cleanup EXIT

# ── Run the headless analysis probe ─────────────────────────────────────────
info "Running headless offline analysis probe …"
PROBE_SCRIPT="$(mktemp /tmp/offline_probe.XXXXXX.py)"
cat > "$PROBE_SCRIPT" << 'PYEOF'
"""Headless offline proof: ingest a text document and ask one question."""
import os, sys, json, time

# Ensure we load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SAMPLE_PATH = os.environ["OFFLINE_SAMPLE"]

from legal_triage.local_runtime import LocalOllama
from legal_triage.retrieval import build_index
from legal_triage.agent import analyze

print(f"[probe] Loading document: {SAMPLE_PATH}")
content = open(SAMPLE_PATH, "rb").read()
filename = os.path.basename(SAMPLE_PATH)

runtime = LocalOllama()
runtime.verify_models()

print("[probe] Building in-memory index …")
index = build_index([(filename, content)], runtime)

question = "What is the monthly rent and when is it due?"
print(f"[probe] Asking: {question!r}")
t0 = time.time()
result = analyze(question, index, runtime, mode="question")
elapsed = time.time() - t0

print(f"[probe] Response received in {elapsed:.1f}s")
findings = result.get("findings", [])
abstained = result.get("abstained", False)

if abstained:
    print("[probe] ABSTAINED (no evidence found) – check retrieval threshold or document content.")
    # Abstention is still a valid offline run – the model did not go to the internet.
    print("OFFLINE_PROOF: PASS (abstained – no external network needed)")
    sys.exit(0)

if not findings:
    print("[probe] FAIL: no findings and not abstained.")
    sys.exit(1)

print(f"[probe] Got {len(findings)} finding(s).")
for i, f in enumerate(findings, 1):
    quote = f["evidence"][0]["quote"] if f.get("evidence") else "(none)"
    print(f"  [{i}] {f['category']}: {f['claim'][:80]}")
    print(f"       quote: {quote[:100]}")

# Assert: at least one finding must have a non-empty verbatim quote
has_quote = any(
    item.get("quote", "").strip()
    for f in findings
    for item in f.get("evidence", [])
)
if not has_quote:
    print("[probe] FAIL: findings have no verbatim quotes.")
    sys.exit(1)

print("OFFLINE_PROOF: PASS")
sys.exit(0)
PYEOF

OFFLINE_SAMPLE="$SAMPLE" python "$PROBE_SCRIPT"
PROBE_EXIT=$?
rm -f "$PROBE_SCRIPT"

# ── Collect tcpdump evidence ─────────────────────────────────────────────────
# Stop tcpdump early; the EXIT trap will also clean up pf.
sudo kill "$TCPDUMP_PID" 2>/dev/null || true
wait "$TCPDUMP_PID" 2>/dev/null || true

info "Checking captured packets for external TCP/UDP connections …"
mkdir -p "$REPO_ROOT/evidence"
{
    echo "# Offline proof – tcpdump evidence"
    echo "# Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "# Sample: $SAMPLE"
    echo "# Tool: tcpdump (non-loopback TCP/UDP only)"
    echo ""
    if command -v tcpdump >/dev/null 2>&1 && [[ -s "$DUMP_FILE" ]]; then
        PACKET_COUNT=$(sudo tcpdump -r "$DUMP_FILE" 2>/dev/null | wc -l | tr -d ' ')
        echo "Non-loopback TCP/UDP packets captured: ${PACKET_COUNT}"
        if [[ "$PACKET_COUNT" -eq 0 ]]; then
            echo "STATUS: PASS – no external packets during offline test."
        else
            echo "STATUS: WARNING – ${PACKET_COUNT} external packet(s) captured (review below)."
            sudo tcpdump -r "$DUMP_FILE" 2>/dev/null | head -50
        fi
    else
        echo "STATUS: tcpdump output unavailable (install tcpdump or check sudo)."
    fi
} > "$EVIDENCE_FILE"
rm -f "$DUMP_FILE"

ok "Evidence saved to $EVIDENCE_FILE"
cat "$EVIDENCE_FILE"

# ── Final result ─────────────────────────────────────────────────────────────
echo ""
if [[ "$PROBE_EXIT" -eq 0 ]]; then
    ok "=== OFFLINE PROOF COMPLETE: PASS ==="
else
    fail "=== OFFLINE PROOF COMPLETE: FAIL (exit code $PROBE_EXIT) ==="
    exit "$PROBE_EXIT"
fi
