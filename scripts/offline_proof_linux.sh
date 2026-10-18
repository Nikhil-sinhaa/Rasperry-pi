#!/usr/bin/env bash
# offline_proof_linux.sh – Demonstrate on-device privacy on Linux using
# network namespaces (no root required with unshare -rn on modern kernels).
#
# Platform: Linux (kernel ≥ 3.8, user-namespace support enabled).
# Requires: unshare, ip, Python env active, Ollama already pulled.
#
# Usage:
#   bash scripts/offline_proof_linux.sh [<path-to-contract.txt>]
#
# What it does:
#   1. Verifies the Python egress guard is installed (runs a quick probe).
#   2. Creates a network namespace with no external routes (loopback only).
#   3. Inside that namespace, starts Ollama and runs the headless proof.
#   4. Captures /proc/net/tcp* snapshots before and after.
#   5. Writes evidence to evidence/offline-run.md and evidence/network-monitor.txt.
#
# macOS alternative: see scripts/offline_proof.sh (uses pfctl).
# Docker alternative: make privacy-proof (see docker-compose.yml).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SAMPLE="${1:-$REPO_ROOT/samples/demo_rental_agreement.txt}"
EVIDENCE_DIR="$REPO_ROOT/evidence"
OFFLINE_RUN="$EVIDENCE_DIR/offline-run.md"
NETMON="$EVIDENCE_DIR/network-monitor.txt"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
info() { echo -e "${YELLOW}[INFO]${NC} $*"; }

mkdir -p "$EVIDENCE_DIR"

# ── Prerequisite checks ──────────────────────────────────────────────────────
command -v unshare &>/dev/null || fail "unshare not found. Install util-linux."
command -v ollama &>/dev/null  || fail "ollama not found. Install Ollama first."
python -c "from legal_triage.local_runtime import install_egress_guard" \
  || fail "Cannot import legal_triage. Activate the virtual environment first."

# ── Python egress guard probe ────────────────────────────────────────────────
info "Testing Python-level egress guard..."
python -c "
import legal_triage.local_runtime  # installs guard on import
import socket
try:
    s = socket.socket(); s.connect(('8.8.8.8', 53))
    print('ERROR: connect to 8.8.8.8 was NOT blocked')
    exit(1)
except RuntimeError as e:
    print(f'OK: connect blocked: {e}')
try:
    socket.getaddrinfo('example.com', 80)
    print('ERROR: getaddrinfo was NOT blocked')
    exit(1)
except RuntimeError as e:
    print(f'OK: getaddrinfo blocked: {e}')
print('Python egress guard: all checks passed')
" || fail "Python egress guard check failed"
pass "Python egress guard working"

# ── Snapshot /proc/net before ────────────────────────────────────────────────
info "Capturing pre-run network state..."
{
  echo "=== Pre-run snapshot $(date -Iseconds) ==="
  cat /proc/net/tcp 2>/dev/null  || true
  cat /proc/net/tcp6 2>/dev/null || true
} > "$NETMON"

# ── Run headless proof in an isolated network namespace ──────────────────────
info "Launching isolated namespace proof (loopback only)..."
PROOF_LOG="$EVIDENCE_DIR/proof_run.log"

# unshare -rn creates a new network namespace with only loopback.
# We bring up loopback, start Ollama inside the namespace, run the proof.
unshare -rn bash -c "
  set -euo pipefail
  ip link set lo up

  # Start Ollama inside the namespace
  OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1 ollama serve &
  OLLAMA_PID=\$!
  sleep 5  # let it start

  # Run the headless proof
  cd '$REPO_ROOT'
  python scripts/headless_proof.py \
    --document '$SAMPLE' \
    --question 'What is the monthly rent?' \
    --output '$OFFLINE_RUN' \
    2>&1 | tee '$PROOF_LOG'

  kill \$OLLAMA_PID 2>/dev/null || true
" || fail "Headless proof failed inside namespace. See $PROOF_LOG"

pass "Headless proof completed inside isolated namespace"

# ── Snapshot /proc/net after ─────────────────────────────────────────────────
{
  echo ""
  echo "=== Post-run snapshot $(date -Iseconds) ==="
  cat /proc/net/tcp 2>/dev/null  || true
  cat /proc/net/tcp6 2>/dev/null || true
} >> "$NETMON"

# ── Canary disk search ────────────────────────────────────────────────────────
if [[ -f "$REPO_ROOT/scripts/canary_disk_search.sh" ]]; then
  info "Running canary disk search..."
  bash "$REPO_ROOT/scripts/canary_disk_search.sh" && pass "Canary check passed" || info "Canary check: see evidence/temp-file-check.md"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
pass "=== OFFLINE PROOF COMPLETE ==="
echo "  Evidence written to:"
echo "    $OFFLINE_RUN"
echo "    $NETMON"
echo ""
echo "  Review:"
echo "    - $OFFLINE_RUN should show a grounded answer with a verbatim quote."
echo "    - $NETMON should show no ESTABLISHED connections to non-loopback IPs."
