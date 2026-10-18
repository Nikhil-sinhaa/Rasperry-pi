#!/usr/bin/env bash
set -euo pipefail

echo '=== OS ==='
if [ -r /etc/os-release ]; then cat /etc/os-release; else uname -a; fi
echo
echo '=== Python ==='
python --version
python -m pip show streamlit requests numpy pypdf python-dotenv 2>/dev/null | grep -E '^(Name|Version):' || true
echo
echo '=== Ollama ==='
if command -v ollama >/dev/null 2>&1; then
  ollama --version
  echo '-- locally installed model tags --'
  ollama list
  echo '-- active model resource allocation (run after a model call) --'
  ollama ps
else
  echo 'Ollama command not found. Install/configure it before model-backed evaluation.'
fi
echo
echo '=== Hardware ==='
if command -v lscpu >/dev/null 2>&1; then lscpu | grep -E '^(Model name|CPU\(s\)|Architecture):' || true; fi
if command -v free >/dev/null 2>&1; then free -h; fi
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true; fi
