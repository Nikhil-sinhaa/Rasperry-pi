.PHONY: install run test audit eval versions cuad-gold cuad-eval secret-scan offline-proof \
        privacy-proof benchmark-models benchmark-one benchmark-quick sweep-threshold hand-review \
        hand-review-triage pick-citation-sample eval-full phase3

install:
	python -m pip install -r requirements.txt

run:
	streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false

test:
	pytest -q

audit:
	python scripts/static_privacy_audit.py

eval:
	python scripts/run_evals.py

cuad-gold:
	@echo "Extracting CUAD gold cases from data.zip …"
	unzip -o -d /tmp/cuad_extract scripts/data.zip CUADv1.json
	python scripts/cuad_to_gold.py \
	  --src /tmp/cuad_extract/CUADv1.json \
	  --subset dev --contracts 20 --out .
	@rm -rf /tmp/cuad_extract

cuad-eval: cuad-gold
	@echo "Running CUAD benchmark …"
	python scripts/score_cuad.py \
	  --gold eval/gold/cuad_gold.json \
	  --contracts-dir data/cuad_eval/contracts \
	  --out evidence/cuad-eval
	@echo "Report written to evidence/cuad-eval.json"

secret-scan:
	@echo "Scanning for secrets (gate G5)…"
	@git grep -rn --no-index -i \
	  -e "api_key" -e "secret_key" -e "password" -e "token" \
	  -e "OPENAI_API_KEY" -e "sk-" \
	  -- ':!*.md' ':!*.txt' ':!.env.example' \
	  && echo "⚠ Potential secrets found – review above before pushing" \
	  || echo "✅ No obvious secrets detected"

versions:
	bash scripts/collect_versions.sh

offline-proof:
	@echo "Running offline proof (macOS pfctl – requires sudo) …"
	bash scripts/offline_proof.sh $(SAMPLE)

# Linux network-namespace offline proof (no root required on modern kernels)
privacy-proof:
	@echo "Running Linux offline proof (network namespace) …"
	bash scripts/offline_proof_linux.sh $(SAMPLE)

# Docker-based isolated proof (cross-platform, requires Docker)
privacy-proof-docker:
	@echo "Running Docker isolated proof …"
	docker compose --profile proof run --rm proof

# ---------------------------------------------------------------------------
# Phase 3 – Evaluation
# ---------------------------------------------------------------------------

# Benchmark all four candidate models. Writes evidence/model-benchmark.md
# and evidence/model-benchmark.json. Requires Ollama + all four models pulled.
benchmark-models:
	@echo "Phase 3: benchmarking gemma3n:e2b, gemma3n:e4b, gemma4:e2b, gemma4:e4b …"
	python scripts/benchmark_models.py \
	  --output evidence/model-benchmark.md \
	  --json-output evidence/model-benchmark.json

# Benchmark a single model (faster for iteration). Usage: make benchmark-one MODEL=gemma3n:e2b
benchmark-one:
	python scripts/benchmark_models.py --models $(MODEL) \
	  --output evidence/model-benchmark.md \
	  --json-output evidence/model-benchmark.json

# Latency + RAM probe only (no full gold eval, ~2 min per model)
benchmark-quick:
	python scripts/benchmark_models.py --representative-only \
	  --output evidence/model-benchmark.md \
	  --json-output evidence/model-benchmark.json

# Sweep retrieval similarity threshold 0.15–0.45 against gold eval.
# Writes evidence/retrieval-threshold-sweep.md.
sweep-threshold:
	@echo "Phase 3: sweeping retrieval threshold …"
	python scripts/sweep_threshold.py \
	  --eval-dir eval/gold/ \
	  --output evidence/retrieval-threshold-sweep.md

# Run agent on unseen documents and write evidence/hand-review.md.
# Reviewer fills in the Verdict column manually.
hand-review:
	@echo "Phase 3: collecting findings on unseen documents …"
	python scripts/hand_review.py --output evidence/hand-review.md

hand-review-triage:
	python scripts/hand_review.py \
	  --mode triage --output evidence/hand-review-triage.md

# Generate the citation sample table from the gold eval JSON output.
# Run AFTER: make eval-full
pick-citation-sample:
	@echo "Phase 3: picking citation sample from live eval output …"
	python scripts/pick_citation_sample.py \
	  --source evidence/model-evaluation.json --n 20

# Run full gold eval and write JSON + markdown reports
eval-full:
	python scripts/run_evals.py \
	  --output evidence/model-evaluation.json \
	  --markdown evidence/model-evaluation.md

# One-shot Phase 3 sequence (allow ~60 min per model on Pi)
phase3: eval-full benchmark-models sweep-threshold hand-review pick-citation-sample cuad-eval
	@echo ""
	@echo "Phase 3 complete. Fill in evidence files:"
	@echo "  evidence/model-benchmark.md           -- update Chosen Default section"
	@echo "  evidence/retrieval-threshold-sweep.md -- confirm chosen threshold"
	@echo "  evidence/hand-review.md               -- fill in Verdict column"
	@echo "  evidence/citation-review.md           -- fill in Verdict column"
	@echo "  evidence/cuad-eval.json               -- copy by-category table to model-benchmark.md"
