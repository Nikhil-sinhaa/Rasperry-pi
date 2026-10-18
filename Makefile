.PHONY: install run test audit eval versions

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

versions:
	bash scripts/collect_versions.sh
