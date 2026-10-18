# Start Here

1. Install Ollama and disable its cloud features as shown in `README.md`.
2. While online, run `ollama pull gemma4:e2b` and `ollama pull nomic-embed-text`.
3. Create an environment, install `requirements.txt`, and copy `.env.example` to `.env`.
4. Run `python scripts/check_local_setup.py`, `pytest -q`, and `python scripts/static_privacy_audit.py`.
5. Start `streamlit run app.py --server.address 127.0.0.1`; upload `samples/demo_rental_agreement.txt`.
6. Test the absent-information and adversarial cases in `tests/ADVERSARIAL_TESTS.md`.
7. Run `python scripts/run_evals.py` and the network/offline evidence procedure in `README.md` before submission.

**Do not report a privacy test or model evaluation as passed until you run it on your device.** Read `README.md` for the full PS2 L1–L10 mapping, limitations, model licence references, and evidence checklist.
