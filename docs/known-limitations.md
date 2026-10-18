# Known Limitations

> **Owner:** Member 4 (copy summary into README §Known Limitations / S8)  
> **Updated:** add a row every time a test failure or edge case is identified.

Every miss is recorded here with the cause and current status. Documented failures score
better than hidden ones. Do not remove rows; mark them **Fixed** with the commit hash instead.

---

## Format

| # | Area | Symptom | Root cause | Status |
|---|------|---------|------------|--------|
| example | Quote validation | Model wraps quote in quotation marks; validation fails | Grounding strips leading/trailing `"` but not smart-quotes | Fixed `abc1234` |

---

## Active Limitations

| # | Area | Symptom | Root cause | Status |
|---|------|---------|------------|--------|
| 1 | Ingestion | Scanned / image-only PDF pages return no text | pypdf cannot OCR rasterised pages | **By design** – flagged in UI and README; no fix planned |
| 2 | Retrieval | Very long contracts (>45 000 chars) may miss clauses in dense middle sections | Chunk count is large; per-topic search helps but does not guarantee full coverage | Open – see `scripts/sweep_retrieval_threshold.py` |
| 3 | Grounding | Multi-sentence quotes that span a page break are sometimes truncated | Chunk boundary splits the quote | Open – workaround: re-ask with narrower question |
| 4 | Model | Gemma occasionally returns findings without a `lawyer_question` field | Model omits optional field at low temperature | Open – UI handles missing field gracefully |
| 5 | Eval | CUAD benchmark covers commercial contracts only | Dataset scope (Hendrycks et al. 2021) | **By design** – stated in README and benchmark output |
| 6 | Privacy | RAM is not an encrypted vault; session state persists until cleared | Streamlit in-memory design | **By design** – documented; mitigated by "Clear" button |
| 7 | UX | No OCR for scanned PDFs | OCR libraries add size/latency; out of scope for PS2 | **By design** – honest limitation shown in UI |

---

## Fixed Limitations

| # | Area | Symptom | Fix | Commit |
|---|------|---------|-----|--------|
| F1 | Verdict filter | Ambiguous words (valid/binding/void) caused over-refusal | Gemma disambiguation call added | `39bd189` |
| F2 | Triage coverage | Single bulk search missed clauses in long documents | `retrieve_triage_per_topic()` added | `39bd189` |
| F3 | Citation | Non-loopback Ollama URLs were not blocked at startup | Config validator added | `39bd189` |
