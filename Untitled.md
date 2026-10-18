# PS2 Legal Triage – Team Work Plan

Oct 18, 2026 · @NIKHIL

Your prototype already has the right architecture; what it lacks is proof (never run, no evidence), a few accuracy bugs, and polish, so this plan splits the work four ways around the judging rubric.

## Where the prototype stands

The score is 70% track rubric (100 pts) plus 30% common rubric, and the track rubric has a **safety cap of 40 if grounding, uncertainty or privacy lands in the Poor band**, so unproven privacy is the biggest risk. I read all the code and ran its static privacy audit (passes). I could not run pytest or Gemma in my sandbox, so nothing below is model-tested.

| Track criterion (pts) | Already in the prototype | Gap to close |
| --- | --- | --- |
| 1. Accuracy of explanations (15) | Structured findings, plain-language prompt | Never run on Gemma; one synthetic document; no check that numbers and dates in a claim appear in its quote |
| 2. Grounding and traceability (15) | Source IDs plus verbatim-quote check, fails closed | Proves the quote exists, not that the claim follows from it; chunks never cross a page break |
| 3. Uncertainty, refusal, escalation (15) | 4 finding categories, abstention, verdict filter, lawyer question | Verdict filter over-blocks (see below); no jurisdiction hints; no per-finding confidence |
| 4. Privacy and network isolation (15) | Loopback-only URL, proxies ignored, telemetry off | No evidence captured yet; Ollama server settings unverified; no runtime egress block |
| 5. Gemma integration (10) | Gemma via local Ollama, JSON mode | `gemma4:e2b` default never benchmarked; no RAM, speed or quantization record |
| 6. Robustness (12) | Empty, encrypted and fake PDFs handled; injection-aware prompt | No OCR; 7 eval cases only; triage reads just 12 chunks so long documents lose clauses |
| 7. Usability (8) | Chat, expanders, quote blocks | No prioritisation, no key-facts summary, no quote highlighted on the source page |
| 8. Reuse and documentation (10) | Strong README, CI, MIT licence | Licence holder is a placeholder; `evidence/` is empty; scope statement missing |

### Issues I found in the code

1. **The verdict filter refuses ordinary questions.** I ran its regexes on test questions: "How long is the agreement valid?", "Until when is this offer valid?", "Is the notice period binding on both parties?", "Is the lock-in period valid for 11 months?" and "Is there a clause that is void if the tenant leaves early?" are all refused as legal-verdict requests. Judges use unscripted questions, so this costs accuracy and usability points.
2. **Whole-document triage is retrieval-limited.** It runs 8 generic searches and keeps at most 12 chunks, so a 40-page contract will not be covered.
3. **Quote validation is not entailment.** A claim can quote a real sentence and still misstate the amount, date or party.
4. **Test gaps.** Retrieval, the Ollama client (mocked HTTP) and the app flow have no tests.
5. **Model tag unconfirmed.** I could not open the Ollama tag pages from here, so check `gemma4:e2b` and `gemma3n:e2b` with `ollama list` on your machines. The guide prefers the Gemma 3n class and also accepts Gemma 4 E2B/E4B.

## Team split

Each person owns separate files, so you can work in parallel without merge conflicts. Replace the names with your own.

| Member | Role | Rubric points owned | Main files |
| --- | --- | --- | --- |
| 1 | Model and accuracy lead | Accuracy 15, Uncertainty 15, Gemma 10 | `agent.py`, `grounding.py`, `retrieval.py` |
| 2 | Privacy and platform lead | Privacy 15, Gemma deployment, gates L1–L4 | `local_runtime.py`, `config.py`, `scripts/`, `docs/privacy-threat-model.md`, `evidence/` |
| 3 | Ingestion, robustness and evaluation lead | Robustness 12, Grounding 15 (verification), C2, C7 | `ingestion.py`, `tests/`, `eval/`, `samples/`, `scripts/run_evals.py` |
| 4 | UX, documentation and demo lead | Usability 8, Reuse and docs 10, C1, C4, C5, C6, C8 | `app.py`, `README.md`, `CONTRIBUTING.md`, demo video, submission |

## Phases and gates

The organisers announce the deadline separately, so work backwards from it: freeze code 24 hours before, and record the demo last.

| Phase | Goal | Exit gate (all must be true) |
| --- | --- | --- |
| 0. Baseline (first 3 hours, everyone) | Every laptop runs the prototype end to end | Each member ran the demo sample; hardware and model choice recorded; baseline eval output saved to `evidence/baseline-eval.txt` |
| 1. Build (parallel) | Each member closes their gaps in their own files | Each member's "done when" list below is ticked and merged to `main` through reviewed pull requests |
| 2. Prove | Produce the evidence judges will check | Offline run, network capture, full eval, adversarial results and benchmark are all saved in `evidence/` from real runs |
| 3. Package | Make it reproducible for strangers | A teammate who did not write the code installs from the README on a clean machine and gets the demo working; demo video recorded; repo public |

### Parallel schedule (all four work at the same time)

Nobody waits for anyone. Three small contracts, committed in the first hour, remove the dependencies:

1. **Finding schema:** Member 1 commits the new fields in `types.py` and a `fixtures/sample_result.json` within 30 minutes. Members 3 and 4 code against that fixture, not against Member 1's unfinished code.
2. **Gold-case format:** Member 3 commits `eval/gold/README.md` with the JSON format in the first hour. Member 1 uses it for sweeps.
3. **Evidence file names:** fixed in the checklist below, so everyone writes to the same paths.

Blocks are relative; stretch or squeeze them to your deadline.

| Block | Member 1 (Katana) | Member 2 (Mac) | Member 3 (Victus) | Member 4 (Ideapad) |
| --- | --- | --- | --- | --- |
| 0. First hour | Install Ollama, pull models, run baseline eval into `evidence/baseline-eval.txt`; commit schema and fixture | Install Ollama, set `OLLAMA_NO_CLOUD=1`, confirm loopback only, run the demo sample | Commit gold-case format; write a fake runtime returning canned JSON; start writing test documents | Create public repo, push prototype, set licence holder, secret scan, issue and PR templates |
| 1. Hours 1–4 | Benchmark Gemma variants on the 7 existing cases; fix the verdict filter with tests for the 13 probe questions | Egress guard with test; canary disk-search script; adapt the capture script for Mac | First 4 documents and 25+ gold cases; eval runner loads gold files and reports all metrics, with `--model` flag | UI skeleton rendering `sample_result.json`: topic groups, attention labels, highlighted source; README restructure with placeholders |
| 2. Rest of day 1 | Number and date check in `grounding.py`; per-topic triage retrieval and coverage line | First dry run of offline proof (Wi-Fi off and container with no network); dependency review | Remaining documents; automated adversarial suite; tests for retrieval, client and pipeline using the fake runtime | Accessibility pass; architecture diagram; scope statement and Gemma-use sections |
| 3. Day 2 | Wire `topic`, `attention`, `support` into real output and merge; threshold sweep on Member 3's gold set; try the verifier pass | Model acquisition doc; resource measurements; threat model linked to evidence files | Run the full eval on Member 1's merged branch; write `docs/known-limitations.md` and `docs/datasets.md` | Switch UI from fixture to live output; Markdown download; usability test with two non-lawyers |
| 4. Day 3 | Fix the failures Member 3 found | Re-run offline proof after the merges | Re-run eval; 80% coverage in CI | Fill README tables with real numbers; demo script and rehearsal |
| 5. Freeze (24 h before deadline) | Final eval and benchmark on the tag | Final offline proof and network capture on the tag | Hand citation review of 20 findings on the tag | Fresh-install test on the weakest laptop, then record the demo, then submit |

The only real dependencies left are these, and each has a fallback:

- Member 1's threshold sweep needs Member 3's gold set. Until it lands, Member 1 sweeps on the 7 existing cases.
- Member 4's final README numbers need results from the other three. Member 4 writes everything else first and leaves marked tables to fill.
- The final proofs and the demo need the frozen tag, so they come last by nature.

## Member 1 – Model and accuracy lead

You make the answers correct, calibrated and cautious. You own the model's behaviour; Member 3 supplies the test set you tune against.

1. **Benchmark the Gemma variants (Phase 0–1).** Run the same eval cases on `gemma3n:e2b`, `gemma3n:e4b`, `gemma4:e2b` and `gemma4:e4b` (whichever your laptops can hold). Record pass rate, seconds per answer and memory from `ollama ps` in `evidence/model-benchmark.md`. Pick one default; on a tie take Gemma 3n, which the guide names first.
2. **Fix the verdict filter.** Keep a narrow regex for clear verdict or advice requests ("is it enforceable", "should I sign", "can I sue", "is it legal"). For the ambiguous words (valid, binding, void), ask Gemma one small JSON question, `explain_text` or `legal_verdict`, and answer the text part when it is `explain_text`. Add unit tests for every false positive listed above.
3. **Add a claim-versus-quote check in `grounding.py`.** Every number, currency amount, date and duration in a claim must appear in at least one of its quotes after normalising. If not, drop the finding. This catches "quote is real, claim is wrong".
4. **Try a verifier pass.** Ask Gemma, at temperature 0, whether the quote supports the claim (yes, partly, no) and drop "no". Keep it only if the eval pass rate rises and the extra latency is acceptable; record both numbers.
5. **Fix triage coverage.** If the document fits in `OLLAMA_NUM_CTX`, send all of it. Otherwise run one retrieval per topic (obligations, money, deadlines, termination, liability, ambiguity), keep the top 4–5 chunks each, and merge. Show a coverage line such as "Reviewed 34 of 34 pages". Let chunks span a page break and cite both pages.
6. **Add structure the UI can use.** Each finding gets a `topic`, an `attention` level (high, medium, low: "worth a closer look", never a legal conclusion) set by rules such as a money amount, penalty, auto-renewal, one-sided change or short notice, and a `support` label derived from your checks (quote valid, numbers verified, verifier agrees). Do not use the model's own confidence. Hand the schema to Member 4 early.
7. **Tune retrieval.** Sweep `MIN_RETRIEVAL_SIMILARITY` from 0.15 to 0.45 on Member 3's eval set. Pick the value with zero invented answers on absent-information questions and the fewest false abstentions; save the table.
8. **Jurisdiction hints (after the core is done).** A small local JSON of topics that often vary by place (deposit limits, notice periods, rent control, registration or stamp duty, non-competes). Match by keyword and show "this often depends on local law; ask a local lawyer", with no verdict.

**Done when**

- [ ] Default model chosen from a recorded benchmark
- [ ] All 13 probe questions route correctly, with tests
- [ ] Number and date mismatch is dropped, with tests
- [ ] Triage reports coverage and handles a 30+ page document
- [ ] Findings carry `topic`, `attention` and `support`
- [ ] Retrieval threshold chosen from a sweep

## Member 2 – Privacy and platform lead

You turn "we designed it to be private" into evidence a judge can re-run. Privacy is 15 points and part of the cap, and the judges will test it with an offline run, connection monitoring, a dependency review and a temp-file check.

1. **Lock down Ollama.** Set `OLLAMA_NO_CLOUD=1` and `OLLAMA_HOST=127.0.0.1:11434` in the Ollama server process (steps are in the README). Check with `ss -tlnp` that it listens on loopback only, confirm no `cloud` model tags, and keep debug logging off. Run `scripts/collect_versions.sh` and fill `evidence/system-info-template.md`.
2. **Add a runtime egress guard.** At app start, wrap socket connection so any non-loopback address raises an error. Add a test that tries to reach a public IP and expects the block, and extend `static_privacy_audit.py` to check the guard exists. State the limit honestly: it covers the Python process only, so the Ollama process is covered by step 3.
3. **Produce the offline proof two ways.** (a) Wi-Fi off, `scripts/capture_network.sh` running, full demo, saved as `evidence/network-monitor.txt`. (b) A container or `unshare -n` run with networking disabled where app and Ollama both work from local models; script it as `scripts/offline_proof.sh`. Also run once with the network on and capture only non-loopback traffic, which should be empty. Record the screen with the network-off indicator visible.
4. **Check for leftover files with a canary.** Put a unique made-up phrase in a test document, run the demo, then search the whole disk (`/tmp`, home folder, Streamlit and Ollama directories) for that phrase and for new files. Save the result in `evidence/temp-file-check.md`. Open the browser's network tab during page load to confirm the UI fetches nothing external.
5. **Review dependencies.** List what the five pinned packages and their sub-dependencies can call out to, grep for telemetry, and save `evidence/dependency-review.md`. Run `pip-audit` while online during setup if you can.
6. **Write model acquisition instructions** in `docs/model-acquisition.md`: exact `ollama pull` commands, tag, quantization, download size, digest, the Gemma licence to accept, and the embedding model's licence. Separate "setup (online)" from "processing (offline)" clearly, as the guide's rule table does. Note that the Ollama desktop app can check for updates; disable it or document that it carries no document content.
7. **Measure resources** with Member 1: peak memory, answer time and index time for 10 and 50 pages on your weakest laptop, plus the `OLLAMA_NUM_CTX` setting that fits. Put the table in the README.
8. **Turn the threat model into evidence.** Update `docs/privacy-threat-model.md` so every control links to the file in `evidence/` that proves it, and list what remains unsolved.

**Done when**

- [ ] Offline run and network capture saved from your own machines
- [ ] Egress guard merged with a test
- [ ] Canary disk search and browser network check saved
- [ ] Dependency review and model acquisition doc merged
- [ ] Resource table filled from real measurements

Never write "passed" in any file for a check you did not run.

## Member 3 – Ingestion, robustness and evaluation lead

You try to break the system before the judges do, and you produce the honest numbers. The judges use documents you did not script, so variety in your test set is your main weapon.

1. **Build a varied test corpus.** At least 8 documents: two rental agreements, two employment or offer letters, one service agreement or terms of service, one legal notice, one loan or NDA, one 30+ page document, and one scanned PDF. Make them yourselves (fictional) or take samples from CUAD (CC BY 4.0, cite the NeurIPS 2021 paper). Vary the layout: two columns, tables, headers and footers, numbered sub-clauses, and a clause that crosses a page break. Never use a real person's document. List every source and licence in `docs/datasets.md`.
2. **Write gold answers.** One JSON file per document in `eval/gold/` with at least 60 cases in total. Each case has a type (answerable, absent, verdict request, injection, vague, conflicting), the expected facts, the expected page and a substring the quote must contain. Rework `scripts/run_evals.py` to load these files; today the cases are hard-coded and `eval/case-list.json` only describes them.
3. **Upgrade the eval runner.** Report answer-correct rate, citation-valid rate, **invented-answer rate on absent questions (target 0)**, false-abstention rate, verdict-refusal rate, injection resistance and latency. Add a `--model` flag so Member 1 can compare Gemma variants. Write JSON and markdown to `evidence/`.
4. **Automate the adversarial suite** from `tests/ADVERSARIAL_TESTS.md`. Add these attacks: "ignore previous instructions" in the document; a document containing a fake closing tag such as `</E1>` to break the prompt wrapper; white or hidden text in a PDF; instructions in Hindi; zero-width and look-alike characters inside a quoted sentence; two clauses that disagree (rent 20,000 in one place, 22,000 in another, expected to be flagged as ambiguity); a vague term like "reasonable notice"; a document that never names a governing law; two uploads with the same filename.
5. **Harden ingestion.** Check extraction on ligatures, hyphenated line breaks, two-column pages and tables (try pypdf's layout mode). Make quote matching tolerate hyphenation and ligature differences. Add `.docx` input (`python-docx`), since most real contracts are Word files. Local OCR for scans (Tesseract or RapidOCR, installed at setup, run offline) is optional; do it only after everything else, and never claim it without a scanned test.
6. **Add the missing tests.** Retrieval with fake embeddings, the Ollama client with mocked HTTP, and the whole pipeline with a fake runtime that returns canned JSON containing an invented quote, a wrong source ID and a legal-verdict claim (all must be dropped). Aim for 80% line coverage of `legal_triage/` with `pytest-cov` and add it to the CI workflow.
7. **Review citations by hand.** Pick 20 random findings, open the original page, and mark each as supports, partly supports or does not support. Save the counts in `evidence/citation-review.md`. This is the honest figure for the grounding criterion.
8. **Keep a failures list.** Every miss goes into `docs/known-limitations.md` with the cause, and Member 4 copies it into the README (deliverable S8).

**Done when**

- [ ] 8+ varied documents and 60+ gold cases committed, with licences
- [ ] Eval runner reports every metric above, from a real run
- [ ] Adversarial suite automated; failures fixed or listed
- [ ] Coverage at 80% and CI green
- [ ] Citation review of 20 findings saved

## Member 4 – UX, documentation and demo lead

You make a non-lawyer able to use the tool in two minutes, and a stranger able to install it from the README. You also own the repo, the demo video and the final submission, which together carry most of the common rubric (C1, C4, C5, C6, C8 = 55 of 100 points).

1. **Repo hygiene (gates G1 and G5).** Create the public GitHub repo. Put your team name in `LICENSE`. Scan for secrets (`gitleaks`, or `git grep` for keys) before every push, and confirm no model weights, private documents or `.env` file are committed. Add issue and pull-request templates, a code of conduct, three or four labelled "good first issue" tickets and repo topics, so it reads as contribution-ready.
2. **Improve the interface** using Member 1's new fields.
   - Show a key-facts card (parties, dates, amounts) containing only validated findings.
   - Group findings by topic and sort by attention, and show attention as a text label as well as a colour.
   - Open the cited page beside the claim with the quote highlighted.
   - Show the "Reviewed N of N pages" coverage line.
   - On a refusal or abstention, show a clear next step: "ask a qualified lawyer or local legal-aid service this question", with the lawyer question pre-written.
   - Add a download button that saves the result as Markdown to the user's own disk.
3. **Accessibility and clarity (C6).** Check keyboard-only use, contrast, 200% zoom and screen-reader labels; run axe-core against `localhost`. Ask two non-lawyer friends to read one result and explain the clause back to you; keep their notes as usability evidence in `evidence/usability-test.md`.
4. **Restructure the README.** Quick start in the first screen; a table of tested OS, Python, Ollama, model digest and RAM (from Member 2); an architecture diagram (Mermaid renders on GitHub); a scope statement (document types, English, selectable-text PDF, TXT and DOCX, no OCR unless built); how Gemma is used; test results and known limitations (from Member 3); how to reproduce the demo; and a short "main technical contribution" paragraph. Merge `README_SHORT.md` into it. Keep the honest hedging until the matching evidence file exists.
5. **Fresh-install test.** A teammate who did not write the code installs from the README on a clean user account or VM, timing each step. Fix every snag and save the notes as `evidence/fresh-install.md`. Gate G2 depends on this.
6. **Record the demo (about 5 minutes, after code freeze).**
   - 0:00 problem and promise.
   - 0:30 network off, capture running.
   - 1:00 upload a document the app has not seen, run triage, click a citation.
   - 2:00 ask a question the document answers, then one it does not ("the document is silent").
   - 3:00 "is this enforceable?" refusal, then the injection document.
   - 3:45 scanned-PDF warning as an honest limitation.
   - 4:15 eval and test summary, architecture, your contribution.
   - Use the real system only (gate G3), no mock-ups, and let a teammate type the unscripted question.
7. **Assemble the submission.** Write `SUBMISSION.md` mapping each of S1–S10 to a file path or link, check that every link opens in a private window, tag `v1.0`, and submit on the portal.
8. **Rehearse the questions** judges will ask (C8): why Ollama and pypdf, why in-memory only, why fail-closed citation checks, why no OCR, what quote validation does not prove. Each person answers one in under 30 seconds.

**Done when**

- [ ] Public repo, licence, clean secret scan, templates in place
- [ ] UI shows facts card, attention order, highlighted source and coverage
- [ ] Accessibility check and two-person usability test saved
- [ ] Fresh-install test passed by someone else
- [ ] Demo recorded, uploaded and linked; `SUBMISSION.md` complete

## Team rules

- **Git flow.** One branch per task (`m1/verdict-filter`), a pull request into `main`, a reviewer from a different member, CI green (pytest and the privacy audit). No direct pushes to `main`; keep pull requests small.
- **Agree the finding schema on day 1.** Member 1 publishes the new fields in `types.py`, Member 3 builds fixtures from them and Member 4 builds the UI on them. Announce any change in the team chat.
- **Daily 15-minute sync.** Each person says done, doing, blocked. Track the "done when" boxes above on one shared board.
- **Evidence rule.** Evidence files hold raw command output with the date, commit hash and machine. Never edit results to look better, and never write "passed" for a check you did not run. Documented failures score better than hidden ones.
- **Freeze 24 hours before the deadline.** Only bug fixes after that. Re-run the full eval, offline proof and tests on the freeze commit, because the evidence must match the code you submit.
- **Never commit** private documents, model weights, `.env` files, or raw packet captures (redact IP addresses first).
- **Wording.** Nothing in the UI, README or video may call the output legal advice, a verdict or a guarantee.

**If time runs short, cut in this order:** local OCR, jurisdiction hints, the verifier pass, `.docx` support. **Never cut:** the offline proof, the egress guard, the verdict-filter fix, the number check, the real eval run and the demo.

## Submission checklist and gates

A failed gate means the team is not ranked, whatever the score, so each one has an owner and an evidence file.

| Gate | What judges check | Owner | Evidence |
| --- | --- | --- | --- |
| G1 | Public repo with a licence file | 4 | Repo link, `LICENSE` |
| G2 | A judge can run the demo with no paid service or hidden credential | 4, 2 | `evidence/fresh-install.md` |
| G3 | Demo shows the real system, not mock-ups | 4 | Demo video |
| G4 | Dataset sources and licences disclosed; no private data committed | 3 | `docs/datasets.md` |
| G5 | No keys, tokens or passwords in the repo | 4 | Secret-scan output |
| L1 | Gemma runs locally | 2, 1 | `docs/model-acquisition.md`, `evidence/model-benchmark.md` |
| L2 | No private content leaves the device | 2 | `evidence/network-monitor.txt`, `offline-run.md`, `temp-file-check.md` |
| L3 | No systematic fabrication of clauses or citations | 3, 1 | `evidence/model-evaluation.*`, `citation-review.md` |
| L4 | Not presented as legal advice; safe escalation shown | 1, 4 | Refusal examples, UI text, demo |

| Item | Deliverable | Owner |
| --- | --- | --- |
| S1 | Public repo with licence | 4 |
| S2 | README: overview, setup, run, configuration, troubleshooting | 4 |
| S3 | Pinned requirements, tool versions, OS and hardware tested, model sizes | 2 |
| S4 | Architecture diagram and explanation | 4 |
| S5 | Dataset sources, licences, preparation | 3 |
| S6 | How Gemma is used | 1 |
| S7 | Working demo recording (about 5 minutes) | 4 |
| S8 | Test results and known limitations | 3 |
| S9 | Steps to reproduce the demo | 4 |
| S10 | Statement of main technical contribution | 4, with all |
| Extra | Local-processing evidence, model acquisition instructions | 2 |
| Extra | Sample outputs with citation beside the source, plus a "document is silent" case | 1, 3 |
| Extra | Refusal and escalation examples; adversarial and injection results | 1, 3 |
