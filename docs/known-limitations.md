# Known Limitations
<!-- Member 3 maintains this file. Member 4 copies it into the README (deliverable S8). -->
<!-- Add every observed miss here with its cause and status. -->
<!-- Documented failures score better than hidden ones. -->

## Format

| # | Limitation | Cause | Severity | Status |
|---|-----------|-------|----------|--------|
| … | … | … | high/medium/low | open/fixed/workaround |

## Current known limitations

| # | Limitation | Cause | Severity | Status |
|---|-----------|-------|----------|--------|
| 1 | Scanned/image-only PDFs are not processed | No OCR; pypdf cannot extract image text | high | open – stated in UI |
| 2 | Clauses on very long documents may be missed even with per-topic retrieval | Embedding similarity threshold may filter out weakly-scored but relevant chunks | medium | open – retrieval threshold tuning pending (Member 1) |
| 3 | Quote validation does not prove semantic entailment | Verbatim match confirms quote exists, not that the claim follows from it | medium | open – verifier pass partially addresses this |
| 4 | Non-selectable PDFs (forms, old scans) produce empty extraction without OCR warning | pypdf returns empty text; ingestion raises IngestionError with OCR message | high | open – stated in UI |
| 5 | Ligatures (ﬁ, ﬂ, ﬀ) in PDFs may break quote matching | pypdf may not normalise ligatures; normalise_quote() does not substitute them | low | open – Member 3 normalisation added |
| 6 | Hyphenated line breaks in multi-column PDFs may split words | pypdf may return "agree-\nment" instead of "agreement" | low | open – Member 3 normalisation added |
| 7 | DOCX files were not previously supported | Original ingestion handled PDF and TXT only | medium | fixed – Member 3 adds python-docx support |
| 8 | Verdict filter (before Member 1 fix) blocked ordinary questions | Over-broad regex matched "valid", "binding", etc. | high | fixed – Member 1 narrowed patterns |
| 9 | Invented facts in claims were not caught | Quote validation confirms quote exists but not claim numbers | high | fixed – Member 1 added claim_vs_quote_check |
| 10 | Whole-document triage only used 12 chunks from 8 generic queries | Retrieval missed clauses in long documents | medium | fixed – Member 1 added per-topic retrieval and full-doc path |
| 11 | Gold eval cases were hard-coded in run_evals.py | No structured test corpus | medium | fixed – Member 3 added eval/gold/ JSON files and upgraded runner |
| 12 | Two-column PDFs may produce interleaved text extraction | pypdf reads columns left-to-right by character position, not column order | medium | open – test with pypdf layout mode; workaround: convert to TXT |
| 13 | Injection via Unicode look-alike characters in a quoted sentence is not fully defended | The verbatim quote check normalises whitespace only | low | open – adversarial test added; impact assessed |

## How to add entries

1. Describe the limitation as a user-visible behaviour (what goes wrong).
2. State the technical cause (what in the code allows it).
3. Assign severity (high = affects core rubric criteria; medium = affects quality; low = edge case).
4. Set status to `open`, `fixed – [description]`, or `workaround – [description]`.
5. Open a GitHub issue and cross-reference it here.
