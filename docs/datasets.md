# Dataset Sources and Licences
<!-- Member 3 deliverable – docs/datasets.md -->
<!-- Maps to submission gate G4: dataset sources and licences disclosed -->

All documents in `samples/` are **entirely fictional** and were created by the
project team for the sole purpose of testing the PS2 Legal Triage application.
No real persons, organisations, addresses, or transactions are represented.

## Corpus summary

| File | Type | Pages | Source | Licence |
|------|------|-------|--------|---------|
| `demo_rental_agreement.txt` | Rental agreement | 1 (TXT) | Team-authored fictional | CC0 (team waives copyright) |
| `adversarial_injection_test.txt` | Adversarial injection | 1 (TXT) | Team-authored | CC0 |
| `rental_agreement_2.txt` | Rental agreement | 1 (TXT) | Team-authored fictional | CC0 |
| `employment_offer_letter.txt` | Employment offer | 1 (TXT) | Team-authored fictional | CC0 |
| `service_agreement.txt` | Service/SaaS | 1 (TXT) | Team-authored fictional | CC0 |
| `legal_notice.txt` | Legal notice (demand) | 1 (TXT) | Team-authored fictional | CC0 |
| `nda.txt` | Non-disclosure agreement | 1 (TXT) | Team-authored fictional | CC0 |
| `loan_agreement.txt` | Loan agreement | 1 (TXT) | Team-authored fictional | CC0 |
| `terms_of_service.txt` | Terms of service | 1 (TXT) | Team-authored fictional | CC0 |
| `long_rental_agreement.txt` | Multi-section rental | 8 sections (TXT) | Team-authored fictional | CC0 |

## Privacy statement

- No real document belonging to any person, organisation, or client was used.
- All names, amounts, dates, and addresses are invented.
- No document contains personal data as defined by applicable data protection law.
- No CUAD documents were used in the current corpus. If CUAD samples are added in
  future, they must be cited as: Hendrycks et al., "CUAD: An Expert-Annotated NLP
  Dataset for Legal Contract Review," NeurIPS 2021, licensed CC BY 4.0.

## Gold evaluation cases

| File | Cases | Document |
|------|-------|----------|
| `eval/gold/rental1_gold.json` | 14 | `demo_rental_agreement.txt` |
| `eval/gold/rental2_gold.json` | 8 | `rental_agreement_2.txt` |
| `eval/gold/employment_gold.json` | 8 | `employment_offer_letter.txt` |
| `eval/gold/service_gold.json` | 7 | `service_agreement.txt` |
| `eval/gold/notice_gold.json` | 4 | `legal_notice.txt` |
| `eval/gold/nda_gold.json` | 4 | `nda.txt` |
| `eval/gold/loan_gold.json` | 7 | `loan_agreement.txt` |
| `eval/gold/long_rental_gold.json` | 9 | `long_rental_agreement.txt` |

Total: 61 cases across 8 document types.

## Layout variants covered

- ☑ Plain prose with numbered clauses (rental, employment, NDA)
- ☑ Itemised payment schedule with multiple amounts (service agreement, loan)
- ☑ Short-form legal notice with demand and deadline
- ☑ Multi-section long document with page markers (long rental)
- ☑ Terms of service with unilateral-change and auto-renewal clauses
- ☐ Two-column PDF layout – not yet available; add before code freeze
- ☐ Scanned/image PDF – not supported (OCR not implemented; documented in known-limitations.md)
- ☐ DOCX format – supported by ingestion after Member 3 addition

## How to add more documents

1. Create a fictional document with no real personal data.
2. Name it descriptively (e.g. `franchise_agreement.txt`).
3. Place it in `samples/`.
4. Add gold cases to a new file in `eval/gold/` following the schema in existing files.
5. Add a row to the table above and confirm the licence.
6. Never commit a real person's document.
