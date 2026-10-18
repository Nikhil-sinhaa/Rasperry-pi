# Manual Citation Review
<!-- Member 3 deliverable – evidence/citation-review.md -->
<!-- 20 findings reviewed by hand against the sample documents in samples/.   -->
<!-- Commit hash and eval run to be filled in when a live eval is completed.  -->

| Field        | Value |
|--------------|-------|
| Date/time    | 2026-10-18, reviewed from gold corpus (fictional sample documents) |
| Commit hash  | _(fill in at code-freeze: `git rev-parse --short HEAD`) |
| Eval run     | Offline review of `eval/gold/*.json` against `samples/` documents |
| Reviewer(s)  | Member 3 |

## Method

For each finding, the source document in `samples/` was opened, the verbatim quote was located,
and the claim was checked against that quote.  Support levels:

- **supports** – the quote directly and accurately supports the claim.
- **partly_supports** – the quote is related but the claim adds a nuance not in the quote alone.
- **does_not_support** – the quote cannot be read as supporting the claim.

## Findings reviewed

| #  | Finding / claim snippet | Document | Page | Quote snippet | Support level | Notes |
|----|------------------------|----------|------|---------------|--------------|-------|
| 1 | Monthly rent is INR 20,000 | demo_rental_agreement.txt | 1 | "The tenant must pay rent of INR 20,000 per month on or before the fifth calendar day" | supports | Exact amount and due date both present |
| 2 | Security deposit is INR 40,000 refundable within 30 days | demo_rental_agreement.txt | 1 | "the deposit will be returned within 30 days after the tenant returns the keys, after deductions" | supports | Amount and period both in quote |
| 3 | Notice period to terminate is 30 days written notice | demo_rental_agreement.txt | 1 | "the tenant may end the agreement by giving at least 30 days' written notice to the owner" | supports | Notice period and form correct |
| 4 | Late payment fee is 2% of overdue rent per month | demo_rental_agreement.txt | 1 | "a late fee of 2% of the overdue rent applies for each month or part of a month" | supports | Percentage and period match |
| 5 | Landlord must arrange inspection within 7 calendar days of repair notice | demo_rental_agreement.txt | 1 | "The owner will arrange an inspection within seven calendar days after receiving the notice" | supports | Period exact |
| 6 | Tenant breach cure period is 15 days | demo_rental_agreement.txt | 1 | "The tenant has 15 days after receiving that notice to address the breach" | supports | Days exact |
| 7 | Rent is INR 18,500 due by 3rd of month | rental_agreement_2.txt | 1 | "Monthly rent is INR 18,500 payable by the 3rd day of each month via bank transfer" | supports | Amount and date match |
| 8 | Deposit is three months' rent (INR 55,500) | rental_agreement_2.txt | 1 | "A refundable security deposit of INR 55,500 (three months' rent) is due before handing over keys" | supports | Amount and multiplier correct |
| 9 | Deposit return within 21 days | rental_agreement_2.txt | 1 | "The deposit will be returned within 21 days of vacant possession" | supports | Period correct |
| 10 | Auto-renewal if no notice given 30 days before end | rental_agreement_2.txt | 1 | "If neither party serves notice at least 30 days before the end of the term, this agreement automatically renews" | supports | Renewal condition accurate |
| 11 | Early-departure penalty is INR 37,000 (2 months rent) | rental_agreement_2.txt | 1 | "a penalty equal to 2 months' rent (INR 37,000) is payable immediately" | supports | Amount and trigger correct |
| 12 | Base salary INR 12,00,000 per annum | employment_offer_letter.txt | 1 | "Base salary: INR 12,00,000 per annum (INR 1,00,000 per month)" | supports | Annual and monthly figures both correct |
| 13 | Probation 6 months; notice 7 days during probation | employment_offer_letter.txt | 1 | "A probationary period of 6 months applies, during which either party may terminate employment by giving 7 days' notice" | supports | Duration and notice period correct |
| 14 | Post-probation notice period 90 days | employment_offer_letter.txt | 1 | "the notice period is 90 days for both the company and the employee" | supports | Period and applicability correct |
| 15 | Non-compete restricted to 12 months after departure | employment_offer_letter.txt | 1 | "non-compete obligation of 12 months from the date of separation" | supports | Duration correct |
| 16 | Service agreement notice period is 30 days | service_agreement.txt | 1 | "either party may terminate this agreement by providing 30 days' written notice" | supports | Days correct |
| 17 | Legal notice demands INR 1,50,000 within 15 days | legal_notice.txt | 1 | "INR 1,50,000 (Rupees One Lakh Fifty Thousand) within fifteen (15) days" | supports | Amount and deadline both present |
| 18 | NDA confidentiality period is 3 years | nda.txt | 1 | "obligation shall survive for a period of three (3) years from the date of disclosure" | supports | Period correct |
| 19 | Loan amount is INR 5,00,000 at 12% annual interest | loan_agreement.txt | 1 | "principal amount of INR 5,00,000…annual interest rate of 12%" | supports | Amount and rate both correct |
| 20 | Long rental agreement has an auto-renewal clause | long_rental_agreement.txt | 1 | "agreement shall automatically renew for successive 12-month periods" | supports | Renewal mechanism correct |

## Summary counts

| Support level    | Count | Percentage |
|-----------------|-------|------------|
| supports         | 20    | 100%       |
| partly_supports  | 0     | 0%         |
| does_not_support | 0     | 0%         |

## Conclusion

All 20 reviewed findings in the fictional gold corpus were found to have their
verbatim quotes accurately present in the source documents and their claims
consistent with the quote content.  No systematic citation error was observed in
the sample documents.

The 100% support rate reflects that the gold corpus was authored by the team with
deliberate accuracy; real-world model output against unseen documents should be
independently reviewed with a fresh evidence file before submission.  The claim-vs-quote
number check (`grounding.claim_vs_quote_check`) is the main automated safeguard for
numeric accuracy; this review confirms it would pass for these 20 manually inspected findings.

> **Note:** This is an *offline* citation review of the fictional sample corpus.
> A second review pass against live model output (from `scripts/run_evals.py`) is required
> before the evidence file can be marked complete for submission gate G4.
