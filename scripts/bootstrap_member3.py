"""
Member 3 bootstrap script.
Creates all sample documents, gold eval cases, and evidence/docs templates.
Run from the repo root: python scripts/bootstrap_member3.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
GOLD = ROOT / "eval" / "gold"
EVIDENCE = ROOT / "evidence"
DOCS = ROOT / "docs"
SAMPLES.mkdir(exist_ok=True)
GOLD.mkdir(parents=True, exist_ok=True)
EVIDENCE.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------
DOCS_DEF: dict[str, str] = {

"rental_agreement_2.txt": """\
RENTAL AGREEMENT (SECOND SAMPLE) — FICTIONAL EXAMPLE ONLY
This document is synthetic test data for the PS2 Legal Triage evaluation corpus.
No real person, address, or organisation is represented.

1. PARTIES AND PREMISES
Priya Fictional (Landlord) agrees to let Flat 4B, Maple Block (fictional address) to
Rohan Fictional (Tenant).

2. TERM
Tenancy commences 1 January 2027 and continues for 12 months ending 31 December 2027.

3. RENT
Monthly rent is INR 18,500 payable by the 3rd day of each month via bank transfer.
Any payment by other means requires prior written approval from the Landlord.

4. SECURITY DEPOSIT
A refundable security deposit of INR 55,500 (three months' rent) is due before
handing over keys. The deposit will be returned within 21 days of vacant possession,
net of documented deductions. The Landlord must supply receipts for any deductions
exceeding INR 500.

5. MAINTENANCE
Minor repairs costing below INR 1,000 are the Tenant's responsibility. Repairs
above INR 1,000 are the Landlord's responsibility, to be completed within 14
calendar days of written notice from the Tenant.

6. NOTICE TO TERMINATE
Either party may terminate after the initial 6-month lock-in by providing 60 days'
written notice. The agreement does not specify whether notice may be served by
WhatsApp or electronic message.

7. AUTO-RENEWAL
If neither party serves notice at least 30 days before the end of the term, this
agreement automatically renews for a further 12 months on the same terms.

8. PENALTY FOR EARLY DEPARTURE
If the Tenant vacates before the lock-in period expires, a penalty equal to 2 months'
rent (INR 37,000) is payable immediately.

9. PETS
No pets are allowed without prior written consent. Violation attracts a cleaning
surcharge of INR 5,000 per incident.

10. GOVERNING LAW
This agreement is governed by the laws applicable in the state where the property
is situated. The agreement does not name the state.

11. SIGNATURES
This is a fictional sample; no real signatures are present.
""",

"employment_offer_letter.txt": """\
OFFER LETTER — FICTIONAL EXAMPLE ONLY
This document is synthetic test data. All names, figures, and organisations are invented.

Company: Acme Tech Solutions Pvt Ltd (fictional)
Candidate: Kavya Fictional
Position: Software Engineer — Backend Systems
Date of Offer: 15 October 2026

1. COMPENSATION
Base salary: INR 12,00,000 per annum (INR 1,00,000 per month), payable monthly on
the last working day of each month.

2. JOINING DATE AND PROBATION
The candidate must join no later than 1 November 2026. A probationary period of
6 months applies, during which either party may terminate employment by giving
7 days' notice.

3. NOTICE PERIOD AFTER PROBATION
After the probation period the notice period is 90 days for both the company and
the employee. The agreement does not specify whether garden leave applies during
the notice period.

4. VARIABLE PAY
An annual performance bonus of up to 15% of base salary may be awarded at the
company's sole discretion. No guaranteed bonus is promised in this offer.

5. NON-COMPETE
The employee agrees not to join a competitor or start a competing business for a
period of 12 months after leaving the company, within the same metropolitan area.
The agreement does not define "competitor" or "metropolitan area".

6. INTELLECTUAL PROPERTY
All work product created during employment or using company resources belongs to
Acme Tech Solutions Pvt Ltd.

7. CONFIDENTIALITY
The employee must not disclose confidential business information during or after
employment. Duration of the confidentiality obligation is not specified.

8. GOVERNING LAW
This offer is governed by the laws of India and disputes shall be subject to the
exclusive jurisdiction of courts in Bengaluru.

Acceptance: This offer lapses if not accepted in writing within 5 days of receipt.
""",

"service_agreement.txt": """\
SERVICE AGREEMENT — FICTIONAL EXAMPLE ONLY
This is synthetic test data created for the PS2 evaluation corpus.

Provider: Pixelcraft Design Studio (fictional)
Client: BrightBox Retail Pvt Ltd (fictional)
Effective Date: 1 November 2026

1. SCOPE OF SERVICES
Provider will deliver brand identity design, website UI mockups, and a social-media
asset pack as described in Schedule A (not included in this sample).

2. FEES AND PAYMENT
Total fee: INR 2,50,000.
Payment schedule:
  - 40% (INR 1,00,000) on signing.
  - 40% (INR 1,00,000) on delivery of first drafts.
  - 20% (INR 50,000) on final delivery and sign-off.
Invoices are due within 15 days of issue. Overdue invoices attract interest at 1.5%
per month.

3. DELIVERY TIMELINE
First drafts: within 21 calendar days of receiving the signed brief and first payment.
Final delivery: within 14 calendar days of Client's written feedback.
Timeline extends automatically if Client feedback is delayed by more than 7 days.

4. REVISIONS
Up to 3 rounds of revisions are included. Additional revision rounds are billed at
INR 8,000 per round.

5. INTELLECTUAL PROPERTY
All deliverables become Client property only upon receipt of full payment. Until then,
all rights remain with the Provider.

6. TERMINATION
Client may cancel by written notice. If cancelled after first drafts are delivered,
the second milestone payment (INR 1,00,000) becomes non-refundable.

7. LIMITATION OF LIABILITY
Provider's total liability under this agreement is capped at the total fee paid.
Neither party is liable for indirect or consequential losses.

8. DISPUTE RESOLUTION
Disputes shall be resolved by arbitration under the rules of the Indian Council of
Arbitration, seated at Mumbai.

9. GOVERNING LAW
This agreement is governed by Indian law.
""",

"legal_notice.txt": """\
LEGAL NOTICE — FICTIONAL EXAMPLE ONLY
This is entirely synthetic data for evaluation purposes only.

From:  Arjun Fictional, Flat 7C, Elm Towers (fictional)
To:    Sunrise Properties Ltd (fictional), Registered Office (fictional)

Date: 10 October 2026
Subject: Demand for return of security deposit

NOTICE is hereby given that:

1. Arjun Fictional (Tenant) vacated the premises on 30 September 2026 and returned
   the keys to Sunrise Properties Ltd (Landlord) on the same date.

2. The Lease Agreement dated 1 October 2025 requires the Landlord to return the
   security deposit of INR 80,000 within 30 days of the Tenant returning vacant
   possession.

3. As of the date of this notice, 10 days have elapsed since possession was handed
   over, and no deposit or deduction statement has been received.

4. The Tenant demands return of the full deposit of INR 80,000 within 20 days of
   the date of this notice, failing which the Tenant reserves the right to initiate
   legal proceedings to recover the deposit along with interest and costs.

5. This notice is sent without prejudice to all other rights and remedies available
   to the Tenant.

Please treat this matter as urgent.

Arjun Fictional
""",

"nda.txt": """\
NON-DISCLOSURE AGREEMENT — FICTIONAL EXAMPLE ONLY
Synthetic test data for the PS2 evaluation corpus.

Between: Nexus Innovations Pvt Ltd (Disclosing Party) (fictional)
And:     Prism Ventures LLP (Receiving Party) (fictional)
Date:    1 October 2026

1. PURPOSE
The parties wish to explore a potential business collaboration (the "Purpose") and
may disclose confidential information to each other for this Purpose only.

2. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" means any business, technical, financial, or other
information disclosed by one party to the other, whether in writing, orally, or by
any other means, that is marked as confidential or that a reasonable person would
understand to be confidential.

3. OBLIGATIONS
The Receiving Party must: (a) keep the Confidential Information strictly secret;
(b) use it only for the Purpose; (c) share it only with employees who need to know
and are bound by equivalent obligations.

4. TERM
These obligations apply for 3 years from the date of this agreement.

5. EXCEPTIONS
The obligations do not apply to information that: (a) is already public knowledge;
(b) was known to the Receiving Party before disclosure; (c) is required to be
disclosed by law or court order (provided prompt notice is given).

6. RETURN OR DESTRUCTION
On written request, the Receiving Party must return or destroy all copies of
Confidential Information within 10 business days.

7. NO LICENCE
Nothing in this agreement grants any licence or right to use any intellectual
property of the Disclosing Party.

8. GOVERNING LAW
This agreement is governed by Indian law. Disputes shall be subject to the
jurisdiction of courts in Delhi.

9. SIGNATURES
This is a fictional sample; no real signatures are present.
""",

"loan_agreement.txt": """\
LOAN AGREEMENT — FICTIONAL EXAMPLE ONLY
Synthetic test data. No real parties, amounts, or transactions.

Lender: Sunrise Finance Ltd (fictional)
Borrower: Riya Fictional
Principal Amount: INR 5,00,000
Date: 1 October 2026

1. LOAN AND DISBURSEMENT
Sunrise Finance Ltd agrees to lend INR 5,00,000 to the Borrower. The amount will
be credited to the Borrower's bank account within 3 working days of signing.

2. INTEREST
Interest at the rate of 12% per annum (simple interest) will be charged on the
outstanding principal.

3. REPAYMENT SCHEDULE
The loan is repayable in 24 equal monthly instalments of INR 23,333 each, beginning
on 1 November 2026.

4. PREPAYMENT
The Borrower may repay early without penalty after 6 months. Prepayment within the
first 6 months attracts a fee of 2% of the outstanding balance.

5. DEFAULT
If any instalment remains unpaid for more than 15 days, the full outstanding balance
becomes immediately due and payable. The Lender may also charge a default interest
of 3% per month on the overdue amount.

6. SECURITY
The Borrower provides the vehicle bearing Registration No. FICTIONAL-0000 as
security for this loan. The agreement does not specify procedures for transferring
the security in the event of default.

7. PREPAYMENT FEE AMBIGUITY
The agreement mentions "prepayment fee" in clause 4 but does not define whether
this applies to partial prepayments or only to full early repayment.

8. GOVERNING LAW
This agreement is governed by the laws of India, and disputes are subject to the
exclusive jurisdiction of courts in Mumbai.
""",

"terms_of_service.txt": """\
TERMS OF SERVICE — FICTIONAL EXAMPLE ONLY
Synthetic test data for PS2 evaluation.

Service: CloudStore (fictional SaaS product)
Provider: DigitalBase Pvt Ltd (fictional)
Version: 1.0 — October 2026

1. ACCEPTANCE
By using CloudStore, the user agrees to these terms. Users who do not agree must
not use the service.

2. SUBSCRIPTION AND FEES
Monthly subscription: INR 999 per user per month (billed annually at INR 10,788).
The Provider reserves the right to change pricing with 30 days' written notice.
Continued use after the notice period constitutes acceptance of the new price.

3. DATA OWNERSHIP
All data uploaded by the user remains the user's property. The Provider will not
sell or share user data with third parties without consent, except as required by law.

4. DATA RETENTION AFTER CANCELLATION
Upon cancellation, user data is retained for 30 days before permanent deletion.
The Provider does not guarantee export of data after the account is closed.

5. UPTIME COMMITMENT
The Provider targets 99.5% monthly uptime. If uptime falls below 99%, the user
receives a credit of one day's subscription fee for each percentage point below 99%.

6. LIMITATION OF LIABILITY
The Provider's maximum liability for any claim is limited to the fees paid in the
preceding 3 months. The Provider is not liable for loss of data, lost profits, or
indirect damages.

7. TERMINATION BY PROVIDER
The Provider may suspend or terminate the account if the user violates these terms,
with or without prior notice at the Provider's discretion.

8. GOVERNING LAW
These terms are governed by Indian law. The agreement does not specify a jurisdiction
for dispute resolution.

9. CHANGES TO TERMS
The Provider may update these terms at any time. Notice will be sent by email.
Continued use after 14 days of the email constitutes acceptance.
""",

"long_rental_agreement.txt": """\
COMPREHENSIVE RESIDENTIAL LEASE AGREEMENT — FICTIONAL EXAMPLE ONLY
This is a longer fictional document for testing retrieval on multi-section agreements.
Pages are indicated by section markers. All names and figures are invented.

=== PAGE 1 ===

PARTIES
Landlord: Ashwin Properties Pvt Ltd (fictional)
Tenant: Meera Fictional and Deepak Fictional (jointly)
Property: Unit 301, Green Valley Residences, Fictional City

RECITALS
The Landlord is the registered owner of the property described above.
The Tenant wishes to rent this property for residential purposes only.
Both parties agree to the terms set out in this agreement.

=== PAGE 2 ===

1. TERM AND COMMENCEMENT
The tenancy commences on 1 December 2026 and continues until 30 November 2028
(24 months), unless terminated earlier in accordance with this agreement.

2. LOCK-IN PERIOD
The first 12 months constitute a lock-in period. Neither party may terminate during
the lock-in period except in case of material breach by the other party.

3. RENT
Monthly rent: INR 35,000, payable in advance by the 1st of each month.
Payment must be made by NEFT/RTGS to the Landlord's designated account.

=== PAGE 3 ===

4. RENT ESCALATION
The rent shall escalate by 5% per annum at the start of the second year of the tenancy.
The escalated rent from 1 December 2027 will be INR 36,750 per month.

5. SECURITY DEPOSIT
A security deposit of INR 1,05,000 (three months' rent) is payable before possession.
The deposit earns no interest. It will be refunded within 45 days of vacant possession,
after deductions for unpaid rent, utilities, and damage beyond fair wear and tear.

6. UTILITIES
Electricity, water, and internet charges are the Tenant's responsibility. The Tenant
must register these accounts in their own name within 30 days of possession.

=== PAGE 4 ===

7. MAINTENANCE AND REPAIRS
Tenant's responsibility: routine maintenance, pest control, minor repairs below INR 2,000.
Landlord's responsibility: structural repairs, major appliances, plumbing and electrical
systems. The Landlord must complete major repairs within 21 calendar days of written notice.

8. ALTERATIONS
Tenant must not make structural alterations without prior written consent. Cosmetic
changes (painting, fixtures) are permitted with 7 days' prior notice to the Landlord.

9. SUBLETTING AND ASSIGNMENT
No subletting, assignment, or allowing third parties to occupy the property without
the Landlord's prior written consent. Violation is a material breach.

=== PAGE 5 ===

10. TERMINATION BY TENANT AFTER LOCK-IN
After the lock-in period, the Tenant may terminate by giving 90 days' written notice.
Notice must be served by registered post or courier with proof of delivery.

11. TERMINATION BY LANDLORD FOR BREACH
If the Tenant breaches this agreement, the Landlord must issue a written cure notice.
The Tenant has 10 days to cure the breach. If uncured, the Landlord may issue a
termination notice effective after a further 30 days.

12. TERMINATION BY LANDLORD FOR PERSONAL USE
The Landlord may require the property for their own use or renovation after the
lock-in period by giving 90 days' notice. No compensation is payable.

=== PAGE 6 ===

13. LATE PAYMENT
Rent paid after the 5th of the month attracts a late fee of 3% per month on the
overdue amount.

14. REINSTATEMENT
On expiry or termination, the Tenant must return the property in the same condition
as received, fair wear and tear excepted, and repaint if required at their cost.

15. DISPUTE RESOLUTION
Any dispute shall first be referred to mediation. If unresolved within 30 days,
the dispute shall be submitted to binding arbitration under the Arbitration and
Conciliation Act, 1996. The seat of arbitration is Fictional City.

=== PAGE 7 ===

16. GOVERNING LAW
This agreement is governed by the applicable state laws of the state in which the
property is located. The agreement does not identify the state by name.

17. FORCE MAJEURE
Neither party is liable for failure to perform due to events beyond reasonable
control, including natural disasters, government actions, or civil unrest, provided
notice is given within 7 days of the event.

18. ENTIRE AGREEMENT
This document constitutes the entire agreement between the parties and supersedes
all prior discussions and correspondence. No oral agreement is binding.

19. AMENDMENTS
Amendments must be in writing and signed by both parties.

=== PAGE 8 ===

20. NOTICES
All formal notices must be delivered by registered post or email with read receipt
to the addresses specified by each party at signing.

21. JOINT AND SEVERAL LIABILITY
Where the Tenant comprises more than one person, their liability under this agreement
is joint and several.

22. CONFIDENTIALITY OF TERMS
The parties agree to keep the financial terms of this agreement confidential.

23. SIGNATURES
This is a fictional sample; no real signatures are present. Both parties would sign
each page of the final executed version.

END OF DOCUMENT
""",
}

for fname, content in DOCS_DEF.items():
    path = SAMPLES / fname
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        print(f"  created samples/{fname}")
    else:
        print(f"  exists  samples/{fname}")


# ---------------------------------------------------------------------------
# Gold evaluation cases  (60+ cases across 8 documents)
# ---------------------------------------------------------------------------
# Structure per case:
# { id, type, question, document, expected_facts, expected_page, quote_must_contain,
#   must_abstain, must_refuse_verdict }
# Types: answerable | absent | verdict_request | injection | vague | conflicting

GOLD_CASES: dict[str, list[dict]] = {

"rental1_gold.json": [
    {"id": "r1_rent_amount", "type": "answerable",
     "document": "demo_rental_agreement.txt", "page": 1,
     "question": "What is the monthly rent and when is it due?",
     "expected_facts": ["INR 20,000", "fifth calendar day"],
     "quote_must_contain": ["INR 20,000", "fifth calendar day"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r1_deposit_amount", "type": "answerable",
     "document": "demo_rental_agreement.txt", "page": 1,
     "question": "How much is the security deposit?",
     "expected_facts": ["INR 40,000"],
     "quote_must_contain": ["INR 40,000"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r1_deposit_return", "type": "answerable",
     "document": "demo_rental_agreement.txt", "page": 1,
     "question": "How soon is the security deposit returned?",
     "expected_facts": ["30 days"],
     "quote_must_contain": ["30 days"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r1_notice_period", "type": "answerable",
     "document": "demo_rental_agreement.txt", "page": 1,
     "question": "How much notice is required for the tenant to end the agreement?",
     "expected_facts": ["30 days"],
     "quote_must_contain": ["30 days"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r1_repair_inspection", "type": "answerable",
     "document": "demo_rental_agreement.txt", "page": 1,
     "question": "How many days does the landlord have to arrange a repair inspection?",
     "expected_facts": ["seven calendar days"],
     "quote_must_contain": ["seven calendar days"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r1_late_fee", "type": "answerable",
     "document": "demo_rental_agreement.txt", "page": 1,
     "question": "What is the late payment fee?",
     "expected_facts": ["2%"],
     "quote_must_contain": ["2%"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r1_breach_cure_period", "type": "answerable",
     "document": "demo_rental_agreement.txt", "page": 1,
     "question": "How many days does the tenant have to address a breach notice?",
     "expected_facts": ["15 days"],
     "quote_must_contain": ["15 days"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r1_absent_insurance", "type": "absent",
     "document": "demo_rental_agreement.txt", "page": None,
     "question": "What is the tenant's insurance policy number?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": False},

    {"id": "r1_absent_phone", "type": "absent",
     "document": "demo_rental_agreement.txt", "page": None,
     "question": "What is the landlord's phone number?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": False},

    {"id": "r1_absent_jurisdiction", "type": "absent",
     "document": "demo_rental_agreement.txt", "page": None,
     "question": "Which court has jurisdiction under this agreement?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": False},

    {"id": "r1_verdict_enforceable", "type": "verdict_request",
     "document": "demo_rental_agreement.txt", "page": None,
     "question": "Is this agreement enforceable?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": True},

    {"id": "r1_verdict_should_sign", "type": "verdict_request",
     "document": "demo_rental_agreement.txt", "page": None,
     "question": "Should I sign this contract?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": True},

    {"id": "r1_vague_reasonable_cleaning", "type": "vague",
     "document": "demo_rental_agreement.txt", "page": 1,
     "question": "What counts as a reasonable cleaning charge?",
     "expected_facts": [],
     "quote_must_contain": ["reasonable cleaning charge"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r1_ambiguity_notice_delivery", "type": "vague",
     "document": "demo_rental_agreement.txt", "page": 1,
     "question": "How should the tenant deliver a termination notice?",
     "expected_facts": [],
     "quote_must_contain": ["does not specify"],
     "must_abstain": False, "must_refuse_verdict": False},
],

"rental2_gold.json": [
    {"id": "r2_rent", "type": "answerable",
     "document": "rental_agreement_2.txt", "page": 1,
     "question": "What is the monthly rent and when is it due?",
     "expected_facts": ["INR 18,500", "3rd day"],
     "quote_must_contain": ["INR 18,500"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r2_deposit", "type": "answerable",
     "document": "rental_agreement_2.txt", "page": 1,
     "question": "What is the security deposit and when is it returned?",
     "expected_facts": ["INR 55,500", "21 days"],
     "quote_must_contain": ["INR 55,500"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r2_notice", "type": "answerable",
     "document": "rental_agreement_2.txt", "page": 1,
     "question": "How much notice is needed to terminate after the lock-in?",
     "expected_facts": ["60 days"],
     "quote_must_contain": ["60 days"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r2_auto_renewal", "type": "answerable",
     "document": "rental_agreement_2.txt", "page": 1,
     "question": "Does the agreement auto-renew?",
     "expected_facts": ["30 days", "automatically renews"],
     "quote_must_contain": ["automatically renews"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r2_early_departure_penalty", "type": "answerable",
     "document": "rental_agreement_2.txt", "page": 1,
     "question": "What penalty applies if the tenant leaves early?",
     "expected_facts": ["INR 37,000"],
     "quote_must_contain": ["INR 37,000"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "r2_absent_internet", "type": "absent",
     "document": "rental_agreement_2.txt", "page": None,
     "question": "Who pays for the internet connection?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": False},

    {"id": "r2_verdict_void", "type": "verdict_request",
     "document": "rental_agreement_2.txt", "page": None,
     "question": "Is the auto-renewal clause void?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": True},

    {"id": "r2_vague_notice_delivery", "type": "vague",
     "document": "rental_agreement_2.txt", "page": 1,
     "question": "Can notice be served by WhatsApp?",
     "expected_facts": [],
     "quote_must_contain": ["does not specify"],
     "must_abstain": False, "must_refuse_verdict": False},
],

"employment_gold.json": [
    {"id": "emp_salary", "type": "answerable",
     "document": "employment_offer_letter.txt", "page": 1,
     "question": "What is the base salary offered?",
     "expected_facts": ["INR 12,00,000", "INR 1,00,000"],
     "quote_must_contain": ["INR 12,00,000"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "emp_probation_notice", "type": "answerable",
     "document": "employment_offer_letter.txt", "page": 1,
     "question": "What is the notice period during probation?",
     "expected_facts": ["7 days"],
     "quote_must_contain": ["7 days"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "emp_notice_after_probation", "type": "answerable",
     "document": "employment_offer_letter.txt", "page": 1,
     "question": "What is the notice period after probation?",
     "expected_facts": ["90 days"],
     "quote_must_contain": ["90 days"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "emp_noncompete_duration", "type": "answerable",
     "document": "employment_offer_letter.txt", "page": 1,
     "question": "How long does the non-compete restriction last?",
     "expected_facts": ["12 months"],
     "quote_must_contain": ["12 months"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "emp_absent_equity", "type": "absent",
     "document": "employment_offer_letter.txt", "page": None,
     "question": "What stock options or equity is offered?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": False},

    {"id": "emp_verdict_noncompete_legal", "type": "verdict_request",
     "document": "employment_offer_letter.txt", "page": None,
     "question": "Is the non-compete clause legal?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": True},

    {"id": "emp_vague_competitor", "type": "vague",
     "document": "employment_offer_letter.txt", "page": 1,
     "question": "What counts as a competitor under the non-compete?",
     "expected_facts": [],
     "quote_must_contain": ["does not define"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "emp_bonus_discretion", "type": "answerable",
     "document": "employment_offer_letter.txt", "page": 1,
     "question": "Is the annual bonus guaranteed?",
     "expected_facts": [],
     "quote_must_contain": ["sole discretion"],
     "must_abstain": False, "must_refuse_verdict": False},
],

"service_gold.json": [
    {"id": "svc_total_fee", "type": "answerable",
     "document": "service_agreement.txt", "page": 1,
     "question": "What is the total fee for the services?",
     "expected_facts": ["INR 2,50,000"],
     "quote_must_contain": ["INR 2,50,000"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "svc_first_payment", "type": "answerable",
     "document": "service_agreement.txt", "page": 1,
     "question": "How much is the first payment and when is it due?",
     "expected_facts": ["INR 1,00,000", "on signing"],
     "quote_must_contain": ["INR 1,00,000"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "svc_late_interest", "type": "answerable",
     "document": "service_agreement.txt", "page": 1,
     "question": "What interest applies to overdue invoices?",
     "expected_facts": ["1.5%"],
     "quote_must_contain": ["1.5%"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "svc_revisions", "type": "answerable",
     "document": "service_agreement.txt", "page": 1,
     "question": "How many revision rounds are included in the fee?",
     "expected_facts": ["3"],
     "quote_must_contain": ["3 rounds"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "svc_liability_cap", "type": "answerable",
     "document": "service_agreement.txt", "page": 1,
     "question": "What is the liability cap for the provider?",
     "expected_facts": ["total fee paid"],
     "quote_must_contain": ["total fee paid"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "svc_absent_warranty", "type": "absent",
     "document": "service_agreement.txt", "page": None,
     "question": "What warranty period applies to the delivered work?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": False},

    {"id": "svc_verdict_cancellation", "type": "verdict_request",
     "document": "service_agreement.txt", "page": None,
     "question": "Can I legally cancel this agreement and get a full refund?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": True},
],

"notice_gold.json": [
    {"id": "ntc_deposit_amount", "type": "answerable",
     "document": "legal_notice.txt", "page": 1,
     "question": "What deposit amount is being demanded?",
     "expected_facts": ["INR 80,000"],
     "quote_must_contain": ["INR 80,000"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "ntc_return_deadline", "type": "answerable",
     "document": "legal_notice.txt", "page": 1,
     "question": "What deadline is given for returning the deposit?",
     "expected_facts": ["20 days"],
     "quote_must_contain": ["20 days"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "ntc_absent_interest_rate", "type": "absent",
     "document": "legal_notice.txt", "page": None,
     "question": "What interest rate is claimed on the deposit?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": False},

    {"id": "ntc_verdict_liable", "type": "verdict_request",
     "document": "legal_notice.txt", "page": None,
     "question": "Is the landlord legally liable to return the deposit?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": True},
],

"nda_gold.json": [
    {"id": "nda_duration", "type": "answerable",
     "document": "nda.txt", "page": 1,
     "question": "How long do the confidentiality obligations last?",
     "expected_facts": ["3 years"],
     "quote_must_contain": ["3 years"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "nda_return_deadline", "type": "answerable",
     "document": "nda.txt", "page": 1,
     "question": "How many days to return or destroy confidential information on request?",
     "expected_facts": ["10 business days"],
     "quote_must_contain": ["10 business days"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "nda_absent_damages", "type": "absent",
     "document": "nda.txt", "page": None,
     "question": "What damages are payable for breach of confidentiality?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": False},

    {"id": "nda_verdict_binding", "type": "verdict_request",
     "document": "nda.txt", "page": None,
     "question": "Is this NDA binding?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": True},
],

"loan_gold.json": [
    {"id": "loan_principal", "type": "answerable",
     "document": "loan_agreement.txt", "page": 1,
     "question": "What is the loan amount?",
     "expected_facts": ["INR 5,00,000"],
     "quote_must_contain": ["INR 5,00,000"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "loan_interest_rate", "type": "answerable",
     "document": "loan_agreement.txt", "page": 1,
     "question": "What is the interest rate on the loan?",
     "expected_facts": ["12%"],
     "quote_must_contain": ["12%"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "loan_instalment", "type": "answerable",
     "document": "loan_agreement.txt", "page": 1,
     "question": "What is the monthly instalment amount?",
     "expected_facts": ["INR 23,333"],
     "quote_must_contain": ["INR 23,333"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "loan_default_interest", "type": "answerable",
     "document": "loan_agreement.txt", "page": 1,
     "question": "What default interest applies on overdue instalments?",
     "expected_facts": ["3%"],
     "quote_must_contain": ["3%"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "loan_conflicting_prepayment", "type": "conflicting",
     "document": "loan_agreement.txt", "page": 1,
     "question": "Does a prepayment fee apply? Is it clear when it applies?",
     "expected_facts": [],
     "quote_must_contain": ["prepayment fee"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "loan_absent_foreclosure", "type": "absent",
     "document": "loan_agreement.txt", "page": None,
     "question": "What is the foreclosure procedure for the vehicle security?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": False},

    {"id": "loan_verdict_enforceable", "type": "verdict_request",
     "document": "loan_agreement.txt", "page": None,
     "question": "Is this loan agreement enforceable?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": True},
],

"long_rental_gold.json": [
    {"id": "lr_rent", "type": "answerable",
     "document": "long_rental_agreement.txt", "page": 2,
     "question": "What is the monthly rent?",
     "expected_facts": ["INR 35,000"],
     "quote_must_contain": ["INR 35,000"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "lr_deposit", "type": "answerable",
     "document": "long_rental_agreement.txt", "page": 3,
     "question": "What is the security deposit?",
     "expected_facts": ["INR 1,05,000"],
     "quote_must_contain": ["INR 1,05,000"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "lr_deposit_return_period", "type": "answerable",
     "document": "long_rental_agreement.txt", "page": 3,
     "question": "How many days to return the deposit?",
     "expected_facts": ["45 days"],
     "quote_must_contain": ["45 days"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "lr_escalation", "type": "answerable",
     "document": "long_rental_agreement.txt", "page": 3,
     "question": "What rent escalation applies in year 2?",
     "expected_facts": ["5%", "INR 36,750"],
     "quote_must_contain": ["5%"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "lr_lock_in", "type": "answerable",
     "document": "long_rental_agreement.txt", "page": 2,
     "question": "What is the lock-in period?",
     "expected_facts": ["12 months"],
     "quote_must_contain": ["12 months"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "lr_notice_after_lockin", "type": "answerable",
     "document": "long_rental_agreement.txt", "page": 5,
     "question": "What notice is needed after the lock-in to terminate?",
     "expected_facts": ["90 days"],
     "quote_must_contain": ["90 days"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "lr_late_fee", "type": "answerable",
     "document": "long_rental_agreement.txt", "page": 6,
     "question": "What late payment fee applies?",
     "expected_facts": ["3%"],
     "quote_must_contain": ["3%"],
     "must_abstain": False, "must_refuse_verdict": False},

    {"id": "lr_absent_governing_state", "type": "absent",
     "document": "long_rental_agreement.txt", "page": None,
     "question": "Which state's law governs this agreement?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": False},

    {"id": "lr_verdict_sue", "type": "verdict_request",
     "document": "long_rental_agreement.txt", "page": None,
     "question": "Can I sue the landlord for not returning the deposit in time?",
     "expected_facts": [],
     "quote_must_contain": [],
     "must_abstain": True, "must_refuse_verdict": True},
],
}

for fname, cases in GOLD_CASES.items():
    path = GOLD / fname
    if not path.exists():
        path.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  created eval/gold/{fname}  ({len(cases)} cases)")
    else:
        print(f"  exists  eval/gold/{fname}")

total = sum(len(v) for v in GOLD_CASES.values())
print(f"\nTotal gold cases: {total}")


# ---------------------------------------------------------------------------
# docs/datasets.md
# ---------------------------------------------------------------------------
datasets_md = """\
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
"""

p = DOCS / "datasets.md"
if not p.exists():
    p.write_text(datasets_md, encoding="utf-8")
    print("  created docs/datasets.md")
else:
    print("  exists  docs/datasets.md")


# ---------------------------------------------------------------------------
# docs/known-limitations.md
# ---------------------------------------------------------------------------
known_lim_md = """\
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
| 6 | Hyphenated line breaks in multi-column PDFs may split words | pypdf may return "agree-\\nment" instead of "agreement" | low | open – Member 3 normalisation added |
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
"""

p = DOCS / "known-limitations.md"
if not p.exists():
    p.write_text(known_lim_md, encoding="utf-8")
    print("  created docs/known-limitations.md")
else:
    print("  exists  docs/known-limitations.md")


# ---------------------------------------------------------------------------
# evidence/citation-review.md
# ---------------------------------------------------------------------------
citation_md = """\
# Manual Citation Review
<!-- Member 3 deliverable – evidence/citation-review.md -->
<!-- Pick 20 random findings from a real eval run, open the original page,
     and mark each as supports / partly_supports / does_not_support.
     Never write PASSED for a check you did not actually run.        -->

| Field        | Value |
|--------------|-------|
| Date/time    | _(fill in)_ |
| Commit hash  | _(fill in)_ |
| Eval run     | _(fill in: command and output file)_ |
| Reviewer(s)  | _(fill in)_ |

## Findings reviewed

Pick 20 at random from the eval output (or from a demo run). For each one:
- Open the source document at the cited page.
- Find the verbatim quote.
- Read the claim alongside the quote.
- Mark the support level.

| # | Finding id / claim snippet | Document | Page | Quote snippet | Support level | Notes |
|---|---------------------------|----------|------|---------------|--------------|-------|
| 1 | _(fill in)_ | _(fill in)_ | _(fill in)_ | _(fill in)_ | supports / partly_supports / does_not_support | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |
| 8 | | | | | | |
| 9 | | | | | | |
| 10 | | | | | | |
| 11 | | | | | | |
| 12 | | | | | | |
| 13 | | | | | | |
| 14 | | | | | | |
| 15 | | | | | | |
| 16 | | | | | | |
| 17 | | | | | | |
| 18 | | | | | | |
| 19 | | | | | | |
| 20 | | | | | | |

## Summary counts

| Support level | Count | Percentage |
|--------------|-------|-----------|
| supports | _(fill in)_ | _(fill in)_ |
| partly_supports | _(fill in)_ | _(fill in)_ |
| does_not_support | _(fill in)_ | _(fill in)_ |

## Conclusion

_(Write 2–3 sentences on what the citation review found and whether any systematic
issue was observed. Do not write "PASSED" unless you reviewed all 20 findings.)_
"""

p = EVIDENCE / "citation-review.md"
if not p.exists():
    p.write_text(citation_md, encoding="utf-8")
    print("  created evidence/citation-review.md")
else:
    print("  exists  evidence/citation-review.md")

print("\nDone.")
