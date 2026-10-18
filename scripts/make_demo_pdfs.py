"""Generate demo PDFs for PS2 demonstration and evaluation.

Run from repo root:
    python scripts/make_demo_pdfs.py

Creates samples/pdfs/:
  01_rental_agreement_clear.pdf   – complete, well-drafted rental agreement
  02_employment_offer_letter.pdf  – employment offer with probation + non-compete
  03_nda_with_concerns.pdf        – NDA with one-sided auto-renewal concern
  04_loan_agreement.pdf           – personal loan agreement
  05_document_silent_test.pdf     – document that is SILENT on early-termination notice
                                    (tests that agent says "silent" not invents an answer)

Each PDF is multi-page with real clause text so nomic-embed-text can embed it.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
except ImportError:
    raise SystemExit("Install reportlab first:  pip install reportlab")

OUT_DIR = Path(__file__).resolve().parents[1] / "samples" / "pdfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLES = getSampleStyleSheet()
BODY   = STYLES["BodyText"]
H1     = STYLES["Heading1"]
H2     = STYLES["Heading2"]


def _doc(name: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        str(OUT_DIR / name),
        pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
    )


def _p(text: str, style=None):
    return Paragraph(text, style or BODY)


def _h1(text: str):
    return Paragraph(text, H1)


def _h2(text: str):
    return Paragraph(text, H2)


def _sp():
    return Spacer(1, 0.4*cm)


# ---------------------------------------------------------------------------
# 01 – Rental agreement (clear, well-drafted)
# ---------------------------------------------------------------------------

def make_01_rental_clear():
    doc = _doc("01_rental_agreement_clear.pdf")
    story = [
        _h1("RESIDENTIAL RENTAL AGREEMENT"),
        _p("This Residential Rental Agreement (<b>Agreement</b>) is entered into on "
           "1 October 2026 between:"),
        _p("<b>Landlord:</b> Cedar Lane Properties LLP, registered at 45 MG Road, "
           "Bengaluru 560001 (hereinafter &quot;Landlord&quot;)"),
        _p("<b>Tenant:</b> Aarav Sharma, resident at Flat 4B, Sunrise Apartments, "
           "Bengaluru 560008 (hereinafter &quot;Tenant&quot;)."),
        _sp(),
        _h2("Clause 1 – Premises"),
        _p("The Landlord hereby lets to the Tenant the residential premises at "
           "Flat 4B, Sunrise Apartments, 12 Residency Road, Bengaluru 560025 "
           "(hereinafter &quot;Premises&quot;) on the terms set out in this Agreement."),
        _sp(),
        _h2("Clause 2 – Term"),
        _p("This Agreement commences on 1 October 2026 and shall continue for a "
           "fixed term of twelve (12) months, expiring on 30 September 2027, "
           "unless sooner terminated in accordance with this Agreement."),
        _sp(),
        _h2("Clause 3 – Rent"),
        _p("The Tenant shall pay a monthly rent of <b>INR 20,000</b> (Rupees Twenty "
           "Thousand Only) payable in advance on or before the fifth calendar day of "
           "each calendar month by bank transfer to the Landlord's account (IFSC: "
           "SBIN0001234, Account No. 9876543210)."),
        _sp(),
        _h2("Clause 4 – Security Deposit"),
        _p("Before taking occupation of the Premises, the Tenant shall pay a "
           "refundable security deposit of <b>INR 40,000</b> (Rupees Forty Thousand Only). "
           "The deposit will be returned within 30 days after the Tenant has vacated "
           "the Premises and handed over the keys, subject to deductions for unpaid "
           "rent, damage beyond normal wear and tear, or outstanding utility charges."),
        _sp(),
        _h2("Clause 5 – Late Payment"),
        _p("If the monthly rent is not received by the due date, the Tenant shall "
           "pay a late fee of 2% of the overdue rent for each month or part of a "
           "month the payment remains outstanding."),
        _sp(),
        _h2("Clause 6 – Use of Premises"),
        _p("The Premises shall be used solely for residential purposes. The Tenant "
           "shall not carry on any commercial activity or permit any nuisance."),
        _sp(),
        _h2("Clause 7 – Subletting"),
        _p("The Tenant shall not sublet, underlet, or otherwise assign or transfer "
           "any interest in the Premises without the prior written consent of the "
           "Landlord, which may be withheld at the Landlord's discretion."),
        _sp(),
        _h2("Clause 8 – Notice to Terminate"),
        _p("Either party may terminate this Agreement on expiry of the fixed term or "
           "thereafter by giving <b>60 days' written notice</b> to the other party. Notice "
           "must be delivered by registered post or by hand to the address stated in "
           "this Agreement."),
        _sp(),
        _h2("Clause 9 – Repairs and Maintenance"),
        _p("The Landlord shall be responsible for structural repairs and maintenance "
           "of the building fabric. The Tenant shall keep the interior of the Premises "
           "clean and in good condition, and shall promptly report any damage or "
           "disrepair to the Landlord in writing."),
        _sp(),
        _h2("Clause 10 – Landlord's Right of Entry"),
        _p("The Landlord or its authorised agent may enter the Premises at reasonable "
           "times with at least 24 hours' prior written notice to carry out inspections, "
           "repairs, or to show the Premises to prospective tenants during the last "
           "two months of the term."),
        _sp(),
        _h2("Clause 11 – Tenant Default"),
        _p("If the Tenant fails to pay rent for more than 10 days after the due date, "
           "breaches any material term of this Agreement, or vacates the Premises "
           "without notice, the Landlord may serve a written notice of breach. "
           "The Tenant then has 15 days to remedy the breach. If the breach is not "
           "remedied, the Landlord may terminate the Agreement."),
        _sp(),
        _h2("Clause 12 – Governing Law"),
        _p("This Agreement is governed by the laws of the State of Karnataka and "
           "subject to the jurisdiction of courts in Bengaluru."),
        _sp(),
        _p("<b>IN WITNESS WHEREOF</b> the parties have executed this Agreement on the "
           "date first written above."),
        _sp(),
        _p("______________________________&nbsp;&nbsp;&nbsp; ______________________________"),
        _p("Landlord (Cedar Lane Properties LLP)&nbsp;&nbsp;&nbsp; Tenant (Aarav Sharma)"),
    ]
    doc.build(story)
    print(f"Created: {OUT_DIR}/01_rental_agreement_clear.pdf")


# ---------------------------------------------------------------------------
# 02 – Employment offer letter
# ---------------------------------------------------------------------------

def make_02_employment():
    doc = _doc("02_employment_offer_letter.pdf")
    story = [
        _h1("LETTER OF APPOINTMENT"),
        _p("Date: 5 October 2026"),
        _sp(),
        _p("<b>To:</b><br/>Priya Nair<br/>Flat 7, Lotus Colony<br/>Pune 411001"),
        _sp(),
        _p("Dear Priya,"),
        _p("We are pleased to offer you the position of <b>Senior Software Engineer</b> "
           "at Nexwave Technologies Pvt. Ltd. (hereinafter &quot;Company&quot;). "
           "This letter sets out the key terms of your employment."),
        _sp(),
        _h2("1. Commencement Date"),
        _p("Your employment commences on <b>1 November 2026</b>, subject to successful "
           "completion of background verification."),
        _sp(),
        _h2("2. Compensation"),
        _p("Your gross annual compensation (CTC) is <b>INR 18,00,000</b> (Rupees Eighteen "
           "Lakh Only) per annum, paid monthly as INR 1,50,000 per month, subject to "
           "applicable statutory deductions (TDS, PF, ESI)."),
        _sp(),
        _h2("3. Probationary Period"),
        _p("You will be on probation for the first <b>six months</b> of employment. "
           "During probation, either party may terminate employment by giving "
           "<b>7 days' written notice</b> (or payment in lieu)."),
        _sp(),
        _h2("4. Confirmation and Notice Period Post-Probation"),
        _p("On satisfactory completion of the probationary period, you will be confirmed "
           "as a permanent employee. Thereafter, the notice period for termination by "
           "either party is <b>90 days</b>. The Company reserves the right to make "
           "payment in lieu of notice."),
        _sp(),
        _h2("5. Confidentiality"),
        _p("During and after employment, you shall keep all Company trade secrets, "
           "customer data, source code, and proprietary information strictly confidential. "
           "This obligation survives termination of employment indefinitely."),
        _sp(),
        _h2("6. Non-Compete"),
        _p("For a period of <b>12 months</b> following separation, you shall not, "
           "without written consent, join or assist any direct competitor of the Company "
           "operating in the same product segment in India."),
        _sp(),
        _h2("7. Intellectual Property"),
        _p("All work product, inventions, and code created during employment shall be "
           "the exclusive property of the Company. You assign all related intellectual "
           "property rights to the Company."),
        _sp(),
        _h2("8. Governing Law"),
        _p("This offer letter is governed by the laws of India and subject to the "
           "jurisdiction of courts in Pune, Maharashtra."),
        _sp(),
        _p("Please sign and return a copy of this letter by <b>10 October 2026</b> "
           "to confirm your acceptance."),
        _sp(),
        _p("Yours sincerely,"),
        _sp(),
        _p("______________________________"),
        _p("Director – Human Resources<br/>Nexwave Technologies Pvt. Ltd."),
        _sp(),
        _p("<b>ACCEPTANCE:</b> I, Priya Nair, accept the offer on the terms above."),
        _p("Signature: ______________________________&nbsp;&nbsp; Date: __________"),
    ]
    doc.build(story)
    print(f"Created: {OUT_DIR}/02_employment_offer_letter.pdf")


# ---------------------------------------------------------------------------
# 03 – NDA with a concern
# ---------------------------------------------------------------------------

def make_03_nda():
    doc = _doc("03_nda_with_concerns.pdf")
    story = [
        _h1("NON-DISCLOSURE AGREEMENT"),
        _p("This Non-Disclosure Agreement (<b>Agreement</b>) is entered into on "
           "15 October 2026 between:"),
        _p("<b>Disclosing Party:</b> Innovex Solutions Pvt. Ltd., Mumbai."),
        _p("<b>Receiving Party:</b> Ravi Mehta, Consultant."),
        _sp(),
        _h2("1. Definition of Confidential Information"),
        _p("&quot;Confidential Information&quot; means any and all information, data, "
           "know-how, trade secrets, financial data, business plans, software, customer "
           "lists, and any other proprietary information disclosed by the Disclosing "
           "Party to the Receiving Party, whether disclosed in writing, orally, or "
           "by any other means."),
        _sp(),
        _h2("2. Obligations of the Receiving Party"),
        _p("The Receiving Party shall: (a) hold all Confidential Information in strict "
           "confidence; (b) not disclose any Confidential Information to any third "
           "party without prior written consent of the Disclosing Party; (c) use "
           "Confidential Information solely for the purpose of evaluating a potential "
           "business collaboration."),
        _sp(),
        _h2("3. Exclusions"),
        _p("Obligations under Clause 2 do not apply to information that: (a) is or "
           "becomes publicly available through no fault of the Receiving Party; "
           "(b) was already known to the Receiving Party before disclosure; "
           "(c) is independently developed by the Receiving Party without use of "
           "Confidential Information; or (d) must be disclosed by law or court order."),
        _sp(),
        _h2("4. Duration"),
        _p("This Agreement and the obligations of confidentiality shall remain in "
           "force for a period of <b>three (3) years</b> from the date of this Agreement. "
           "However, this Agreement shall <b>automatically renew</b> for successive one-year "
           "periods unless either party gives 90 days' written notice of non-renewal "
           "before the expiry of any term. Note: automatic renewal is triggered silently "
           "and may extend obligations beyond the initial three years."),
        _sp(),
        _h2("5. Return of Information"),
        _p("On written request, the Receiving Party shall promptly return or destroy "
           "all Confidential Information and any copies thereof."),
        _sp(),
        _h2("6. No Licence"),
        _p("Nothing in this Agreement grants the Receiving Party any licence, right, "
           "title, or interest in or to any Confidential Information."),
        _sp(),
        _h2("7. Governing Law"),
        _p("This Agreement is governed by the laws of India and subject to the "
           "exclusive jurisdiction of courts in Mumbai, Maharashtra."),
        _sp(),
        _p("<b>SIGNED</b> by the parties on the date first written above."),
        _sp(),
        _p("______________________________&nbsp;&nbsp;&nbsp; ______________________________"),
        _p("Innovex Solutions Pvt. Ltd.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Ravi Mehta"),
    ]
    doc.build(story)
    print(f"Created: {OUT_DIR}/03_nda_with_concerns.pdf")


# ---------------------------------------------------------------------------
# 04 – Loan agreement
# ---------------------------------------------------------------------------

def make_04_loan():
    doc = _doc("04_loan_agreement.pdf")
    story = [
        _h1("PERSONAL LOAN AGREEMENT"),
        _p("This Loan Agreement (<b>Agreement</b>) is made on 1 October 2026 between:"),
        _p("<b>Lender:</b> Sunita Rao, residing at 88 Park Avenue, Chennai 600018."),
        _p("<b>Borrower:</b> Karan Bose, residing at 22 Anna Nagar, Chennai 600040."),
        _sp(),
        _h2("1. Loan Amount"),
        _p("The Lender agrees to lend the Borrower a principal sum of "
           "<b>INR 5,00,000</b> (Rupees Five Lakh Only), disbursed in full on "
           "1 October 2026."),
        _sp(),
        _h2("2. Interest Rate"),
        _p("The loan shall carry interest at the rate of <b>12% per annum</b> "
           "(simple interest) on the outstanding principal, calculated monthly."),
        _sp(),
        _h2("3. Repayment Schedule"),
        _p("The Borrower shall repay the loan in <b>24 equal monthly instalments</b> "
           "of INR 23,537 each (principal + interest), payable on the first day of "
           "each calendar month commencing 1 November 2026 and ending 1 October 2028."),
        _sp(),
        _h2("4. Prepayment"),
        _p("The Borrower may prepay the outstanding principal in full or in part at "
           "any time without penalty, provided 15 days' prior written notice is given "
           "to the Lender. Prepayment interest is calculated only on the outstanding "
           "balance up to the date of prepayment."),
        _sp(),
        _h2("5. Default"),
        _p("If the Borrower fails to pay any instalment within 10 days of its due date, "
           "or commits any material breach of this Agreement, the entire outstanding "
           "principal and accrued interest shall immediately become due and payable "
           "(<b>acceleration</b>). The default interest rate shall be "
           "<b>18% per annum</b> on the overdue amount."),
        _sp(),
        _h2("6. Security"),
        _p("The Borrower shall provide a post-dated cheque for each instalment. "
           "No other security is required under this Agreement."),
        _sp(),
        _h2("7. Governing Law"),
        _p("This Agreement is governed by the laws of India. Disputes shall be "
           "submitted to the exclusive jurisdiction of courts in Chennai, Tamil Nadu."),
        _sp(),
        _p("Signed by both parties:"),
        _sp(),
        _p("______________________________&nbsp;&nbsp;&nbsp; ______________________________"),
        _p("Lender (Sunita Rao)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
           "Borrower (Karan Bose)"),
    ]
    doc.build(story)
    print(f"Created: {OUT_DIR}/04_loan_agreement.pdf")


# ---------------------------------------------------------------------------
# 05 – Silent document (no early termination clause)
# ---------------------------------------------------------------------------

def make_05_silent():
    doc = _doc("05_document_silent_test.pdf")
    story = [
        _h1("SERVICE AGREEMENT"),
        _p("This Service Agreement (<b>Agreement</b>) is made on 1 October 2026 between:"),
        _p("<b>Service Provider:</b> TechServe Solutions, Hyderabad."),
        _p("<b>Client:</b> BlueStar Retail Pvt. Ltd., Hyderabad."),
        _sp(),
        _h2("1. Scope of Services"),
        _p("The Service Provider agrees to deliver software maintenance and support "
           "services as described in Schedule A attached hereto. The scope includes "
           "bug fixes, security patches, and feature enhancements approved by the "
           "Client in writing."),
        _sp(),
        _h2("2. Term"),
        _p("This Agreement commences on 1 October 2026 and continues for a period "
           "of twelve (12) months unless earlier terminated under Clause 3."),
        _sp(),
        _h2("3. Termination for Cause"),
        _p("Either party may terminate this Agreement immediately on written notice "
           "if the other party commits a material breach and fails to remedy it within "
           "30 days of written notice of the breach."),
        _sp(),
        _h2("4. Payment"),
        _p("The Client shall pay a monthly retainer of <b>INR 75,000</b> (Rupees "
           "Seventy-Five Thousand Only) payable within 15 days of receipt of invoice. "
           "Late payments attract interest at 1.5% per month."),
        _sp(),
        _h2("5. Intellectual Property"),
        _p("All deliverables produced by the Service Provider under this Agreement "
           "shall be owned by the Client on full payment. The Service Provider "
           "retains ownership of its pre-existing tools and frameworks."),
        _sp(),
        _h2("6. Confidentiality"),
        _p("Each party shall keep the other party's confidential information strictly "
           "confidential and shall not disclose it to any third party. This obligation "
           "continues for 2 years after termination of this Agreement."),
        _sp(),
        _h2("7. Limitation of Liability"),
        _p("The Service Provider's aggregate liability under this Agreement shall not "
           "exceed the total fees paid by the Client in the 3 months immediately "
           "preceding the event giving rise to the claim."),
        _sp(),
        _h2("8. Governing Law"),
        _p("This Agreement is governed by the laws of India and the parties submit "
           "to the exclusive jurisdiction of courts in Hyderabad, Telangana."),
        _sp(),
        _p("<b>NOTE FOR EVALUATORS:</b> This document deliberately omits any early "
           "termination for convenience clause. An agent asked 'What is the early "
           "termination notice period?' should respond that the document is silent on "
           "this, not invent an answer."),
        _sp(),
        _p("Signed by both parties:"),
        _sp(),
        _p("______________________________&nbsp;&nbsp;&nbsp; ______________________________"),
        _p("TechServe Solutions&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
           "BlueStar Retail Pvt. Ltd."),
    ]
    doc.build(story)
    print(f"Created: {OUT_DIR}/05_document_silent_test.pdf")


if __name__ == "__main__":
    make_01_rental_clear()
    make_02_employment()
    make_03_nda()
    make_04_loan()
    make_05_silent()
    print(f"\nAll 5 demo PDFs written to {OUT_DIR}")
    print("\nRecommended test sequence:")
    print("  01_rental_agreement_clear.pdf → Ask: 'What is the notice period?'")
    print("    Expected: 60 days written notice, cite Clause 8, show exact source passage.")
    print("  05_document_silent_test.pdf   → Ask: 'What is the early termination notice period?'")
    print("    Expected: Document is silent — agent should NOT invent an answer.")
