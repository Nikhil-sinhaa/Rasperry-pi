# Usability Test Evidence

> **Owner:** Member 4  
> **Requirement:** Ask two non-lawyer friends to read one result and explain the clause back to you.
> Keep their notes here as usability evidence (accessibility criterion C6).

---

## Test Protocol

Each participant:
1. Opens the app at `http://127.0.0.1:8501` (no prior briefing on how it works).
2. Uploads `samples/demo_rental_agreement.txt` and clicks **Process documents locally**.
3. Clicks **Run whole-document triage**.
4. Reads the first three findings silently for 2 minutes.
5. Explains aloud what the document says about **rent** and **deposit return**.
6. Answers two follow-up questions (see below).

**Follow-up questions:**
- "Would you know where to find the exact clause this came from?"
- "Is there anything on the screen that confused you or felt unclear?"

---

## Participant 1

| Field | Value |
|-------|-------|
| Date | TODO |
| Tester name / initials | TODO |
| Legal background | None / some / legal professional |
| App version (commit) | TODO (`git rev-parse --short HEAD`) |

**Explanation of rent clause (their words):**

> TODO – quote what they said verbatim

**Explanation of deposit return clause (their words):**

> TODO

**Follow-up Q1 (find the clause):**

> TODO

**Follow-up Q2 (confusion / unclear):**

> TODO

**Facilitator notes:**

> TODO

---

## Participant 2

| Field | Value |
|-------|-------|
| Date | TODO |
| Tester name / initials | TODO |
| Legal background | None / some / legal professional |
| App version (commit) | TODO |

**Explanation of rent clause (their words):**

> TODO

**Explanation of deposit return clause (their words):**

> TODO

**Follow-up Q1 (find the clause):**

> TODO

**Follow-up Q2 (confusion / unclear):**

> TODO

**Facilitator notes:**

> TODO

---

## Accessibility Checks

Run these before the demo freeze and record results below.

| Check | Tool / Method | Result |
|-------|--------------|--------|
| Keyboard-only navigation | Tab through all controls; no mouse | TODO |
| Colour contrast (WCAG AA) | Browser DevTools / axe-core | TODO |
| 200% zoom | Browser zoom; check no content is cut off | TODO |
| Screen-reader labels | axe-core against `http://127.0.0.1:8501` | TODO |

**axe-core command (install once):**
```bash
npx axe-core-cli http://127.0.0.1:8501 --exit
```

**Violations found:**

> TODO – paste axe-core output summary here

---

## Summary

> TODO – one paragraph summarising what the two participants understood correctly,
> what confused them, and what was changed as a result.
