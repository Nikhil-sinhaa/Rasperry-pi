---
name: Feature request / good first issue
about: Suggest an improvement or new feature
labels: enhancement
---

## Problem / motivation

<!-- What problem does this solve, or what user need does it address? -->

## Proposed solution

<!-- Describe the change you'd like. Keep it concrete. -->

## Alternatives considered

<!-- Any alternative approaches? Why is your proposal better? -->

## Privacy and safety impact

<!-- Does this change affect data egress, Ollama endpoint handling, quote validation,
or the legal-verdict filter? If so, describe the impact and how you will test it. -->

## Checklist for contributors

- [ ] I have read `CONTRIBUTING.md`
- [ ] This does not introduce external API calls or cloud services
- [ ] I will add or update tests for any new logic
- [ ] I will run `pytest -q` and `python scripts/static_privacy_audit.py` before opening a PR
