# Security and privacy reporting

This project is a hackathon prototype for preliminary legal-document understanding, not an independently audited secure system.

Do not report a vulnerability by attaching a real contract, legal notice, identity document, or any other sensitive file. Instead, provide a synthetic minimal reproduction, the steps taken, relevant version numbers, and redacted logs. Do not include API keys, passwords, access tokens, or model outputs containing personal data.

The project maintainers should review suspected privacy-boundary failures privately before public disclosure. For a public student/hackathon repository, replace this generic guidance with the team's actual contact and response process before publication.

## Known limitations

- Uploaded content and derived vectors/quotes exist in the running process's memory and are not encrypted at the application layer.
- The tool does not include OCR, malware scanning, sandboxing of PDFs, encrypted persistent storage, access-control management, or multi-user isolation beyond the local Streamlit session.
- Verbatim quote checks prove that a quote occurs in a retrieved passage; they do not prove semantic correctness of the associated interpretation.
- System-level malware, compromised dependencies/runtime, privileged host processes, browser extensions, or a machine that is not actually offline are outside this prototype's guarantees.
