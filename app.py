from __future__ import annotations

import textwrap
from collections import defaultdict
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Load settings before importing modules that read environment variables.
load_dotenv()

from legal_triage.agent import AnalysisError, analyze  # noqa: E402
from legal_triage.config import SETTINGS  # noqa: E402
from legal_triage.ingestion import IngestionError  # noqa: E402
from legal_triage.local_runtime import LocalModelError, LocalOllama  # noqa: E402
from legal_triage.retrieval import LocalIndex, build_index  # noqa: E402


st.set_page_config(
    page_title="Private Legal Document Triage",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

CATEGORY_LABELS = {
    "explicit_statement": "What the document explicitly says",
    "potential_concern": "Potential concern to review",
    "ambiguity": "Ambiguous or unclear wording",
    "jurisdiction_dependency": "May depend on jurisdiction",
}

# Text label + colour used for attention (Member 4: show as text label AND colour)
ATTENTION_CONFIG = {
    "high":   {"label": "🔴 High attention",   "color": "#c0392b"},
    "medium": {"label": "🟡 Medium attention", "color": "#d68910"},
    "low":    {"label": "🟢 Low attention",    "color": "#1e8449"},
}

TOPIC_LABELS = {
    "obligations": "📋 Obligations",
    "money":       "💰 Money / Payments",
    "deadlines":   "📅 Deadlines / Dates",
    "termination": "🚪 Termination",
    "liability":   "⚠️ Liability",
    "jurisdiction":"🌍 Jurisdiction",
    "ambiguity":   "❓ Ambiguity",
    "general":     "📄 General",
}

# Topics that surface in the key-facts card
KEY_FACTS_TOPICS = {"money", "deadlines", "termination", "obligations"}

_ATTENTION_ORDER = {"high": 0, "medium": 1, "low": 2}

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_runtime() -> LocalOllama:
    return LocalOllama()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def initialise_state() -> None:
    defaults = {
        "local_index": None,
        "index_metadata": None,
        "messages": [],
        "upload_epoch": 0,
        "model_check": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Markdown export helper (Member 4: download button)
# ---------------------------------------------------------------------------


def _result_to_markdown(result: dict, title: str, question: str = "") -> str:
    lines: list[str] = [f"# {title}", ""]
    if question:
        lines += [f"**Question:** {question}", ""]
    coverage = result.get("coverage_line")
    if coverage:
        lines += [f"*{coverage}*", ""]
    if result.get("legal_boundary"):
        lines += ["> ⚠️ **Legal boundary:** " + result.get("message", ""), ""]
        lq = result.get("lawyer_question", "")
        if lq:
            lines += [f"> **Suggested question for a qualified legal professional:** {lq}", ""]
        return "\n".join(lines)
    if result.get("abstained"):
        lines += ["> ℹ️ " + result.get("message", ""), ""]
        return "\n".join(lines)

    findings = result.get("findings", [])
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        by_topic[f.get("topic", "general")].append(f)

    for topic, topic_findings in by_topic.items():
        topic_label = TOPIC_LABELS.get(topic, topic.title())
        lines += [f"## {topic_label}", ""]
        for f in sorted(topic_findings, key=lambda x: _ATTENTION_ORDER.get(x.get("attention", "low"), 2)):
            attn = f.get("attention", "low")
            attn_label = ATTENTION_CONFIG.get(attn, {}).get("label", attn)
            cat_label = CATEGORY_LABELS.get(f.get("category", ""), "Finding")
            lines += [f"### {cat_label}  ·  {attn_label}", ""]
            lines.append(f.get("claim", ""))
            lines.append("")
            for ev in f.get("evidence", []):
                src = ev.get("source", {})
                fname = src.get("filename", "document")
                pages = src.get("all_pages", [src.get("page_number", "?")])
                page_str = "+".join(str(p) for p in pages)
                lines += [f"> **Source:** {fname}, page {page_str}", f"> ```", f"> {ev.get('quote', '')}", "> ```", ""]
            lq = f.get("lawyer_question", "")
            if lq:
                lines += [f"**Question for a qualified legal professional:** {lq}", ""]

    jn = result.get("jurisdiction_notes", [])
    if jn:
        lines += ["## 🌍 Jurisdiction notes", ""]
        for note in jn:
            lines += [f"- {note}"]
        lines.append("")

    lines += [
        "---",
        "*Generated by Private Legal Document Triage Agent. "
        "This is a preliminary document aid, not legal advice.*",
        f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Key-facts card (Member 4)
# ---------------------------------------------------------------------------


def _render_key_facts_card(findings: list[dict]) -> None:
    """Show a compact card of validated facts for money, deadlines, termination, obligations."""
    key_facts = [
        f for f in findings
        if f.get("topic") in KEY_FACTS_TOPICS and f.get("category") == "explicit_statement"
    ]
    if not key_facts:
        return
    st.markdown("#### 📌 Key facts at a glance")
    cols = st.columns(min(len(key_facts), 3))
    for i, fact in enumerate(key_facts[:6]):
        topic = fact.get("topic", "general")
        icon = TOPIC_LABELS.get(topic, "📄").split()[0]
        claim = fact.get("claim", "")
        short = textwrap.shorten(claim, width=120, placeholder="…")
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"{icon} **{TOPIC_LABELS.get(topic, topic.title()).split(' ', 1)[-1].strip()}**")
                st.caption(short)


# ---------------------------------------------------------------------------
# Core result renderer (Member 4 overhaul)
# ---------------------------------------------------------------------------


def render_result(result: dict, title: str = "Analysis", question: str = "") -> None:
    # --- Header row: title + download button ---
    col_title, col_dl = st.columns([4, 1])
    with col_title:
        st.markdown(f"**{title}**")
    with col_dl:
        md_bytes = _result_to_markdown(result, title, question).encode()
        st.download_button(
            label="⬇ Download .md",
            data=md_bytes,
            file_name="legal-triage-result.md",
            mime="text/markdown",
            use_container_width=True,
            help="Save this result as a Markdown file to your own disk",
        )

    # --- Coverage line (triage only) ---
    coverage_line = result.get("coverage_line")
    if coverage_line:
        st.caption(f"📖 {coverage_line}")

    # --- Legal boundary (refusal) ---
    if result.get("legal_boundary"):
        st.warning(result.get("message", ""))
        lq = result.get("lawyer_question", "")
        if lq:
            with st.container(border=True):
                st.markdown("**💬 Suggested question for a qualified legal professional or local legal-aid service:**")
                st.info(f'"{lq}"')
        return

    # --- Abstention ---
    if result.get("abstained"):
        st.info(result.get("message", ""))
        return

    if result.get("message"):
        st.caption(result["message"])

    findings = result.get("findings", [])
    if not findings:
        return

    # --- Key-facts card ---
    _render_key_facts_card(findings)

    # --- Group by topic, sort each group by attention ---
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        by_topic[f.get("topic", "general")].append(f)

    # Render topic sections in a fixed order
    topic_order = list(TOPIC_LABELS.keys())
    extra_topics = sorted(k for k in by_topic if k not in topic_order)

    for topic in topic_order + extra_topics:
        if topic not in by_topic:
            continue
        topic_findings = sorted(
            by_topic[topic],
            key=lambda f: _ATTENTION_ORDER.get(f.get("attention", "low"), 2),
        )
        topic_label = TOPIC_LABELS.get(topic, f"📄 {topic.title()}")
        with st.expander(f"{topic_label}  ·  {len(topic_findings)} finding(s)", expanded=(topic in {"obligations", "money", "deadlines"})):
            for number, finding in enumerate(topic_findings, start=1):
                cat_label = CATEGORY_LABELS.get(finding.get("category", ""), "Finding")
                attention = finding.get("attention", "low")
                attn_cfg = ATTENTION_CONFIG.get(attention, ATTENTION_CONFIG["low"])

                with st.container(border=True):
                    col_l, col_r = st.columns([3, 1])
                    with col_l:
                        st.markdown(f"**{number}. {cat_label}**")
                    with col_r:
                        # Attention as coloured text label (Member 4 requirement)
                        st.markdown(
                            f"<span style='color:{attn_cfg['color']};font-weight:600'>{attn_cfg['label']}</span>",
                            unsafe_allow_html=True,
                        )
                    st.write(finding.get("claim", ""))

                    evidence = finding.get("evidence", [])
                    for citation in evidence:
                        source = citation.get("source", {})
                        filename = source.get("filename", "uploaded document")
                        all_pages = source.get("all_pages", [source.get("page_number", "?")])
                        page_str = "+".join(str(p) for p in all_pages)
                        st.caption(f"Source: **{filename}**, page **{page_str}**")
                        # Render quote; st.code keeps it inert (no HTML/Markdown injection)
                        st.code(citation.get("quote", ""), language=None)

                    lawyer_question = finding.get("lawyer_question", "")
                    if lawyer_question:
                        st.markdown("**💬 Question to discuss with a qualified legal professional**")
                        st.info(lawyer_question)

    # --- Jurisdiction hints ---
    jurisdiction_notes = result.get("jurisdiction_notes", [])
    if jurisdiction_notes:
        with st.expander("🌍 Jurisdiction notes"):
            for note in jurisdiction_notes:
                st.info(note)

    # --- Retrieved passages (debug) ---
    sources = result.get("sources", [])
    if sources:
        with st.expander(f"All retrieved passages ({len(sources)})"):
            for source in sources:
                st.markdown(
                    f"**{source.get('source_id')} · {source.get('filename')} · page {source.get('page_number')}**"
                )
                st.caption(
                    f"Retrieval similarity (debug aid, not a confidence score): {source.get('similarity')}"
                )
                st.code(source.get("text", ""), language=None)


# ---------------------------------------------------------------------------
# Run analysis and store in session
# ---------------------------------------------------------------------------


def run_and_store(question: str, mode: str) -> None:
    runtime = get_runtime()
    index: LocalIndex = st.session_state.local_index
    try:
        with st.spinner("Searching local embeddings and analyzing with local Gemma…"):
            result = analyze(question, index, runtime, mode=mode)
    except (LocalModelError, AnalysisError, ValueError, RuntimeError) as exc:
        # Error messages intentionally exclude raw prompts, document excerpts and model output.
        st.error(str(exc))
        return
    st.session_state.messages.append(
        {"role": "assistant", "question": question, "mode": mode, "result": result}
    )
    render_result(result, "Whole-document triage" if mode == "triage" else "Answer", question=question)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

initialise_state()

st.title("⚖️ Private Legal Document Triage Agent")
st.caption("Local PDF/text ingestion · Local embeddings · Local Gemma · Verifiable passages")
st.warning(
    "For preliminary document understanding only — not legal advice. This app cannot decide legal validity or enforceability. "
    "Important, unclear, or jurisdiction-dependent issues should be reviewed by a qualified legal professional."
)

# --- Sidebar ---
with st.sidebar:
    st.subheader("Runtime and privacy")
    st.code(
        f"Gemma model: {SETTINGS.chat_model}\n"
        f"Embedding model: {SETTINGS.embedding_model}\n"
        f"Ollama endpoint: {SETTINGS.ollama_base_url}\n"
        f"Min retrieval similarity: {SETTINGS.min_retrieval_similarity}",
        language="text",
    )
    if st.button("Check local models", use_container_width=True, key="check_models_btn"):
        try:
            details = get_runtime().verify_models()
            st.session_state.model_check = (True, details)
        except (LocalModelError, ValueError) as exc:
            st.session_state.model_check = (False, str(exc))
    if st.session_state.model_check:
        ok, details = st.session_state.model_check
        if ok:
            st.success(f"Local models found: {details['chat_model']} and {details['embedding_model']}.")
        else:
            st.error(details)

    st.divider()
    st.markdown("**Privacy boundaries**")
    st.write(
        "The app accepts only a loopback Ollama URL and disables inherited HTTP proxy settings. "
        "Uploaded files, extracted text, vectors, and chat state stay in this app process's session memory. "
        "Do not expose Streamlit or Ollama to a network."
    )
    st.caption(
        "For a privacy claim, also disable Ollama cloud features and produce network-monitoring evidence as described in README.md."
    )
    st.divider()
    if st.button("Clear documents, vectors, and chat", type="secondary", use_container_width=True, key="clear_btn"):
        old_upload_key = f"uploaded_files_{st.session_state.upload_epoch}"
        st.session_state.local_index = None
        st.session_state.index_metadata = None
        st.session_state.messages = []
        st.session_state.upload_epoch += 1
        st.session_state.model_check = None
        if old_upload_key in st.session_state:
            del st.session_state[old_upload_key]
        st.rerun()

# --- Step 1: upload ---
st.subheader("1 · Add a document")
upload_key = f"uploaded_files_{st.session_state.upload_epoch}"
uploaded_files = st.file_uploader(
    "Select text-based PDF or UTF-8 .txt files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key=upload_key,
    help=(
        f"Up to {SETTINGS.max_files} files; {SETTINGS.max_file_mb} MB per file. "
        "Scanned PDFs need OCR, which this build does not include."
    ),
)

if st.button(
    "Process documents locally",
    type="primary",
    disabled=not uploaded_files,
    use_container_width=False,
    key="process_btn",
):
    try:
        runtime = get_runtime()
        with st.spinner("Checking local models, extracting text in memory, and building local embeddings…"):
            runtime.verify_models()
            file_payloads = [(file.name, file.getvalue()) for file in uploaded_files]
            new_index = build_index(file_payloads, runtime)
        st.session_state.local_index = new_index
        st.session_state.index_metadata = new_index.metadata
        st.session_state.messages = []
        st.success("Index created in memory. No uploaded file or vector index was saved by this app to disk.")
    except (IngestionError, LocalModelError, ValueError, RuntimeError) as exc:
        st.error(str(exc))

if st.session_state.index_metadata:
    meta = st.session_state.index_metadata
    st.success(
        f"Indexed {len(meta['files'])} file(s), {meta['pages']} total page(s) "
        f"({meta.get('text_pages', meta['pages'])} with extractable text), and {meta['chunks']} chunks."
    )
    st.caption("Files: " + ", ".join(meta["files"]))
    for warning in meta.get("warnings", []):
        st.warning(warning)

# --- Step 2: triage / question ---
st.divider()
st.subheader("2 · Triage or ask a question")
if st.session_state.local_index is None:
    st.info("Process at least one document to enable triage and question answering.")
else:
    # Replay transcript
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message.get("content", ""))
            else:
                mode_title = (
                    "Whole-document triage"
                    if message.get("mode") == "triage"
                    else f"Answer to: {message.get('question', '')}"
                )
                render_result(message.get("result", {}), mode_title, question=message.get("question", ""))

    left, right = st.columns([1, 2])
    with left:
        if st.button("Run whole-document triage", type="primary", use_container_width=True, key="triage_btn"):
            run_and_store(
                "Summarize important clauses, obligations, rights, deadlines, money/penalties, termination terms, ambiguities and potential concerns. Cite exact passages.",
                "triage",
            )
    with right:
        st.caption("The triage view is limited to passages retrieved by the local search. Review the full document too.")

    question = st.chat_input("Ask about a clause, obligation, deadline, penalty, or potential concern…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            run_and_store(question, "question")
