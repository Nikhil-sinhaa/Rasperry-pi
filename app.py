from __future__ import annotations

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

CATEGORY_LABELS = {
    "explicit_statement": "What the document explicitly says",
    "potential_concern": "Potential concern to review",
    "ambiguity": "Ambiguous or unclear wording",
    "jurisdiction_dependency": "May depend on jurisdiction",
}

ATTENTION_LABELS = {
    "high": "🔴 High – worth a closer look",
    "medium": "🟡 Medium – worth a closer look",
    "low": "🟢 Low",
}

TOPIC_LABELS = {
    "obligations": "📋 Obligations",
    "money": "💰 Money / Payments",
    "deadlines": "📅 Deadlines / Dates",
    "termination": "🚪 Termination",
    "liability": "⚠️ Liability",
    "jurisdiction": "🌍 Jurisdiction",
    "ambiguity": "❓ Ambiguity",
    "general": "📄 General",
}


@st.cache_resource(show_spinner=False)
def get_runtime() -> LocalOllama:
    return LocalOllama()


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


def render_result(result: dict, title: str = "Analysis") -> None:
    st.markdown(f"**{title}**")
    if result.get("message"):
        if result.get("legal_boundary"):
            st.warning(result["message"])
        elif result.get("abstained"):
            st.info(result["message"])
        else:
            st.caption(result["message"])

    # --- Coverage line (triage only, Member 1) ---
    coverage_line = result.get("coverage_line")
    if coverage_line:
        st.caption(f"📖 {coverage_line}")

    findings = result.get("findings", [])

    # Sort findings: high attention first, then medium, then low (Member 1 schema)
    _attention_order = {"high": 0, "medium": 1, "low": 2}
    findings = sorted(findings, key=lambda f: _attention_order.get(f.get("attention", "low"), 2))

    for number, finding in enumerate(findings, start=1):
        label = CATEGORY_LABELS.get(finding.get("category", ""), "Finding")
        topic = finding.get("topic", "general")
        attention = finding.get("attention", "low")
        topic_label = TOPIC_LABELS.get(topic, topic.title())
        attention_label = ATTENTION_LABELS.get(attention, attention)
        with st.container(border=True):
            col_l, col_r = st.columns([3, 1])
            with col_l:
                st.markdown(f"**{number}. {label}**")
            with col_r:
                st.caption(f"{topic_label}")
            st.caption(attention_label)
            st.write(finding.get("claim", ""))
            evidence = finding.get("evidence", [])
            for citation in evidence:
                source = citation.get("source", {})
                filename = source.get("filename", "uploaded document")
                all_pages = source.get("all_pages", [source.get("page_number", "?")])
                page_str = "+".join(str(p) for p in all_pages)
                st.caption(f"Source: {filename}, page {page_str}")
                # st.code renders user-provided text as inert text rather than Markdown/HTML.
                st.code(citation.get("quote", ""), language=None)
            lawyer_question = finding.get("lawyer_question", "")
            if lawyer_question:
                st.markdown("**Question to discuss with a qualified legal professional**")
                st.write(lawyer_question)

    # --- Jurisdiction hints (Member 1) ---
    jurisdiction_notes = result.get("jurisdiction_notes", [])
    if jurisdiction_notes:
        with st.expander("🌍 Jurisdiction notes"):
            for note in jurisdiction_notes:
                st.info(note)

    sources = result.get("sources", [])
    if sources:
        with st.expander(f"All retrieved passages ({len(sources)})"):
            for source in sources:
                st.markdown(f"**{source.get('source_id')} · {source.get('filename')} · page {source.get('page_number')}**")
                st.caption(f"Retrieval similarity (debug aid, not a confidence score): {source.get('similarity')}")
                st.code(source.get("text", ""), language=None)


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
    st.session_state.messages.append({"role": "assistant", "question": question, "mode": mode, "result": result})
    render_result(result, "Whole-document triage" if mode == "triage" else "Answer")


initialise_state()

st.title("⚖️ Private Legal Document Triage Agent")
st.caption("Local PDF/text ingestion · Local embeddings · Local Gemma · Verifiable passages")
st.warning(
    "For preliminary document understanding only — not legal advice. This app cannot decide legal validity or enforceability. "
    "Important, unclear, or jurisdiction-dependent issues should be reviewed by a qualified legal professional."
)

with st.sidebar:
    st.subheader("Runtime and privacy")
    st.code(
        f"Gemma model: {SETTINGS.chat_model}\n"
        f"Embedding model: {SETTINGS.embedding_model}\n"
        f"Ollama endpoint: {SETTINGS.ollama_base_url}\n"
        f"Min retrieval similarity: {SETTINGS.min_retrieval_similarity}",
        language="text",
    )
    if st.button("Check local models", use_container_width=True):
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
    st.caption("For a privacy claim, also disable Ollama cloud features and produce network-monitoring evidence as described in README.md.")
    st.divider()
    if st.button("Clear documents, vectors, and chat", type="secondary", use_container_width=True):
        old_upload_key = f"uploaded_files_{st.session_state.upload_epoch}"
        st.session_state.local_index = None
        st.session_state.index_metadata = None
        st.session_state.messages = []
        st.session_state.upload_epoch += 1
        st.session_state.model_check = None
        if old_upload_key in st.session_state:
            del st.session_state[old_upload_key]
        st.rerun()

st.subheader("1 · Add a document")
upload_key = f"uploaded_files_{st.session_state.upload_epoch}"
uploaded_files = st.file_uploader(
    "Select text-based PDF or UTF-8 .txt files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key=upload_key,
    help=f"Up to {SETTINGS.max_files} files; {SETTINGS.max_file_mb} MB per file. Scanned PDFs need OCR, which this build does not include.",
)

if st.button("Process documents locally", type="primary", disabled=not uploaded_files, use_container_width=False):
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

st.divider()
st.subheader("2 · Triage or ask a question")
if st.session_state.local_index is None:
    st.info("Process at least one document to enable triage and question answering.")
else:
    # Render the existing transcript before accepting a new action; the newly generated result
    # is displayed once by run_and_store() and appears in the transcript on the next rerun.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message.get("content", ""))
            else:
                mode_title = "Whole-document triage" if message.get("mode") == "triage" else f"Answer to: {message.get('question', '')}"
                render_result(message.get("result", {}), mode_title)

    left, right = st.columns([1, 2])
    with left:
        if st.button("Run whole-document triage", type="primary", use_container_width=True):
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
