from __future__ import annotations

import textwrap
from collections import defaultdict
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Load settings before importing modules that read environment variables.
load_dotenv()

from legal_triage.agent import AnalysisError, analyze  # noqa: E402
from legal_triage.checklist import (  # noqa: E402
    ABSENT,
    POSSIBLY_PRESENT,
    PRESENT,
    ClauseVerdict,
    detect_document_type,
    extract_structured_fields,
    missing_clause_checklist,
)
from legal_triage.config import SETTINGS  # noqa: E402
from legal_triage.history import (  # noqa: E402
    delete_session,
    list_sessions,
    load_session,
    make_session_id,
    save_session,
)
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
        # history: current session ID assigned when documents are processed
        "current_session_id": None,
        "current_session_created": None,
        # flag to suppress auto-save during a history replay load
        "_replaying": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Markdown export helper
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
# Key-facts card
# ---------------------------------------------------------------------------


def _render_key_facts_card(findings: list[dict]) -> None:
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
# Core result renderer
# ---------------------------------------------------------------------------


def render_result(
    result: dict,
    title: str = "Analysis",
    question: str = "",
    *,
    download_key: str,
) -> None:
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
            key=download_key,
        )

    coverage_line = result.get("coverage_line")
    if coverage_line:
        st.caption(f"📖 {coverage_line}")

    # --- Legal boundary (refusal) ---
    if result.get("legal_boundary"):
        st.warning(result.get("message", ""))
        lq = result.get("lawyer_question", "")
        # Prominent escalation action box (L8)
        with st.container(border=True):
            st.markdown(
                "🚨 **This question requires professional legal advice.**  "
                "The app cannot provide a legal verdict or tell you whether to sign."
            )
            if lq:
                st.markdown("**Suggested question for a qualified legal professional or local legal-aid service:**")
                st.info(f'"{lq}"')
            st.markdown(
                "_Find free legal aid: in India, your District Legal Services Authority "
                "(NALSA helpline 15100); elsewhere, search for legal aid in your country._"
            )
        return

    # --- Abstention ---
    if result.get("abstained"):
        with st.container(border=True):
            st.info(result.get("message", ""))
            st.caption(
                "The document does not contain enough retrievable evidence to answer this question. "
                "Review the full document and consider professional legal advice."
            )
        reason = result.get("reason")
        top_similarity = result.get("top_similarity")
        if reason == "below_retrieval_threshold":
            st.caption(
                f"Search stopped before answer generation: best passage similarity was "
                f"{top_similarity} and the configured minimum is "
                f"{SETTINGS.min_retrieval_similarity}. Similarity is a debugging aid, not a confidence score."
            )
        elif reason == "no_validated_quote_evidence":
            st.caption(
                "Passages were retrieved, but the model did not return an answer with a quote that passed "
                "the source and verbatim-text checks."
            )
        sources = result.get("sources", [])
        if sources:
            with st.expander(f"Retrieved passages ({len(sources)})"):
                for source in sources:
                    all_pages = source.get("all_pages", [source.get("page_number", "?")])
                    page_str = "+".join(str(page) for page in all_pages)
                    st.caption(
                        f"{source.get('filename', '')} · page {page_str} · "
                        f"retrieval similarity {source.get('similarity', 'n/a')}"
                    )
                    st.code(source.get("text", ""), language=None)
        return

    if result.get("message"):
        st.caption(result["message"])

    findings = result.get("findings", [])
    if not findings:
        return

    _render_key_facts_card(findings)

    by_topic: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        by_topic[f.get("topic", "general")].append(f)

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

                # Coloured left-border card (attention colour coding)
                border_color = attn_cfg["color"]
                st.markdown(
                    f"<div style='border-left:4px solid {border_color};padding-left:10px;margin-bottom:4px;'>",
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    col_l, col_r = st.columns([3, 1])
                    with col_l:
                        st.markdown(f"**{number}. {cat_label}**")
                    with col_r:
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
                        _render_quote_in_context(
                            citation.get("quote", ""),
                            source.get("text", ""),
                        )

                    lawyer_question = finding.get("lawyer_question", "")
                    if lawyer_question:
                        st.markdown("**💬 Question to discuss with a qualified legal professional**")
                        st.info(lawyer_question)

                    _render_support_badges(finding.get("support", []))
                st.markdown("</div>", unsafe_allow_html=True)

    jurisdiction_notes = result.get("jurisdiction_notes", [])
    if jurisdiction_notes:
        with st.expander("🌍 Jurisdiction notes"):
            for note in jurisdiction_notes:
                st.info(note)

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
# Quote-in-context highlighting
# ---------------------------------------------------------------------------


def _render_quote_in_context(quote: str, source_text: str) -> None:
    import html as _html
    from legal_triage.grounding import normalize_text_for_matching

    q_norm = normalize_text_for_matching(quote)
    s_norm = normalize_text_for_matching(source_text)

    idx = s_norm.find(q_norm) if q_norm else -1
    if idx == -1 or not source_text:
        st.code(quote, language=None)
        return

    ratio = idx / max(len(s_norm), 1)
    orig_start = max(0, int(ratio * len(source_text)) - 10)
    q_len_est = len(quote)
    orig_end = min(len(source_text), orig_start + q_len_est + 20)
    ctx_start = max(0, orig_start - 120)
    ctx_end   = min(len(source_text), orig_end + 120)
    prefix = source_text[ctx_start:orig_start]
    matched = source_text[orig_start:orig_end]
    suffix = source_text[orig_end:ctx_end]

    esc_prefix  = _html.escape(prefix)
    esc_matched = _html.escape(matched)
    esc_suffix  = _html.escape(suffix)

    html_block = (
        "<div style='background:#1e1e2e;border-radius:6px;padding:10px 14px;"
        "font-family:monospace;font-size:0.85em;line-height:1.6;color:#cdd6f4;"
        "white-space:pre-wrap;word-break:break-word;'>"
        f"{'<span style=&quot;color:#585b70&quot;>&hellip;</span>' if ctx_start > 0 else ''}"
        f"{esc_prefix}"
        f"<mark style='background:#f9e2af;color:#1e1e2e;border-radius:3px;"
        f"padding:0 2px;'>{esc_matched}</mark>"
        f"{esc_suffix}"
        f"{'<span style=&quot;color:#585b70&quot;>&hellip;</span>' if ctx_end < len(source_text) else ''}"
        "</div>"
    )
    st.markdown(html_block, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Honest support badges
# ---------------------------------------------------------------------------


def _render_support_badges(support: list[str]) -> None:
    _BADGE_CONFIG = {
        "quote_valid":      ("✔ Quote verified",  "#1e8449", "#d5f5e3"),
        "numbers_verified": ("🔢 Numbers match",   "#1a5276", "#d6eaf8"),
        "verifier_agrees":  ("🤖 Model-checked",   "#7d3c98", "#f5eef8"),
        "unverified":       ("⚠ Unverified",       "#7e5109", "#fdebd0"),
    }
    badge_html = " ".join(
        f"<span style='background:{bg};color:{fg};border-radius:4px;"
        f"padding:2px 8px;font-size:0.78em;font-weight:600'>{label}</span>"
        for key, (label, fg, bg) in _BADGE_CONFIG.items()
        if key in support
    )
    if badge_html:
        st.markdown(badge_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Checklist and structured-extraction rendering
# ---------------------------------------------------------------------------


def _render_checklist(chunks: list) -> None:
    if not chunks:
        return
    full_text = " ".join(c.text for c in chunks)
    doc_type = detect_document_type(full_text)
    verdicts = missing_clause_checklist(chunks, doc_type)
    if not verdicts:
        return

    _VERDICT_ICON = {PRESENT: "✅", POSSIBLY_PRESENT: "⚠️", ABSENT: "❌"}
    absent_required = [v for v in verdicts if v.verdict == ABSENT and v.required]

    label = f"📋 Clause checklist ({doc_type} document)"
    expanded = bool(absent_required)
    with st.expander(label, expanded=expanded):
        st.caption(
            f"Document type detected: **{doc_type}**. "
            "Checklist is keyword-based — review the full document to confirm."
        )
        if absent_required:
            st.warning(
                f"{len(absent_required)} potentially missing required clause(s) detected. "
                "Review the full document before relying on it."
            )
        for v in verdicts:
            icon = _VERDICT_ICON.get(v.verdict, "❓")
            req_label = "" if v.required else " *(optional)*"
            st.markdown(f"{icon} **{v.clause_name}**{req_label}")
            if v.matched_quote and v.verdict != ABSENT:
                st.caption(v.matched_quote[:160])


def _render_structured_fields(chunks: list) -> None:
    if not chunks:
        return
    fields = extract_structured_fields(chunks)
    if not fields:
        return
    from collections import defaultdict as _dd
    grouped: dict[str, list] = _dd(list)
    for f in fields:
        grouped[f.field_name].append(f)

    with st.expander("📊 Extracted structured fields", expanded=False):
        st.caption(
            "These values are extracted by regex from the document text. "
            "Always verify against the original document."
        )
        for field_name, field_list in grouped.items():
            st.markdown(f"**{field_name}**")
            for sf in field_list[:5]:
                col_val, col_src = st.columns([2, 3])
                with col_val:
                    st.code(sf.value, language=None)
                with col_src:
                    st.caption(f"{sf.filename} · page {sf.page_number}")
                    st.caption(sf.quote[:120])


# ---------------------------------------------------------------------------
# Persist checklist / structured fields for message replay
# ---------------------------------------------------------------------------


def _checklist_to_serialisable(chunks: list) -> list[dict]:
    """Return checklist verdicts as plain dicts for JSON storage."""
    if not chunks:
        return []
    try:
        full_text = " ".join(c.text for c in chunks)
        doc_type = detect_document_type(full_text)
        verdicts = missing_clause_checklist(chunks, doc_type)
        return [
            {
                "clause_name": v.clause_name,
                "verdict": v.verdict,
                "required": v.required,
                "matched_quote": v.matched_quote or "",
                "doc_type": doc_type,
            }
            for v in verdicts
        ]
    except Exception:
        return []


def _structured_fields_to_serialisable(chunks: list) -> list[dict]:
    """Return structured fields as plain dicts for JSON storage."""
    if not chunks:
        return []
    try:
        fields = extract_structured_fields(chunks)
        return [
            {
                "field_name": f.field_name,
                "value": f.value,
                "filename": f.filename,
                "page_number": f.page_number,
                "quote": f.quote,
            }
            for f in fields
        ]
    except Exception:
        return []


def _render_checklist_from_stored(checklist_data: list[dict]) -> None:
    """Re-render a checklist from stored serialisable data (for history replay)."""
    if not checklist_data:
        return
    _VERDICT_ICON = {PRESENT: "✅", POSSIBLY_PRESENT: "⚠️", ABSENT: "❌"}
    doc_type = checklist_data[0].get("doc_type", "unknown") if checklist_data else "unknown"
    absent_required = [v for v in checklist_data if v.get("verdict") == ABSENT and v.get("required")]
    with st.expander(f"📋 Clause checklist ({doc_type} document)", expanded=bool(absent_required)):
        st.caption(f"Document type detected: **{doc_type}**. Keyword-based — verify against original.")
        if absent_required:
            st.warning(f"{len(absent_required)} potentially missing required clause(s) detected.")
        for v in checklist_data:
            icon = _VERDICT_ICON.get(v.get("verdict", ABSENT), "❓")
            req_label = "" if v.get("required") else " *(optional)*"
            st.markdown(f"{icon} **{v.get('clause_name', '')}**{req_label}")
            mq = v.get("matched_quote", "")
            if mq and v.get("verdict") != ABSENT:
                st.caption(mq[:160])


def _render_structured_fields_from_stored(fields_data: list[dict]) -> None:
    """Re-render structured fields from stored serialisable data (for history replay)."""
    if not fields_data:
        return
    from collections import defaultdict as _dd
    grouped: dict[str, list] = _dd(list)
    for f in fields_data:
        grouped[f.get("field_name", "")].append(f)
    with st.expander("📊 Extracted structured fields", expanded=False):
        st.caption("Extracted by regex. Always verify against the original document.")
        for field_name, field_list in grouped.items():
            st.markdown(f"**{field_name}**")
            for sf in field_list[:5]:
                col_val, col_src = st.columns([2, 3])
                with col_val:
                    st.code(sf.get("value", ""), language=None)
                with col_src:
                    st.caption(f"{sf.get('filename', '')} · page {sf.get('page_number', '')}")
                    st.caption(sf.get("quote", "")[:120])


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
        st.error(str(exc))
        return

    # Persist checklist + structured fields inside the message so they survive page-refresh replay
    checklist_data: list[dict] = []
    fields_data: list[dict] = []
    if mode == "triage" and not result.get("abstained"):
        checklist_data = _checklist_to_serialisable(index.chunks)
        fields_data = _structured_fields_to_serialisable(index.chunks)

    msg: dict = {
        "role": "assistant",
        "question": question,
        "mode": mode,
        "result": result,
        "checklist": checklist_data,
        "structured_fields": fields_data,
    }
    st.session_state.messages.append(msg)

    render_result(
        result,
        "Whole-document triage" if mode == "triage" else "Answer",
        question=question,
        download_key=f"result_download_{len(st.session_state.messages) - 1}",
    )
    if checklist_data:
        _render_checklist_from_stored(checklist_data)
    if fields_data:
        _render_structured_fields_from_stored(fields_data)

    # Persist to disk after every assistant response
    _autosave_session()


def _autosave_session() -> None:
    """Write the current session to ~/.legal_triage_history/ if we have documents loaded."""
    if st.session_state.get("_replaying"):
        return
    meta = st.session_state.get("index_metadata")
    if not meta:
        return
    filenames: list[str] = meta.get("files", [])
    sid = st.session_state.get("current_session_id")
    created_at = st.session_state.get("current_session_created")
    if not sid:
        return
    try:
        save_session(sid, filenames, st.session_state.messages, created_at=created_at)
    except OSError:
        pass  # never surface a history write error to the user


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

initialise_state()

st.title("⚖️ Private Legal Document Triage Agent")
st.caption("Local PDF/text ingestion · Local embeddings · Local Gemma · Verifiable passages · On-device history")
st.warning(
    "For preliminary document understanding only — not legal advice. "
    "This app cannot decide legal validity or enforceability. "
    "Important, unclear, or jurisdiction-dependent issues should be reviewed by a qualified legal professional."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Privacy badge (L9 / PS2 criterion 4)
    st.markdown(
        "<div style='background:#1e8449;color:white;border-radius:6px;padding:8px 12px;"
        "font-weight:700;font-size:0.9em;text-align:center;margin-bottom:12px;'>"
        "🔒 On-device only — no data leaves this machine"
        "</div>",
        unsafe_allow_html=True,
    )

    # --- Model status chip ---
    st.subheader("Runtime")
    if st.session_state.model_check:
        ok, details = st.session_state.model_check
        if ok:
            st.markdown(
                f"<span style='background:#1e8449;color:white;border-radius:4px;"
                f"padding:2px 10px;font-size:0.85em;font-weight:600'>✔ Models ready</span>  "
                f"`{details.get('chat_model','')}`",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='background:#c0392b;color:white;border-radius:4px;"
                "padding:2px 10px;font-size:0.85em;font-weight:600'>✖ Model error</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Run 'Check local models' to confirm Gemma is loaded locally.")

    if st.button("Check local models", use_container_width=True, key="check_models_btn"):
        try:
            details = get_runtime().verify_models()
            st.session_state.model_check = (True, details)
        except (LocalModelError, ValueError) as exc:
            st.session_state.model_check = (False, str(exc))
        st.rerun()

    # --- Settings expander ---
    with st.expander("⚙️ Settings", expanded=False):
        st.code(
            f"Gemma model: {SETTINGS.chat_model}\n"
            f"Embedding model: {SETTINGS.embedding_model}\n"
            f"Ollama endpoint: {SETTINGS.ollama_base_url}\n"
            f"Min retrieval similarity: {SETTINGS.min_retrieval_similarity}",
            language="text",
        )

    st.divider()

    # --- Privacy details ---
    with st.expander("🔐 Privacy details", expanded=False):
        st.markdown(
            "**What stays on device:**  \n"
            "Document text · Embeddings · LLM prompts & answers · Chat history\n\n"
            "**What leaves the device:** Nothing during processing.\n\n"
            "The app accepts only a loopback Ollama URL and disables inherited HTTP proxy settings. "
            "Uploaded files, extracted text, vectors, and chat state stay in this app process's session memory. "
            "Chat history is saved to `~/.legal_triage_history/` in JSON format. "
            "Do not expose Streamlit or Ollama to a network."
        )
        st.caption(
            "For a full privacy claim, also disable Ollama cloud features and produce network-monitoring "
            "evidence as described in README.md."
        )

    st.divider()

    # --- Session history panel ---
    st.subheader("📂 Session history")
    st.caption("All previous sessions are stored locally on your device only.")

    history = list_sessions()
    if not history:
        st.info("No previous sessions found.")
    else:
        for sess in history[:20]:
            files_label = ", ".join(sess["files"][:2])
            if len(sess["files"]) > 2:
                files_label += f" +{len(sess['files']) - 2} more"
            created = sess.get("created_at", "")[:16].replace("T", " ")
            msg_count = sess.get("message_count", 0)
            btn_label = f"🗂 {created}  ·  {files_label}  ·  {msg_count} msg(s)"
            col_load, col_del = st.columns([5, 1])
            with col_load:
                if st.button(btn_label, key=f"hist_{sess['session_id']}", use_container_width=True):
                    loaded = load_session(sess["session_id"])
                    if loaded:
                        st.session_state._replaying = True
                        st.session_state.messages = loaded.get("messages", [])
                        st.session_state.current_session_id = loaded["session_id"]
                        st.session_state.current_session_created = loaded.get("created_at")
                        # Clear live document state — replay shows stored messages only
                        st.session_state.local_index = None
                        st.session_state.index_metadata = {
                            "files": loaded.get("files", []),
                            "pages": 0,
                            "chunks": 0,
                            "_replayed": True,
                        }
                        st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{sess['session_id']}", help="Delete this session"):
                    delete_session(sess["session_id"])
                    if st.session_state.get("current_session_id") == sess["session_id"]:
                        st.session_state.current_session_id = None
                    st.rerun()

    st.divider()
    if st.button("Clear documents, vectors, and chat", type="secondary", use_container_width=True, key="clear_btn"):
        old_upload_key = f"uploaded_files_{st.session_state.upload_epoch}"
        st.session_state.local_index = None
        st.session_state.index_metadata = None
        st.session_state.messages = []
        st.session_state.upload_epoch += 1
        st.session_state.model_check = None
        st.session_state.current_session_id = None
        st.session_state.current_session_created = None
        st.session_state._replaying = False
        if old_upload_key in st.session_state:
            del st.session_state[old_upload_key]
        st.rerun()

# ---------------------------------------------------------------------------
# Step 1: Upload
# ---------------------------------------------------------------------------

st.subheader("1 · Add a document")

if st.session_state.get("_replaying") and st.session_state.get("index_metadata", {}).get("_replayed"):
    files_label = ", ".join(st.session_state.index_metadata.get("files", []))
    st.info(
        f"📂 **Replaying saved session** — documents: *{files_label}*  \n"
        "Re-upload the same files and click 'Process documents locally' to run new analysis."
    )

upload_key = f"uploaded_files_{st.session_state.upload_epoch}"
uploaded_files = st.file_uploader(
    "Select text-based PDF, UTF-8 .txt, or .docx files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    key=upload_key,
    help=(
        f"Up to {SETTINGS.max_files} files; {SETTINGS.max_file_mb} MB per file. "
        "Scanned PDFs need OCR, which this build does not include. "
        "DOCX files are accepted — paragraph and table text are extracted."
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
        # Assign a new session ID for this upload
        filenames = [f.name for f in uploaded_files]
        sid = make_session_id(filenames)
        st.session_state.current_session_id = sid
        st.session_state.current_session_created = datetime.now().isoformat(timespec="seconds")
        st.session_state._replaying = False
        st.success("Index created in memory. No uploaded file or vector index was saved by this app to disk.")
    except (IngestionError, LocalModelError, ValueError, RuntimeError) as exc:
        st.error(str(exc))

if st.session_state.index_metadata:
    meta = st.session_state.index_metadata
    if not meta.get("_replayed"):
        st.success(
            f"Indexed {len(meta['files'])} file(s), {meta['pages']} total page(s) "
            f"({meta.get('text_pages', meta['pages'])} with extractable text), and {meta['chunks']} chunks."
        )
        st.caption("Files: " + ", ".join(meta["files"]))
    for warning in meta.get("warnings", []):
        if "injection" in warning.lower() or "instruction" in warning.lower():
            st.warning(f"🚨 **Injection attempt detected in uploaded document:** {warning}")
        else:
            st.warning(warning)

# ---------------------------------------------------------------------------
# Step 2: Triage / question
# ---------------------------------------------------------------------------

st.divider()
st.subheader("2 · Triage or ask a question")
if st.session_state.local_index is None and not st.session_state.get("_replaying"):
    st.info("Process at least one document to enable triage and question answering.")
else:
    # Replay transcript
    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message.get("content", ""))
            else:
                mode_title = (
                    "Whole-document triage"
                    if message.get("mode") == "triage"
                    else f"Answer to: {message.get('question', '')}"
                )
                render_result(
                    message.get("result", {}),
                    mode_title,
                    question=message.get("question", ""),
                    download_key=f"result_download_{message_index}",
                )
                # Replay stored checklist + structured fields
                stored_checklist = message.get("checklist", [])
                stored_fields = message.get("structured_fields", [])
                if stored_checklist:
                    _render_checklist_from_stored(stored_checklist)
                if stored_fields:
                    _render_structured_fields_from_stored(stored_fields)

    if not st.session_state.get("_replaying"):
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
    else:
        st.info(
            "📂 This is a saved session replay. "
            "Re-upload the original files and click 'Process documents locally' to ask new questions."
        )

# ---------------------------------------------------------------------------
# Step 3: Two-version comparison (Phase 4)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("3 · Compare two contract versions  *(optional)*")
st.caption(
    "Upload two versions of the same document type (e.g. draft vs final, "
    "your contract vs landlord's template) and ask a specific question. "
    "The agent answers each version independently so you can spot differences "
    "in notice periods, amounts, obligations, etc."
)

cmp_col_a, cmp_col_b = st.columns(2)

with cmp_col_a:
    st.markdown("**Contract A**")
    cmp_files_a = st.file_uploader(
        "Upload Contract A (PDF / TXT / DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="cmp_files_a",
    )
    if st.button("Index Contract A", key="cmp_idx_a", disabled=not cmp_files_a):
        try:
            rt = get_runtime()
            with st.spinner("Indexing Contract A…"):
                rt.verify_models()
                st.session_state["cmp_index_a"] = build_index(
                    [(f.name, f.getvalue()) for f in cmp_files_a], rt
                )
            st.success(f"Contract A indexed ({len(cmp_files_a)} file(s)).")
        except (IngestionError, LocalModelError, ValueError, RuntimeError) as exc:
            st.error(str(exc))

with cmp_col_b:
    st.markdown("**Contract B**")
    cmp_files_b = st.file_uploader(
        "Upload Contract B (PDF / TXT / DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="cmp_files_b",
    )
    if st.button("Index Contract B", key="cmp_idx_b", disabled=not cmp_files_b):
        try:
            rt = get_runtime()
            with st.spinner("Indexing Contract B…"):
                rt.verify_models()
                st.session_state["cmp_index_b"] = build_index(
                    [(f.name, f.getvalue()) for f in cmp_files_b], rt
                )
            st.success(f"Contract B indexed ({len(cmp_files_b)} file(s)).")
        except (IngestionError, LocalModelError, ValueError, RuntimeError) as exc:
            st.error(str(exc))

idx_a = st.session_state.get("cmp_index_a")
idx_b = st.session_state.get("cmp_index_b")

if idx_a and idx_b:
    cmp_question = st.text_input(
        "Question to ask about both contracts",
        placeholder="e.g. What is the notice period to terminate?",
        key="cmp_question",
    )
    if st.button("Compare", type="primary", key="cmp_run_btn", disabled=not cmp_question):
        rt = get_runtime()
        col_res_a, col_res_b = st.columns(2)
        with col_res_a:
            st.markdown("#### Contract A")
            try:
                with st.spinner("Analyzing Contract A…"):
                    res_a = analyze(cmp_question, idx_a, rt, mode="question")
                render_result(
                    res_a,
                    "Contract A",
                    question=cmp_question,
                    download_key="comparison_a_download",
                )
            except (LocalModelError, AnalysisError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
        with col_res_b:
            st.markdown("#### Contract B")
            try:
                with st.spinner("Analyzing Contract B…"):
                    res_b = analyze(cmp_question, idx_b, rt, mode="question")
                render_result(
                    res_b,
                    "Contract B",
                    question=cmp_question,
                    download_key="comparison_b_download",
                )
            except (LocalModelError, AnalysisError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
elif idx_a or idx_b:
    st.info("Index both Contract A and Contract B to enable comparison.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "🔒 All processing is local. No data leaves this device. "
    "Chat history is saved to `~/.legal_triage_history/` on your machine only. "
    "This tool provides preliminary document understanding, not legal advice."
)
