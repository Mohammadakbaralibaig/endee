import streamlit as st
import os
from dotenv import load_dotenv
from utils.document_processor import process_document, chunk_text
from utils.embedder import get_embedding, get_embeddings_batch
from utils.vector_store import init_endee, upsert_chunks, search_similar
from utils.llm import generate_answer

load_dotenv()

st.set_page_config(
    page_title="SafetyLens · Industrial AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #eef0f6 !important;
}
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 99px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1e1b4b !important;
    border-right: none !important;
    box-shadow: 4px 0 32px rgba(79,70,229,0.2) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: #e0e7ff !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div { background: #818cf8 !important; }
[data-testid="stSidebar"] input {
    background: #312e81 !important;
    border: 1px solid #4338ca !important;
    color: #e0e7ff !important;
    border-radius: 10px !important;
}

/* ── Sidebar toggle button ── */
[data-testid="stSidebarCollapsedControl"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border-radius: 0 10px 10px 0 !important;
    color: white !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.5) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: #fff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 14px !important;
    color: #0f172a !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1.2rem !important;
    transition: all 0.25s !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.12) !important;
}
.stTextInput > div > div > input::placeholder { color: #94a3b8 !important; }

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
    background: #f8faff !important;
    border: 2px dashed #c7d2fe !important;
    border-radius: 16px !important;
    transition: all 0.25s !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #6366f1 !important;
    background: #eef2ff !important;
}
section[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #6366f1 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ── Brand ── */
.sl-brand {
    background: linear-gradient(160deg, #312e81 0%, #1e1b4b 100%);
    padding: 1.75rem 1.5rem 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.sl-brand-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.sl-brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 4px 12px rgba(99,102,241,0.4);
}
.sl-brand-name {
    font-family: 'Poppins', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: #fff !important;
    letter-spacing: -0.3px;
}
.sl-brand-name span { color: #a5f3fc !important; }
.sl-brand-sub {
    font-size: 0.62rem;
    color: rgba(255,255,255,0.45) !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.sl-online {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 99px;
    padding: 3px 10px;
    font-size: 0.65rem;
    color: #6ee7b7 !important;
    font-weight: 600;
}
.sl-online-dot { width: 6px; height: 6px; border-radius: 50%; background: #34d399; }

/* ── Sidebar nav ── */
.sl-nav {
    padding: 1.25rem 1.5rem;
}
.sl-nav-title {
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: rgba(255,255,255,0.35) !important;
    margin-bottom: 0.6rem;
    margin-top: 0.75rem;
}
.sl-mini-stat {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.sl-mini-num {
    font-family: 'Poppins', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #a5b4fc !important;
    line-height: 1;
}
.sl-mini-lbl {
    font-size: 0.62rem;
    color: rgba(255,255,255,0.4) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    text-align: right;
}

/* ── Page title ── */
.sl-page-title {
    text-align: center;
    padding: 0.5rem 0 1.5rem;
}
.sl-page-title h1 {
    font-family: 'Poppins', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #0891b2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin: 0;
}
.sl-page-title p {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* ── Hero banner ── */
.sl-hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #0891b2 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.sl-hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -60px;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
    pointer-events: none;
}
.sl-hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 40%;
    width: 200px; height: 200px;
    background: rgba(165,243,252,0.07);
    border-radius: 50%;
    pointer-events: none;
}
.sl-hero-tag {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #a5f3fc;
    margin-bottom: 0.5rem;
}
.sl-hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.sl-hero-title span {
    background: linear-gradient(90deg, #a5f3fc, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sl-hero-desc {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.72);
    line-height: 1.7;
    max-width: 480px;
    margin-bottom: 1.25rem;
}
.sl-pipeline {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}
.sl-pipe {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.68rem;
    color: rgba(255,255,255,0.65);
    font-weight: 500;
    backdrop-filter: blur(4px);
}
.sl-pipe.on {
    background: rgba(165,243,252,0.2);
    border-color: rgba(165,243,252,0.5);
    color: #a5f3fc;
    font-weight: 700;
}
.sl-arr { color: rgba(255,255,255,0.35); font-size: 0.8rem; }

/* ── Stats ── */
.sl-stats {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 1.5rem;
}
.sl-stat {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1rem 0.75rem;
    text-align: center;
    transition: all 0.25s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    cursor: default;
}
.sl-stat:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 28px rgba(99,102,241,0.15);
    border-color: #c7d2fe;
}
.sl-stat-val {
    font-family: 'Poppins', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.sl-stat-key {
    font-size: 0.58rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
    font-weight: 600;
}

/* ── Cards ── */
.sl-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    transition: box-shadow 0.25s;
}
.sl-card:hover { box-shadow: 0 8px 32px rgba(99,102,241,0.1); }
.sl-card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #f1f5f9;
}
.sl-card-icon {
    width: 40px; height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, #eef2ff, #e0e7ff);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 2px 8px rgba(99,102,241,0.15);
}
.sl-card-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #1e293b;
    letter-spacing: -0.2px;
}
.sl-card-sub {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 1px;
}

/* ── File info ── */
.sl-file-info {
    background: linear-gradient(135deg, #f8faff, #eef2ff);
    border: 1px solid #e0e7ff;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #4f46e5;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1rem;
}

/* ── Q&A ── */
.sl-qa {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(99,102,241,0.08);
    margin-bottom: 1.5rem;
}
.sl-qa-head {
    background: linear-gradient(135deg, #f8faff, #eef2ff);
    border-bottom: 1px solid #e0e7ff;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.sl-qa-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #4f46e5;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.sl-qa-badge {
    font-size: 0.65rem;
    color: #6366f1;
    font-weight: 600;
    background: #fff;
    border: 1px solid #e0e7ff;
    padding: 3px 10px;
    border-radius: 99px;
}
.sl-chat {
    height: 360px;
    overflow-y: auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    background: #fafbff;
}
.sl-chat::-webkit-scrollbar { width: 4px; }
.sl-chat::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 99px; }
.sl-empty {
    margin: auto;
    text-align: center;
    padding: 2rem 0;
}
.sl-empty-icon { font-size: 2.5rem; opacity: 0.35; margin-bottom: 0.75rem; }
.sl-empty-text { font-size: 0.85rem; color: #94a3b8; line-height: 1.6; }
.sl-q {
    align-self: flex-end;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    font-size: 0.85rem;
    font-weight: 500;
    max-width: 75%;
    box-shadow: 0 4px 12px rgba(99,102,241,0.25);
    line-height: 1.5;
}
.sl-a {
    align-self: flex-start;
    background: #fff;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #6366f1;
    border-radius: 4px 18px 18px 18px;
    padding: 1rem 1.1rem;
    font-size: 0.85rem;
    color: #1e293b;
    line-height: 1.7;
    max-width: 85%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.sl-a-meta {
    font-size: 0.6rem;
    color: #a5b4fc;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.sl-input-area {
    border-top: 1px solid #e2e8f0;
    padding: 1rem 1.5rem;
    background: #fff;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────
for k, v in [("index_ready", False), ("chat_history", []),
              ("endee_index", None), ("endee_client", None),
              ("chunk_count", 0), ("doc_name", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sl-brand">
        <div class="sl-brand-logo">
            <div class="sl-brand-icon">🔬</div>
            <div>
                <div class="sl-brand-name">Safety<span>Lens</span></div>
            </div>
        </div>
        <div class="sl-brand-sub">Industrial AI · Endee VectorDB</div>
        <div class="sl-online">
            <div class="sl-online-dot"></div> Endee connected
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sl-nav">
        <div class="sl-nav-title">⚙ Server Config</div>
    </div>
    """, unsafe_allow_html=True)
    endee_url = st.text_input("eu", value="http://localhost:8080",
                               label_visibility="collapsed")

    st.markdown("""
    <div class="sl-nav">
        <div class="sl-nav-title">🎛 RAG Parameters</div>
    </div>
    """, unsafe_allow_html=True)
    chunk_size    = st.slider("Chunk Size", 200, 800, 400, 50)
    chunk_overlap = st.slider("Overlap",      0, 200,  50, 25)
    top_k         = st.slider("Top-K",        1,  10,   5,  1)

    if st.session_state.index_ready:
        st.markdown(f"""
        <div class="sl-nav">
            <div class="sl-nav-title">📊 Index Stats</div>
            <div class="sl-mini-stat">
                <div class="sl-mini-num">{st.session_state.chunk_count}</div>
                <div class="sl-mini-lbl">Chunks<br/>in Endee</div>
            </div>
            <div class="sl-mini-stat">
                <div class="sl-mini-num">{len(st.session_state.chat_history)}</div>
                <div class="sl-mini-lbl">Queries<br/>answered</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("↺  Reset & Upload New", use_container_width=True):
            try:
                st.session_state.endee_client.delete_index("safetylens_index")
            except: pass
            for k, v in [("index_ready", False), ("chat_history", []),
                          ("endee_index", None), ("endee_client", None),
                          ("chunk_count", 0), ("doc_name", "")]:
                st.session_state[k] = v
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class="sl-nav">
        <div style="font-size:0.6rem; color:rgba(255,255,255,0.3);
             line-height:2; text-transform:uppercase; letter-spacing:0.08em;">
        RAG · HNSW · Vector Search<br/>
        768-dim · Gemini Embeddings<br/>
        Built for Endee · 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Page Title ────────────────────────────────────────────────────
st.markdown("""
<div class="sl-page-title">
    <h1>SafetyLens</h1>
    <p>Industrial AI powered by Endee Vector Database · RAG · Semantic Search</p>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────
p1 = "on" if not st.session_state.index_ready else ""
p4 = "on" if st.session_state.index_ready else ""

st.markdown(f"""
<div class="sl-hero">
    <div class="sl-hero-tag">Industrial Safety · RAG + Endee Vector Search</div>
    <div class="sl-hero-title">Ask your safety documents.<br/>Get <span>instant answers.</span></div>
    <div class="sl-hero-desc">
        Upload equipment manuals, gas sensor datasheets, or plant safety protocols.
        Ask questions in plain language — get precise, source-grounded AI answers.
    </div>
    <div class="sl-pipeline">
        <div class="sl-pipe {p1}">01 · Upload Doc</div>
        <div class="sl-arr">→</div>
        <div class="sl-pipe">02 · Chunk & Embed</div>
        <div class="sl-arr">→</div>
        <div class="sl-pipe on">03 · Endee Index</div>
        <div class="sl-arr">→</div>
        <div class="sl-pipe {p4}">04 · Ask & Answer</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────────
c   = st.session_state.chunk_count if st.session_state.index_ready else "—"
q   = len(st.session_state.chat_history)
doc = (st.session_state.doc_name[:13] + "…") if st.session_state.doc_name \
      and len(st.session_state.doc_name) > 13 else (st.session_state.doc_name or "—")

st.markdown(f"""
<div class="sl-stats">
    <div class="sl-stat"><div class="sl-stat-val">{c}</div><div class="sl-stat-key">Chunks indexed</div></div>
    <div class="sl-stat"><div class="sl-stat-val">768</div><div class="sl-stat-key">Vector dims</div></div>
    <div class="sl-stat"><div class="sl-stat-val">HNSW</div><div class="sl-stat-key">Index type</div></div>
    <div class="sl-stat"><div class="sl-stat-val">cosine</div><div class="sl-stat-key">Similarity</div></div>
    <div class="sl-stat"><div class="sl-stat-val">{q}</div><div class="sl-stat-key">Queries run</div></div>
    <div class="sl-stat"><div class="sl-stat-val" style="font-size:0.72rem;">{doc}</div><div class="sl-stat-key">Active doc</div></div>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────
if not st.session_state.index_ready:
    st.markdown("""
    <div class="sl-card">
        <div class="sl-card-header">
            <div class="sl-card-icon">📄</div>
            <div>
                <div class="sl-card-title">Upload your Document</div>
                <div class="sl-card-sub">PDF or TXT · Safety manuals, datasheets, protocols</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload", type=["pdf", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown(f"""
        <div class="sl-file-info">
            📎 &nbsp;{uploaded_file.name}
            &nbsp;·&nbsp; {round(uploaded_file.size/1024, 1)} KB
            &nbsp;·&nbsp; Ready to index
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 2, 2])
        with col2:
            go = st.button("🚀  Index into Endee", use_container_width=True)

        if go:
            with st.spinner("📄 Extracting text..."):
                raw_text = process_document(uploaded_file)
            with st.spinner("✂️ Chunking..."):
                chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
            with st.spinner(f"🧠 Embedding {len(chunks)} chunks..."):
                embeddings = get_embeddings_batch(chunks)
            with st.spinner("🗄️ Storing in Endee..."):
                client, index = init_endee(
                    base_url=endee_url,
                    dimension=len(embeddings[0]),
                    index_name="safetylens_index"
                )
                upsert_chunks(index, chunks, embeddings, uploaded_file.name)
            st.session_state.endee_client = client
            st.session_state.endee_index  = index
            st.session_state.index_ready  = True
            st.session_state.chunk_count  = len(chunks)
            st.session_state.doc_name     = uploaded_file.name
            st.rerun()

# ── Q&A Panel ────────────────────────────────────────────────────
else:
    st.markdown(f"""
    <div class="sl-qa">
        <div class="sl-qa-head">
            <div class="sl-qa-title">💬 &nbsp; Q&A — {st.session_state.doc_name}</div>
            <div class="sl-qa-badge">{st.session_state.chunk_count} chunks · HNSW · top-{top_k}</div>
        </div>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div class="sl-chat">
            <div class="sl-empty">
                <div class="sl-empty-icon">🔬</div>
                <div class="sl-empty-text">
                    Document indexed successfully ✓<br/>
                    Type your question below to get started
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chat_html = '<div class="sl-chat">'
        for entry in st.session_state.chat_history:
            chat_html += f'<div class="sl-q">{entry["question"]}</div>'
            chat_html += f"""<div class="sl-a">
                <div class="sl-a-meta">⚡ {len(entry['results'])} chunks retrieved from Endee</div>
                {entry['answer']}
            </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sl-input-area">', unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "q", label_visibility="collapsed",
            placeholder="Ask about safety procedures, risks, exposure limits, or compliance...",
            key="q_input"
        )
    with col2:
        ask = st.button("🔍  Search")
    st.markdown('</div>', unsafe_allow_html=True)

    if ask and question:
        with st.spinner("Searching Endee..."):
            query_vector = get_embedding(question)
            results = search_similar(
                st.session_state.endee_index,
                query_vector, top_k=top_k
            )
            context_chunks = [r["meta"]["text"] for r in results]
        with st.spinner("Generating answer..."):
            answer = generate_answer(question, context_chunks)
        st.session_state.chat_history.append({
            "question": question,
            "answer":   answer,
            "results":  results
        })
        st.rerun()