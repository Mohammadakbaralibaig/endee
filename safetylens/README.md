# ⚡ VectoRAG — Industrial Safety Document Q&A

> RAG-powered AI system for querying industrial safety documents using **Endee Vector Database**

![Python](https://img.shields.io/badge/Python-3.13-blue) ![Endee](https://img.shields.io/badge/Vector_DB-Endee-purple) ![Groq](https://img.shields.io/badge/LLM-Groq_Llama3-green) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 📌 Project Overview & Problem Statement

Industrial plant workers and safety officers deal with hundreds of pages of safety manuals, gas sensor datasheets, and equipment protocols. Finding specific information like exposure limits, alarm thresholds, or maintenance procedures is slow and error-prone.

**VectoRAG (SafetyLens)** solves this by building a complete RAG pipeline that lets users upload any safety document and ask questions in plain language — getting instant, accurate, source-grounded answers.

---

## 🏗️ System Design & Technical Approach
```
INDEXING PIPELINE
─────────────────
PDF/TXT → Text Extraction → Chunking → Embedding → Endee Vector DB
                            (sliding    (hash-based   (HNSW Index,
                             window)     vectors)      cosine sim)

QUERY PIPELINE
──────────────
Question → Embed Query → Endee ANN Search → Top-K Chunks → Groq LLM → Answer
```

### Architecture Decisions

| Component | Technology | Reason |
|---|---|---|
| **Vector Database** | **Endee** | High-performance HNSW-based ANN search, open-source |
| **LLM** | Groq (Llama 3.1) | Fast inference, free tier, works globally |
| **UI** | Streamlit | Rapid prototyping, clean interactive interface |
| **Document Parsing** | pypdf | Reliable PDF text extraction |

---

## 🗄️ How Endee is Used

Endee is the **core retrieval engine** of this entire RAG system.

### 1. Creating the Index
```python
from endee import Endee, Precision

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

client.create_index(
    name="safetylens_index",
    dimension=1536,
    space_type="cosine",
    precision=Precision.INT8
)
```

### 2. Storing Document Chunks as Vectors
```python
index.upsert([{
    "id": "chunk_00001",
    "vector": [0.12, -0.34, ...],
    "meta": {
        "text": "The safe exposure limit for CO is 25 PPM...",
        "doc_name": "safety_manual.pdf",
        "chunk_index": 1
    }
}])
```

### 3. Semantic Search at Query Time
```python
results = index.query(
    vector=query_vector,
    top_k=5,
    ef=128,
    include_vectors=False
)
# Returns most semantically similar chunks instantly
```

---

## 🚀 Setup & Execution Instructions

### Prerequisites
- Python 3.9+
- Docker
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Step 1: Star & Fork Endee Repository
```bash
# 1. Go to https://github.com/endee-io/endee
# 2. Click Star ⭐
# 3. Fork to your account
# 4. Clone your fork
git clone https://github.com/YOUR_USERNAME/endee.git
```

### Step 2: Start Endee Vector Database
```bash
docker run -d -p 8080:8080 --name endee-server endeeio/endee-server:latest
```

Verify:
```bash
curl http://localhost:8080/api/v1/index/list
# Expected: {"indexes": []}
```

### Step 3: Clone & Setup VectoRAG
```bash
git clone https://github.com/Mohammadakbaralibaig/VectoRAG-.git
cd VectoRAG-

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Step 4: Configure API Key
Create a `.env` file:
```
GROQ_API_KEY=your_groq_key_here
```

### Step 5: Run the App
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure
```
VectoRAG/
├── app.py                      # Streamlit UI & main app logic
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed)
├── .streamlit/
│   └── config.toml            # Streamlit theme config
└── utils/
    ├── __init__.py
    ├── document_processor.py  # PDF/TXT extraction & chunking
    ├── embedder.py            # Vector embedding generation
    ├── vector_store.py        # Endee integration
    └── llm.py                 # Groq LLM answer generation
```

---

## ✨ Features

- 📄 PDF & TXT document support
- ⚡ Endee HNSW vector search — fast semantic retrieval
- 💬 Natural language Q&A — ask anything about your document
- 📊 Real-time stats — chunks indexed, queries run, vector dims
- 🎨 Premium UI — gradient design, chat bubbles, smooth animations
- 🔄 Session management — reset and upload new documents

---

## 🧠 RAG Pipeline Explained

**Chunking:** Documents split using sliding window (400 words, 50 word overlap) to preserve context at boundaries.

**Vector Search:** Endee uses HNSW (Hierarchical Navigable Small World) indexing for approximate nearest neighbor search with cosine similarity — finding semantically similar chunks in milliseconds.

**Answer Generation:** Top-K retrieved chunks passed as context to Groq Llama 3.1 with a structured prompt that grounds answers strictly in the document.

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| [Endee](https://github.com/endee-io/endee) | latest | Vector database & ANN search |
| [Streamlit](https://streamlit.io) | ≥1.32 | Web UI |
| [Groq](https://groq.com) | ≥1.1 | LLM inference |
| [pypdf](https://pypdf.readthedocs.io) | ≥4.0 | PDF parsing |
| Docker | 20.10+ | Running Endee server |

---

## 📸 Demo

**Upload a safety manual → Index into Endee → Ask questions → Get instant answers**

*Example query: "What is the safe exposure limit for CO?"*
*Answer: "The safe exposure limit for CO is 25 PPM over 8 hours."*

---

*Built for Endee On-Campus Drive · SDE/ML Engineer Position · 2026*