# 🚀 Standalone RAG - Intelligent Document Q&A Platform

> An enterprise-grade Retrieval-Augmented Generation (RAG) application that transforms your documents into an intelligent knowledge base with instant answers.

![RAG UI](assets/ui.jpg)

---

## 🎯 What is Standalone RAG?

Standalone RAG is a powerful document intelligence platform that allows you to upload documents (PDFs, Word docs, text files) and ask natural language questions about them. The system uses advanced AI to retrieve relevant information and generate accurate, contextual answers based on your document content.

**Your Personal AI on Your System** - Run it locally, own your data, unlimited queries.

### Key Features

✨ **Smart Document Processing**
- Automatic chunking and vectorization of documents
- Support for multiple file formats (PDF, DOCX, TXT)
- Intelligent metadata extraction

🔍 **Advanced Retrieval**
- Semantic search using vector embeddings
- Multi-stage ranking with reranking
- Context-aware document matching

🤖 **AI-Powered Responses**
- Natural language query understanding
- Accurate answer generation with source citations
- Multi-turn conversation support

💬 **Interactive Chat Interface**
- Real-time document upload and processing
- Instant question answering
- Source document highlighting

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│         Streamlit Web Interface (UI)                │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│          FastAPI Backend (REST API)                 │
│  • Chat management                                  │
│  • Document upload & processing                     │
│  • Query handling                                   │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┬──────────────┐
        │                     │              │
┌───────▼────────┐  ┌────────▼────────┐  ┌──▼──────────────┐
│  Celery Worker │  │  Vector Store   │  │  PostgreSQL DB  │
│ (Async Tasks)  │  │  (Qdrant VectorDB) │  (Metadata)     │
└────────────────┘  └─────────────────┘  └─────────────────┘
        │                     │
        └──────────┬──────────┘
                   │
    ┌──────────────▼──────────────┐
    │  LLM Pipeline               │
    │  • Embedding Generation     │
    │  • Document Retrieval       │
    │  • Semantic Reranking       │
    │  • Answer Generation        │
    └─────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core Framework
- **Langchain** - LLM orchestration and chain management
- **FastAPI** - High-performance Python web framework
- **Streamlit** - Interactive web UI for end users

### Data & Storage
- **Qdrant** - Vector database for semantic search
- **PostgreSQL** - Relational database for metadata
- **Redis** - Caching and message broker

### AI & ML
- **OpenAI / Hugging Face** - Language models and embeddings
- **Sentence Transformers** - Local embedding generation
- **CrossEncoder** - Document reranking

### Infrastructure
- **Celery** - Asynchronous task processing
- **Uvicorn** - ASGI application server

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- PostgreSQL 13+
- Qdrant Vector Database
- Redis 6+ (optional, for Celery)

### Quick Start with UV

1. **Clone the repository**
```bash
git clone https://github.com/sahilsnghai/standalone-rag.git
cd standalone-rag
```

2. **Install UV package manager** (if not installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Install dependencies with UV**
```bash
uv sync
```

This will install all dependencies from `pyproject.toml` into a virtual environment.

4. **Configure environment**
```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# ============================================
# OPENAI Configuration (Optional)
# ============================================
OPENAI_API_KEY=sk-...
# Uses local embeddings by default if not set
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ============================================
# PostgreSQL Database
# ============================================
DATABASE_URL=postgresql://username:password@localhost:5432/rag_db
# Example: postgresql://postgres:postgres@localhost:5432/rag_db

# ============================================
# Qdrant Vector Database
# ============================================
QDRANT_URL=http://localhost:6333
# Make sure Qdrant is running on this URL
VECTOR_DB_PATH=./vector_db

# ============================================
# Application Settings
# ============================================
API_HOST=0.0.0.0
API_PORT=8000
UI_HOST=0.0.0.0
UI_PORT=8501

DEBUG=False
```

---

## 🚀 Running the Application

### Step 1: Start Celery Worker
```bash
./run_celery
```

This starts the background task processor for document processing.

### Step 2: Start Frontend & Backend
```bash
python run_app.py
```

This starts both:
- **FastAPI Backend** (http://localhost:8000)
- **Streamlit UI** (http://localhost:8501)

### Initial Setup
⏳ **The application will take a moment to start initially** as it:
- Initializes database connections
- Loads AI models
- Sets up vector store
- Prepares embeddings pipeline

Once you see:
```
Streamlit app is running at http://localhost:8501
```

You're ready to go! 🎉

---

## 💡 How to Use

### 1. Access the Application
Open your browser and go to: `http://localhost:8501`

### 2. Upload Documents (Optional)
- Click **"Upload document"** in the Documents panel
- Select PDF, DOCX, or TXT files
- Watch the real-time processing progress
- Documents are automatically chunked and indexed

### 3. Ask Questions
- Type your question in the message input
- Examples:
  - "What are the main skills mentioned?"
  - "Summarize the education section"
  - "What projects are listed?"
  - Ask anything about your documents!

### 4. Get Instant Answers
- Receive AI-powered answers based on your documents
- View source documents in the "Sources" section
- See relevance scores and chunk references
- Continue the conversation with follow-up questions

---


## ⚙️ Configuration

### Required Services

#### PostgreSQL Database
```bash
# Install PostgreSQL
# macOS: brew install postgresql
# Ubuntu: sudo apt-get install postgresql

# Start PostgreSQL
# macOS: brew services start postgresql
# Ubuntu: sudo systemctl start postgresql

# Create database
createdb rag_db

# Update DATABASE_URL in .env:
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
```

#### Qdrant Vector Database
```bash
# Using Docker (Recommended)
docker run -p 6333:6333 qdrant/qdrant

# Or download from: https://qdrant.tech/documentation/quick-start/
```

#### Redis (Optional, for Celery)
```bash
# macOS: brew install redis
# Ubuntu: sudo apt-get install redis-server

# Start Redis
# macOS: brew services start redis
# Ubuntu: sudo systemctl start redis-server
```

### Environment Variables

See `.env.example` for all available options:

```env
# Essential Configuration
OPENAI_API_KEY=sk-...                 # Optional (uses local by default)
DATABASE_URL=postgresql://...         # Required
QDRANT_URL=http://localhost:6333     # Required
VECTOR_DB_PATH=./vector_db           # Required

# Application Ports
API_HOST=0.0.0.0
API_PORT=8000
UI_HOST=0.0.0.0
UI_PORT=8501

# Advanced Settings
DEBUG=False
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=6
RERANK_K=6
```

## 📊 Features in Detail

### Document Processing
- **Automatic Chunking**: Splits documents into 1000-token chunks with 200-token overlap
- **Multi-Format Support**: PDF, DOCX, TXT with OCR capability
- **Real-Time Processing**: See progress as documents are indexed

### Intelligent Retrieval
1. **Vector Search**: Find semantically similar content
2. **Reranking**: Rank results by relevance
3. **Context Assembly**: Combine chunks with metadata

### Answer Generation
- **Context-Aware**: Uses retrieved documents
- **Source Attribution**: Shows which documents were used
- **Multi-Turn**: Continuous conversation

---

## 📈 Performance

### Embedding Generation
- Local model: 20-50ms per document
- OpenAI: 100-200ms per document

### Document Retrieval
- Vector search: <100ms for 1M documents
- Full pipeline: 1-5 seconds per query

### Storage
- Each document: ~100-500KB (embeddings + metadata)
- 1000 documents: ~100-500MB

---

## 🤝 Contributing

Contributions welcome! 

1. Fork the repository
2. Create feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m 'Add YourFeature'`
4. Push branch: `git push origin feature/YourFeature`
5. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file


## 🙋 Support & Feedback

- **Issues**: [GitHub Issues](https://github.com/sahilsnghai/standalone-rag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sahilsnghai/standalone-rag/discussions)
- **Email**: Sahilsinghai5672@gmail.com



**Made with ❤️ by Sahil Singhai Jain**

⭐ If you find this helpful, star the repository!

