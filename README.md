# Multi-Document-RAG-Chatbot

*An advanced Retrieval-Augmented Generation system for automated, high-performance, multi-document intelligence.*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.9-blue.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

# **1. Problem Statement**

Organizations, researchers, and enterprise teams increasingly depend on large collections of unstructured documents such as PDF reports, research papers, policies, manuals, and business documents. Extracting accurate, context-aware information from these documents is difficult due to the following challenges:

1. **Manual Reading Overhead**
   Users must search across multiple documents, consuming significant time and increasing risk of missed information.

2. **Keyword Search Limitations**
   Traditional keyword-based search returns irrelevant results because it cannot understand semantic meaning, intent, or context.

3. **Fragmented Knowledge**
   Different PDFs may contain interrelated concepts, but users cannot easily connect this information across documents.

4. **Static Search Systems**
   Most tools lack interactive reasoning, natural dialogue, follow-up questioning, and dynamic adaptation to user needs.

5. **Lack of Source Transparency**
   Many AI chat systems provide answers without traceability to original sources, limiting reliability and trust.

6. **High Inference Costs & Latency**
   Enterprise LLMs often require expensive cloud-hosted models and produce slow responses.

**Therefore, a system is required that:**

* Aggregates multiple documents
* Performs semantic-level search
* Understands relationships between documents
* Provides natural conversational responses
* Remains transparent and traceable
* Achieves high performance at low cost

This leads to the development of the **Multi-Document RAG Chatbot**.

---

# **2. Solution Overview**

The proposed solution is a **Multi-Document Retrieval-Augmented Generation (RAG) Chatbot**, capable of ingesting multiple documents, converting them into vector embeddings, retrieving relevant chunks, and generating accurate, context-rich answers powered by a high-speed LLM.

**Key capabilities:**

* Multi-document ingestion and semantic understanding
* Advanced embedding-based retrieval using ChromaDB
* Ultra-fast reasoning with Groq Llama-3.3-70B
* Natural conversational interface in Streamlit
* Full citation and source-traceability
* Configurable architecture (chunking, retrieval parameters, model parameters)

The system ensures that every answer is grounded in the actual documents, enabling dependable business and research usage.

---

# **3. Features**

## **3.1 Multi-Document Intelligence**

* Automatic scanning of directories for PDF files
* Extraction of raw text using PyPDFLoader
* Conversion to structured document units
* Chunking into 2000-character blocks with 500-character overlap
* Cross-document semantic mapping
* Consistent performance regardless of the number of documents

## **3.2 Advanced Semantic Search**

* Vector embeddings computed using HuggingFace Sentence-BERT
* ChromaDB persistent vector store
* Hybrid similarity metrics (cosine, L2 distance, dot-product)
* Retrieval of top-k relevant contextual chunks

## **3.3 Natural Conversational Interface**

* Continuous conversation memory
* Ability to ask follow-up queries
* Business-language query support
* Streaming-style interactive UI (future enhancement ready)

## **3.4 Comprehensive Source Attribution**

* Exact document names and page references
* Ranked citations for every retrieved chunk
* Confidence ranking based on similarity score
* High trustworthiness for enterprise auditing

---

# **4. System Architecture **

The architecture follows a modular RAG pipeline:

### **4.1 Document Ingestion Module**

* Scans folder for PDF files
* Applies preprocessing such as removal of special characters
* Converts PDF → Text using PyPDFLoader

### **4.2 Vectorization Pipeline**

* Intelligent chunk splitting
* Embedding generation using Sentence-BERT
* ChromaDB vector indexing
* Persistent disk storage

### **4.3 Query Processing**

* Cleans user query
* Embeds query using the same embedding model
* Executes top-k vector search
* Assembles a context document containing the relevant chunks

### **4.4 Response Generation**

* Sends prompt + retrieved context to Groq Llama-3.3-70B
* Generates grounded answers
* Attaches citation indices for user verification

### **4.5 User Interface Layer**

* Clean Streamlit chat interface
* History preservation
* PDF indexing status reports
* Real-time answer display

---

# **5. Technology Stack (Detailed)**

| Layer                 | Technologies                             | Purpose                              |
| --------------------- | ---------------------------------------- | ------------------------------------ |
| **Frontend**          | Streamlit 1.38.0                         | Real-time chat UI                    |
| **LLM**               | Groq API + Llama-3.3-70b-versatile       | High-speed reasoning                 |
| **Embeddings**        | Sentence-transformers (all-MiniLM-L6-v2) | Semantic understanding               |
| **Vector Store**      | ChromaDB                                 | Persistent embedding storage         |
| **Document Parsing**  | PyPDFLoader, LangChain loaders           | PDF → text conversion                |
| **Backend Framework** | LangChain                                | RAG pipeline orchestration           |
| **Environment**       | Python 3.8+, virtualenv                  | Stable and isolated execution        |
| **Storage**           | Local ChromaDB directory                 | Lightweight local vector persistence |

---

# **6. Installation**

### Prerequisites
- Python 3.8 or higher
- Groq API account and API key
- 8GB RAM minimum (16GB recommended)
- 2GB free disk space

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/Multi_Doc_RAG_Chatbot.git
   cd Multi_Doc_RAG_Chatbot
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure API Keys**
   Create `config.json` in the project root:
   ```json
   {
       "GROQ_API_KEY": "your-groq-api-key-here",
       "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
       "chunk_size": 2000,
       "chunk_overlap": 500,
       "model_name": "llama-3.3-70b-versatile",
       "temperature": 0
   }
   ```

5. **Prepare Documents**
   Place your PDF files in the `data/` directory:
   ```bash
   mkdir data
   # Copy your PDF files to data/ directory
   ```

6. **Vectorize Documents**
   ```bash
   python vectorize_documents.py
   ```
   Expected output:
   ```
   Found PDF files: ['paper1.pdf', 'paper2.pdf', 'paper3.pdf']
   Loaded 28 documents
   Created 28 text chunks
   Documents successfully vectorized and stored in vector_db_dir
   ```

7. **Launch Application**
   ```bash
   streamlit run main.py
   ```
   Access the application at `http://localhost:8501`

---
