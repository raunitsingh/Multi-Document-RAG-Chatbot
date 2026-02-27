"""
Multi-Document RAG Chatbot
Document Vectorization Pipeline

Architecture Flow:
1. Document Ingestion
2. Preprocessing & Chunking
3. Embedding Generation
4. Vector Indexing (ChromaDB)
5. Persistence Layer
"""

import os
import json
from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# Global Setup
# ============================================================

working_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(working_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

DATA_DIR = os.path.join(working_dir, "data")
VECTOR_DB_DIR = os.path.join(working_dir, "vector_db_dir")


# ============================================================
# 1️⃣ DOCUMENT INGESTION MODULE
# Load PDFs from data directory
# ============================================================

def load_documents() -> List:
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError("❌ 'data' directory not found.")

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        raise ValueError("❌ No PDF files found in 'data' directory.")

    print(f"📂 Found PDF files: {pdf_files}")

    loader = DirectoryLoader(
        path=DATA_DIR,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()
    print(f"✅ Loaded {len(documents)} document pages")

    return documents


# ============================================================
# 2️⃣ PREPROCESSING & CHUNKING
# Split documents into overlapping text chunks
# ============================================================

def split_documents(documents: List):
    text_splitter = CharacterTextSplitter(
        chunk_size=config.get("chunk_size", 2000),
        chunk_overlap=config.get("chunk_overlap", 500)
    )

    text_chunks = text_splitter.split_documents(documents)

    print(f"✂️ Created {len(text_chunks)} text chunks")
    return text_chunks


# ============================================================
# 3️⃣ EMBEDDING GENERATION
# Load HuggingFace embedding model
# ============================================================

def load_embedding_model():
    model_name = config.get(
        "embedding_model",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    print(f"🧠 Loading embedding model: {model_name}")

    return HuggingFaceEmbeddings(model_name=model_name)


# ============================================================
# 4️⃣ VECTOR INDEXING (CHROMADB)
# Create and persist vector database
# ============================================================

def create_vectorstore(text_chunks, embedding_model):
    vectordb = Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_DIR
    )

    print(f"💾 Vector database stored at: {VECTOR_DB_DIR}")
    return vectordb


# ============================================================
# 5️⃣ PIPELINE ORCHESTRATOR
# Full vectorization workflow
# ============================================================

def vectorize_documents():
    try:
        print("🚀 Starting document vectorization pipeline...\n")

        documents = load_documents()
        text_chunks = split_documents(documents)
        embedding_model = load_embedding_model()
        vectordb = create_vectorstore(text_chunks, embedding_model)

        print("\n🎉 Documents successfully vectorized!")
        return vectordb

    except Exception as e:
        print(f"\n❌ Error during vectorization: {e}")
        return None


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    vectorize_documents()