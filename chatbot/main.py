"""
Multi-Document RAG Chatbot
Main Application File

Architecture Flow:
1. Document Ingestion (Load Vector DB)
2. Retrieval Setup
3. Query Processing
4. Response Generation
5. Streamlit UI Layer
"""

import os
import json
import streamlit as st

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from vectorize_documents import embeddings


# ============================================================
# 1️⃣ DOCUMENT INGESTION MODULE
# Load existing persistent vector database
# ============================================================

def load_vectorstore(working_dir: str):
    persist_directory = os.path.join(working_dir, "vector_db_dir")

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            "Vector database not found. Run vectorize_documents.py first."
        )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore


# ============================================================
# 2️⃣ RETRIEVAL SETUP
# Configure retriever and memory
# ============================================================

def setup_retriever(vectorstore, top_k: int = 5):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )


def setup_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )


