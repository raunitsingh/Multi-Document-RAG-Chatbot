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


# ============================================================
# 3️⃣ QUERY PROCESSING + 4️⃣ RESPONSE GENERATION
# Build Conversational RAG Chain
# ============================================================

def build_conversational_chain(config, retriever, memory):
    llm = ChatGroq(
        model=config["model_name"],
        temperature=config.get("temperature", 0),
        max_tokens=config.get("max_tokens", 1024)
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False,
        return_source_documents=True,
        output_key="answer"
    )

    return chain


# ============================================================
# 5️⃣ USER INTERFACE LAYER (STREAMLIT)
# ============================================================

def main():

    # ------------------ Config Setup ------------------
    working_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(working_dir, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    os.environ["GROQ_API_KEY"] = config["GROQ_API_KEY"]

    # ------------------ Streamlit Setup ------------------
    st.set_page_config(
        page_title="Multi-Document RAG Chat",
        page_icon="📚",
        layout="centered"
    )

    st.title("📚 Multi-Document RAG Chatbot")
    st.caption("Semantic Search + Groq Llama-3.3-70B")

    # ------------------ Session State ------------------
    if "chat_history_ui" not in st.session_state:
        st.session_state.chat_history_ui = []

    if "chain" not in st.session_state:
        vectorstore = load_vectorstore(working_dir)
        retriever = setup_retriever(vectorstore, top_k=config.get("retrieval_top_k", 5))
        memory = setup_memory()
        st.session_state.chain = build_conversational_chain(config, retriever, memory)

    # ------------------ Render Previous Messages ------------------
    for message in st.session_state.chat_history_ui:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ------------------ User Input ------------------
    user_input = st.chat_input("Ask questions about your documents...")

    if user_input:
        # Display user message
        st.session_state.chat_history_ui.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                try:
                    response = st.session_state.chain(
                        {"question": user_input}
                    )

                    answer = response["answer"]
                    sources = response.get("source_documents", [])

                    st.markdown(answer)

                    # Optional: Show citations
                    if sources:
                        with st.expander("📎 Source References"):
                            for i, doc in enumerate(sources):
                                source_name = doc.metadata.get("source", "Unknown")
                                page = doc.metadata.get("page", "N/A")
                                st.write(f"**Source {i+1}:** {source_name} | Page: {page}")

                    st.session_state.chat_history_ui.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history_ui.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    main()