"""
main.py
-------
Streamlit application entry point for the Multi-Document RAG Chatbot.

The app:
  1. Loads config and the persisted ChromaDB vector store on startup.
  2. Maintains conversation history in Streamlit session state.
  3. On each user message, runs the full RAG + claim validation pipeline.
  4. Displays the grounded answer with source citations.

Launch:
    streamlit run main.py
"""

import streamlit as st

from utils.config_loader import load_config
from utils.logger import get_logger
from src.vectorizer import load_vector_store
from src.query_processor import process_query
from src.retriever import retrieve_relevant_chunks, format_context
from src.claim_validator import validate_claims
from src.response_generator import generate_response, build_citation_footer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Load Config and Vector Store (cached — runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading configuration and vector store...")
def initialise():
    """Load config and vector store once and cache across reruns."""
    config = load_config("config.json")
    vector_store = load_vector_store(config)
    return config, vector_store


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(config: dict, vector_store) -> None:
    with st.sidebar:
        st.title("🧠 RAG Chatbot")
        st.markdown("---")

        st.subheader("ℹ️ System Info")
        st.markdown(f"**Model:** `{config['model_name']}`")
        st.markdown(f"**Embeddings:** `all-MiniLM-L6-v2`")
        st.markdown(f"**Vector DB:** ChromaDB")

        try:
            count = vector_store._collection.count()
            st.markdown(f"**Indexed Chunks:** `{count}`")
        except Exception:
            st.markdown("**Indexed Chunks:** `unavailable`")

        st.markdown("---")

        st.subheader("⚙️ Retrieval Settings")
        top_k = st.slider(
            "Top-K Chunks", min_value=1, max_value=10,
            value=config.get("retrieval_top_k", 5), step=1
        )
        threshold = st.slider(
            "Similarity Threshold", min_value=0.0, max_value=1.0,
            value=config.get("similarity_threshold", 0.7), step=0.05
        )

        st.markdown("---")

        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.query_history = []
            st.rerun()

        return top_k, threshold


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main():
    # Initialise
    try:
        config, vector_store = initialise()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run `python vectorize_documents.py` first, then restart the app.")
        st.stop()
    except KeyError as e:
        st.error(f"Configuration error: {e}")
        st.stop()

    # Sidebar — returns user-adjusted retrieval settings
    top_k, threshold = render_sidebar(config, vector_store)

    # Session state initialisation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    # Title
    st.title("🧠 Multi-Document RAG Chatbot")
    st.caption(
        "Ask questions about your documents. "
        "Every answer is grounded in the source material with full citations."
    )
    st.markdown("---")

    # Render existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask a question about your documents..."):

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Run pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):

                # 1. Process query
                processed_query = process_query(
                    raw_query=user_input,
                    history=st.session_state.query_history,
                )

                # 2. Retrieve relevant chunks
                chunks = retrieve_relevant_chunks(
                    query=processed_query,
                    vector_store=vector_store,
                    top_k=top_k,
                    similarity_threshold=threshold,
                )

                # 3. Validate claims
                validation_report = validate_claims(chunks)

                # 4. Format context
                context = format_context(chunks)

                # 5. Build conversation history for multi-turn context
                # (keep last 6 turns to stay within token limits)
                api_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-6:]
                    if m["role"] in ("user", "assistant")
                ]

                # 6. Generate response
                answer = generate_response(
                    query=processed_query,
                    context=context,
                    validation_report=validation_report,
                    config=config,
                    conversation_history=api_history,
                )

                # 7. Build citation footer
                citations = build_citation_footer(chunks)
                full_response = answer + citations

            # Display response
            st.markdown(full_response)

            # Show validation info if contradictions were detected
            if validation_report.get("has_contradiction"):
                st.warning(
                    "⚠️ Potential contradictions were detected across sources. "
                    "Review the citations above carefully."
                )

        # Update session state
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        st.session_state.query_history.append(processed_query)


if __name__ == "__main__":
    main()