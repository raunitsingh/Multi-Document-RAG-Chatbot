import os
import json
import streamlit as st

from rag import (
    load_embedding_model,
    load_vectorstore,
    build_conversational_chain
)


def main():

    # Load config
    with open("config/config.json", "r") as f:
        config = json.load(f)

    os.environ["GROQ_API_KEY"] = config["GROQ_API_KEY"]

    # Streamlit setup
    st.set_page_config(page_title="Multi-Document RAG", page_icon="📚")
    st.title("📚 Multi-Document RAG Chatbot")

    if "chain" not in st.session_state:
        embedding_model = load_embedding_model(config["embedding_model"])
        vectorstore = load_vectorstore("vector_db_dir", embedding_model)
        st.session_state.chain = build_conversational_chain(config, vectorstore)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something...")

    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain(
                    {"question": user_input}
                )
                answer = response["answer"]
                st.markdown(answer)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )


if __name__ == "__main__":
    main()