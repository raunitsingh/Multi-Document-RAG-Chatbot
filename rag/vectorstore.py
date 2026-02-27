import os
from langchain_chroma import Chroma


def create_vectorstore(documents, embedding_model, persist_directory: str):
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    return vectordb


def load_vectorstore(persist_directory: str, embedding_model):
    if not os.path.exists(persist_directory):
        raise FileNotFoundError("Vector DB not found. Run indexing first.")

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )