from typing import List
from langchain_text_splitters import CharacterTextSplitter


def split_documents(documents: List, chunk_size: int, chunk_overlap: int):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(documents)