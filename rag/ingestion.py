import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


def load_documents(data_dir: str) -> List:
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Data directory not found.")

    loader = DirectoryLoader(
        path=data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()
    return documents