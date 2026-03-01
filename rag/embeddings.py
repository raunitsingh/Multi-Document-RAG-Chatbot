from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)