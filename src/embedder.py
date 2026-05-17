from langchain_openai import OpenAIEmbeddings
import config

def get_embeddings():
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)