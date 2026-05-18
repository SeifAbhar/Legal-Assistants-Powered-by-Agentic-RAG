from langchain_openai import OpenAIEmbeddings
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

def get_embeddings():
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)