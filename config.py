import os

DATA_DIR = "data"
CASE_LAW_DIR = os.path.join(DATA_DIR, "case_law")
CONTRACTS_DIR = os.path.join(DATA_DIR, "contracts")

VECTOR_CASE_LAW = "faiss_case_law"
VECTOR_CONTRACTS = "faiss_contracts"

EMBEDDING_MODEL = "text-embedding-3-small"   # or "all-mpnet-base-v2"
LLM_MODEL = "gpt-4o"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200