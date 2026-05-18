import os

DATA_DIR = "data"
CASE_LAW_DIR = os.path.join(DATA_DIR, "case_law")
CONTRACTS_DIR = os.path.join(DATA_DIR, "contracts")

CHUNKS_CASE_LAW_JSONL = os.path.join(DATA_DIR, "chunks_case_law.jsonl")
CHUNKS_CONTRACTS_JSONL = os.path.join(DATA_DIR, "chunks_contracts.jsonl")

VECTOR_CASE_LAW = "faiss_case_law"
VECTOR_CONTRACTS = "faiss_contracts"

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

CHUNK_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 50