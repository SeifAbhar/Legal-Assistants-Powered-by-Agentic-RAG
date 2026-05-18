import os
import subprocess
import sys
from pathlib import Path
from langchain_community.vectorstores import FAISS
from .embedder import get_embeddings
from .data_loader import load_case_law_chunks, load_contracts_chunks
sys.path.append(str(Path(__file__).parent.parent))
import config

def _run_token_chunker(input_dir: str, output_file: str):
    script = Path(__file__).parent.parent / "utils" / "token_chunker.py"
    cmd = [
        sys.executable, str(script),
        "--input-dir", input_dir,
        "--output-file", output_file,
        "--chunk-size", str(config.CHUNK_TOKENS),
        "--overlap", str(config.CHUNK_OVERLAP_TOKENS),
        "--model", config.EMBEDDING_MODEL
    ]
    subprocess.run(cmd, check=True)

def ensure_chunks():
    if not os.path.exists(config.CHUNKS_CASE_LAW_JSONL):
        print("Generating chunks for case law...")
        _run_token_chunker(config.CASE_LAW_DIR, config.CHUNKS_CASE_LAW_JSONL)
    if not os.path.exists(config.CHUNKS_CONTRACTS_JSONL):
        print("Generating chunks for contracts...")
        _run_token_chunker(config.CONTRACTS_DIR, config.CHUNKS_CONTRACTS_JSONL)

def build_all_stores():
    ensure_chunks()
    case_chunks = load_case_law_chunks()
    contract_chunks = load_contracts_chunks()
    embeddings = get_embeddings()

    case_db = FAISS.from_documents(case_chunks, embeddings)
    case_db.save_local(config.VECTOR_CASE_LAW)
    print(f"Case law index saved to: {config.VECTOR_CASE_LAW}")

    contract_db = FAISS.from_documents(contract_chunks, embeddings)
    contract_db.save_local(config.VECTOR_CONTRACTS)
    print(f"Contracts index saved to: {config.VECTOR_CONTRACTS}")

def load_case_law_store():
    embeddings = get_embeddings()
    return FAISS.load_local(config.VECTOR_CASE_LAW, embeddings, allow_dangerous_deserialization=True)

def load_contracts_store():
    embeddings = get_embeddings()
    return FAISS.load_local(config.VECTOR_CONTRACTS, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    build_all_stores()