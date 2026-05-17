from langchain_community.vectorstores import FAISS
from .embedder import get_embeddings
from .data_loader import load_and_split_all
import config
import os

def build_all_stores():
    """Build and persist separate FAISS indexes."""
    case_chunks, contract_chunks = load_and_split_all()
    embeddings = get_embeddings()
    
    # Case law store
    case_db = FAISS.from_documents(case_chunks, embeddings)
    case_db.save_local(config.VECTOR_CASE_LAW)
    
    # Contracts store
    contract_db = FAISS.from_documents(contract_chunks, embeddings)
    contract_db.save_local(config.VECTOR_CONTRACTS)
    
    print(f"Case law index: {config.VECTOR_CASE_LAW}")
    print(f"Contracts index: {config.VECTOR_CONTRACTS}")

def load_case_law_store():
    embeddings = get_embeddings()
    return FAISS.load_local(config.VECTOR_CASE_LAW, embeddings, allow_dangerous_deserialization=True)

def load_contracts_store():
    embeddings = get_embeddings()
    return FAISS.load_local(config.VECTOR_CONTRACTS, embeddings, allow_dangerous_deserialization=True)