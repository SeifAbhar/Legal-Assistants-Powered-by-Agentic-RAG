import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

def load_documents_from_folder(folder_path: str):
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(path)
            else:
                continue
            docs.extend(loader.load())
    return docs

def load_and_split_all():
    """Load case law & contracts separately, split, return two lists."""
    case_law_raw = load_documents_from_folder(config.CASE_LAW_DIR)
    contracts_raw = load_documents_from_folder(config.CONTRACTS_DIR)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    case_law_chunks = splitter.split_documents(case_law_raw)
    contract_chunks = splitter.split_documents(contracts_raw)
    
    # Tag metadata for filtering later
    for chunk in case_law_chunks:
        chunk.metadata["doc_type"] = "case_law"
    for chunk in contract_chunks:
        chunk.metadata["doc_type"] = "contract"
    
    return case_law_chunks, contract_chunks