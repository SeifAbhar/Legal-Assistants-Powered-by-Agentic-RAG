import json
from typing import List
from langchain.schema import Document
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

def _load_jsonl_chunks(jsonl_path: str, doc_type: str) -> List[Document]:
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            docs.append(Document(
                page_content=chunk["text"],
                metadata={
                    "source": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "tokens": chunk.get("tokens", 0),
                    "doc_type": doc_type
                }
            ))
    return docs

def load_case_law_chunks() -> List[Document]:
    return _load_jsonl_chunks(config.CHUNKS_CASE_LAW_JSONL, "case_law")

def load_contracts_chunks() -> List[Document]:
    return _load_jsonl_chunks(config.CHUNKS_CONTRACTS_JSONL, "contracts")

def load_and_split_all():
    return load_case_law_chunks(), load_contracts_chunks()