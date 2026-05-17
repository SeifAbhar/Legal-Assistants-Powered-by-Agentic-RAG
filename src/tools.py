from langchain_core.tools import tool
from .vector_store import load_case_law_store, load_contracts_store

@tool
def search_case_law(query: str, k: int = 5) -> str:
    """Search case law database for relevant precedents and judgments."""
    vectordb = load_case_law_store()
    docs = vectordb.similarity_search(query, k=k)
    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        results.append(f"[Case Law {i+1}] Source: {source} (Page {page})\n{doc.page_content[:500]}")
    return "\n\n".join(results)

@tool
def search_contracts(query: str, k: int = 5) -> str:
    """Search contract database for relevant clauses, NDAs, employment agreements."""
    vectordb = load_contracts_store()
    docs = vectordb.similarity_search(query, k=k)
    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        results.append(f"[Contract {i+1}] Source: {source} (Page {page})\n{doc.page_content[:500]}")
    return "\n\n".join(results)

tools = [search_case_law, search_contracts]