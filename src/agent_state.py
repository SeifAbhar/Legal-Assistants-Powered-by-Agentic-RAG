from typing import TypedDict, List, Optional, Annotated
import operator

class AgentState(TypedDict):
    query: str
    sub_queries: List[str]
    retrieved_case_law: Annotated[List[str], operator.add]
    retrieved_contracts: Annotated[List[str], operator.add]
    context_sufficient: bool
    retrieval_attempts: int
    max_retrieval_attempts: int
    answer: Optional[str]
    sources: Optional[List[str]]
    draft_clause: Optional[str]