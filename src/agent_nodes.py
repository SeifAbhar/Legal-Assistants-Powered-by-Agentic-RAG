import json
from .agent_state import AgentState
from .tools import search_case_law, search_contracts
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import config

llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)

def decompose_query(state: AgentState) -> AgentState:
    system = """You are a legal research expert. Given a legal question, decompose it into 2-4 specific sub-queries that will help retrieve relevant case law and contract clauses. Each sub-query should target a different aspect.

Output as a JSON list of strings. Example:
["force majeure definition case law", "contract clause force majeure events", "defendant force majeure burden of proof"]"""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"Decompose this legal question: {state['query']}")])
    try:
        sub_queries = json.loads(response.content)
    except:
        sub_queries = [state["query"]]
    return {**state, "sub_queries": sub_queries, "retrieval_attempts": 0, "context_sufficient": False}

def retrieve_documents(state: AgentState) -> AgentState:
    all_case = []
    all_contract = []
    for sq in state["sub_queries"]:
        all_case.append(search_case_law.invoke(sq))
        all_contract.append(search_contracts.invoke(sq))
    return {
        **state,
        "retrieved_case_law": state.get("retrieved_case_law", []) + all_case,
        "retrieved_contracts": state.get("retrieved_contracts", []) + all_contract,
        "retrieval_attempts": state["retrieval_attempts"] + 1
    }

def evaluate_context(state: AgentState) -> AgentState:
    case_context = "\n\n".join(state["retrieved_case_law"][-5:])
    contract_context = "\n\n".join(state["retrieved_contracts"][-5:])
    system = """You are a legal research evaluator. Review the retrieved case law and contract excerpts. Determine if there is sufficient context to answer the original question comprehensively.

Output ONLY: "SUFFICIENT" or "INSUFFICIENT: <reason>" """
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"""
Original Query: {state['query']}
Sub-queries: {state['sub_queries']}
Retrieval Attempt: {state['retrieval_attempts']}/{state.get('max_retrieval_attempts', 3)}

Case Law Retrieved:
{case_context[:1000]}

Contracts Retrieved:
{contract_context[:1000]}
""")])
    is_sufficient = response.content.strip().upper().startswith("SUFFICIENT")
    return {**state, "context_sufficient": is_sufficient}

def generate_answer(state: AgentState) -> AgentState:
    case_context = "\n\n".join(state["retrieved_case_law"])
    contract_context = "\n\n".join(state["retrieved_contracts"])
    system = """You are a senior legal associate. Using the provided case law and contract excerpts, answer the legal question with proper citations.

Structure your answer as:
1. **Summary**: 2-3 sentence overview
2. **Legal Analysis**: Detailed reasoning with citations
3. **Relevant Case Law**: Key precedents cited
4. **Relevant Contract Clauses**: Applicable provisions
5. **Conclusion**: Clear final determination

Always cite the source document and page when possible. If information is insufficient, explicitly state what is missing."""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"""
Question: {state['query']}

Case Law:
{case_context[:3000]}

Contracts:
{contract_context[:3000]}
""")])
    sources = []
    for doc in state["retrieved_case_law"] + state["retrieved_contracts"]:
        if "Source:" in doc:
            sources.append(doc.split("Source:")[1].split("\n")[0].strip())
    return {**state, "answer": response.content, "sources": list(set(sources))}

def draft_clause(state: AgentState) -> AgentState:
    system = """You are a legal drafter. Based on the analysis above, draft:
1. A sample clause addressing the legal issue (if applicable)
2. A potential counter-argument the opposing party might raise

Be precise and cite applicable precedents."""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"""
Original Query: {state['query']}
Analysis: {state['answer'][:2000]}
""")])
    return {**state, "draft_clause": response.content}