from langgraph.graph import StateGraph, END
from .agent_state import AgentState
from .agent_nodes import decompose_query, retrieve_documents, evaluate_context, generate_answer, draft_clause

def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("decompose", decompose_query)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("evaluate", evaluate_context)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("draft", draft_clause)

    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "evaluate")

    def should_continue(state):
        if not state["context_sufficient"] and state["retrieval_attempts"] < state.get("max_retrieval_attempts", 3):
            return "retrieve"
        return "generate"

    workflow.add_conditional_edges("evaluate", should_continue, {"retrieve": "retrieve", "generate": "generate"})
    workflow.add_edge("generate", "draft")
    workflow.add_edge("draft", END)
    return workflow.compile()

legal_agent = build_agent()