import argparse
from .agent_graph import legal_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    initial_state = {
        "query": args.query,
        "sub_queries": [],
        "retrieved_case_law": [],
        "retrieved_contracts": [],
        "context_sufficient": False,
        "retrieval_attempts": 0,
        "max_retrieval_attempts": args.max_attempts,
        "answer": None,
        "sources": [],
        "draft_clause": None
    }

    print(f"\n🔍 Processing: {args.query}\n{'='*60}")
    for step, state in legal_agent.stream(initial_state):
        node_name = list(step.keys())[0]
        print(f"✓ {node_name}")

    final = legal_agent.invoke(initial_state)
    print("\n" + "="*60)
    print("📋 FINAL ANSWER")
    print("="*60)
    print(final["answer"])
    print("\n📚 SOURCES:", final["sources"])
    if final.get("draft_clause"):
        print("\n📝 DRAFT CLAUSE / COUNTER-ARGUMENT:")
        print(final["draft_clause"])