import argparse
from dotenv import load_dotenv
load_dotenv()

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

    final = legal_agent.invoke(initial_state)
    print(final["answer"])