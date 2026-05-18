#!/usr/bin/env python3
import argparse
import subprocess
import sys
from dotenv import load_dotenv
load_dotenv()

from src.vector_store import build_all_stores
from src.agent_graph import legal_agent
from src.evaluation import run_evaluation

def cmd_build():
    print("Building vector stores...")
    build_all_stores()
    print("Done.")

def cmd_query():
    question = input("Enter your legal question: ").strip()
    if not question:
        print("No question entered.")
        return
    initial_state = {
        "query": question,
        "sub_queries": [],
        "retrieved_case_law": [],
        "retrieved_contracts": [],
        "context_sufficient": False,
        "retrieval_attempts": 0,
        "max_retrieval_attempts": 3,
        "answer": None,
        "sources": [],
        "draft_clause": None
    }
    print("\nProcessing...")
    final = legal_agent.invoke(initial_state)
    print("\n" + "="*60)
    print("📋 FINAL ANSWER")
    print("="*60)
    print(final["answer"])
    print("\n📚 SOURCES:", final["sources"])
    if final.get("draft_clause"):
        print("\n📝 DRAFT CLAUSE / COUNTER-ARGUMENT:")
        print(final["draft_clause"])

def cmd_evaluate():
    print("Running evaluation suite on all test cases...")
    run_evaluation()

def cmd_ui():
    print("Launching Streamlit UI...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal AI Assistant")
    parser.add_argument("command", choices=["build", "query", "evaluate", "ui"],
                        help="Action to perform")
    args = parser.parse_args()

    if args.command == "build":
        cmd_build()
    elif args.command == "query":
        cmd_query()
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "ui":
        cmd_ui()
    else:
        parser.print_help()