# src/evaluation.py
"""
Evaluation module for the Legal RAG Agent.

Measures:
- Faithfulness (does answer only use retrieved docs?)
- Task success rate (did the agent answer correctly?)
- Latency (total time + time to first token)
- Cost (token consumption & $)
"""

import time
import json
import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

from .agent_graph import legal_agent
import config

# ----------------------------------------------------------------------
# Test Suite – All 10 queries from the mock cases
# ----------------------------------------------------------------------
TEST_CASES = [
    {
        "case_id": "case1_force_majeure",
        "query": "Does COVID-19 qualify as force majeure under a supply contract clause listing natural disasters and governmental action?",
        "expected_keywords": ["force majeure", "COVID-19", "pandemic", "not covered", "natural disasters"]
    },
    {
        "case_id": "case2_non_compete",
        "query": "Enforceability of non-compete clauses under California Business and Professions Code Section 16600 for tech employees",
        "expected_keywords": ["California", "Section 16600", "non-compete", "void", "public policy"]
    },
    {
        "case_id": "case3_gdpr_breach",
        "query": "GDPR Article 32 security obligations for cloud data processors and liability under data processing agreements",
        "expected_keywords": ["Article 32", "GDPR", "data processor", "appropriate measures", "encryption"]
    },
    {
        "case_id": "case4_ip_assignment",
        "query": "Employee IP assignment clause enforceability for inventions developed on personal time under California Labor Code 2870",
        "expected_keywords": ["California Labor Code", "2870", "personal time", "invention", "not assignable"]
    },
    {
        "case_id": "case5_liquidated_damages",
        "query": "Liquidated damages clause enforceability test genuine pre-estimate of loss versus unenforceable penalty clause",
        "expected_keywords": ["liquidated damages", "penalty", "pre-estimate", "Cavendish", "unenforceable"]
    },
    {
        "case_id": "case6_auto_renewal",
        "query": "Auto-renewal clause enforceability conspicuous notice requirement SaaS enterprise contracts electronic signature",
        "expected_keywords": ["auto-renewal", "conspicuous", "notice", "electronic signature", "enforceable"]
    },
    {
        "case_id": "case7_whistleblower",
        "query": "Whistleblower retaliation wrongful termination at-will employment exceptions Sarbanes-Oxley private company",
        "expected_keywords": ["whistleblower", "Sarbanes-Oxley", "retaliation", "at-will", "protected activity"]
    },
    {
        "case_id": "case8_construction",
        "query": "Construction contract change order validity verbal instructions scope of work dispute unjust enrichment",
        "expected_keywords": ["change order", "written", "verbal", "scope", "unjust enrichment"]
    },
    {
        "case_id": "case9_trade_secrets",
        "query": "Trade secret misappropriation customer list Defend Trade Secrets Act injunctive relief departing employee",
        "expected_keywords": ["trade secret", "Defend Trade Secrets Act", "customer list", "injunction", "misappropriation"]
    },
    {
        "case_id": "case10_arbitration",
        "query": "Mandatory arbitration clause class action waiver unconscionability consumer contracts small value claims",
        "expected_keywords": ["arbitration", "class action waiver", "unconscionable", "small claims", "public policy"]
    }
]


# ----------------------------------------------------------------------
# Metric 1: Faithfulness (LLM as judge)
# ----------------------------------------------------------------------
def check_faithfulness(answer: str, context: str) -> bool:
    """
    Returns True if all factual claims in the answer are supported by the context.
    """
    judge = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    prompt = f"""You are a legal fact-checker. Given the following context (from legal documents), 
determine whether every factual claim in the answer is directly supported by the context.
Answer only 'YES' or 'NO'.

Context:
{context[:3000]}

Answer:
{answer}

Is the answer fully faithful to the context? (YES/NO):"""
    response = judge.invoke(prompt)
    return response.content.strip().upper() == "YES"


# ----------------------------------------------------------------------
# Metric 2: Task Success (keyword overlap + LLM verification)
# ----------------------------------------------------------------------
def check_success(answer: str, expected_keywords: List[str]) -> bool:
    """
    Simple check: all expected keywords appear in the answer (case-insensitive).
    Also uses an LLM to confirm the answer actually addresses the core question.
    """
    answer_lower = answer.lower()
    keywords_present = all(kw.lower() in answer_lower for kw in expected_keywords)
    if not keywords_present:
        return False
    
    # Extra verification: LLM confirms the answer is correct
    judge = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    prompt = f"""You are a legal evaluator. Given the answer below, does it directly and correctly address 
the legal question? Answer only 'YES' or 'NO'.

Answer:
{answer}

Expected topics: {', '.join(expected_keywords)}

Is the answer correct and complete? (YES/NO):"""
    response = judge.invoke(prompt)
    return response.content.strip().upper() == "YES"


# ----------------------------------------------------------------------
# Main evaluation runner
# ----------------------------------------------------------------------
def run_evaluation():
    results = []
    total_cost = 0.0
    total_tokens = 0
    total_time = 0.0
    successful = 0
    faithful = 0

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n🔎 Running case {i}/{len(TEST_CASES)}: {case['case_id']} ...")
        
        initial_state = {
            "query": case["query"],
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

        # Latency measurement
        start_time = time.perf_counter()
        with get_openai_callback() as cb:
            final_state = legal_agent.invoke(initial_state)
        latency = time.perf_counter() - start_time

        answer = final_state.get("answer", "")
        context_used = "\n\n".join(
            final_state.get("retrieved_case_law", []) + 
            final_state.get("retrieved_contracts", [])
        )

        # Metrics
        is_faithful = check_faithfulness(answer, context_used)
        is_success = check_success(answer, case["expected_keywords"])
        
        tokens = cb.total_tokens
        cost = cb.total_cost

        # Accumulate
        total_cost += cost
        total_tokens += tokens
        total_time += latency
        if is_success:
            successful += 1
        if is_faithful:
            faithful += 1

        results.append({
            "case_id": case["case_id"],
            "query": case["query"],
            "answer": answer[:300],   # first 300 chars for report
            "faithful": is_faithful,
            "success": is_success,
            "latency_sec": round(latency, 3),
            "tokens": tokens,
            "cost_usd": round(cost, 5)
        })

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("📊 EVALUATION REPORT")
    print("="*70)
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Task success rate: {successful}/{len(TEST_CASES)} = {successful/len(TEST_CASES)*100:.1f}%")
    print(f"Faithfulness rate: {faithful}/{len(TEST_CASES)} = {faithful/len(TEST_CASES)*100:.1f}%")
    print(f"Average latency: {total_time/len(TEST_CASES):.2f} seconds")
    print(f"Total tokens consumed: {total_tokens}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Average cost per query: ${total_cost/len(TEST_CASES):.4f}")
    print("="*70)

    # Optional: save detailed report to JSON
    with open("evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n📄 Detailed results saved to evaluation_report.json")

    # Check against thresholds from project doc
    print("\n📈 Project Thresholds Check:")
    avg_latency = total_time / len(TEST_CASES)
    avg_cost = total_cost / len(TEST_CASES)
    print(f"  Latency (< 2s): {'✅' if avg_latency < 2.0 else '❌'} ({avg_latency:.2f}s)")
    print(f"  Cost/query (< ~$0.10): {'✅' if avg_cost < 0.10 else '⚠️'} (${avg_cost:.4f})")
    print(f"  Success Rate (> 80%): {'✅' if successful/len(TEST_CASES) > 0.8 else '⚠️'}")
    print(f"  Faithfulness (> 90%): {'✅' if faithful/len(TEST_CASES) > 0.9 else '⚠️'}")

    return results


# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_evaluation()