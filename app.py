import streamlit as st
import time
from dotenv import load_dotenv
load_dotenv()

from src.agent_graph import legal_agent

st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️", layout="wide")

with st.sidebar:
    st.title("⚖️ Legal RAG Agent")
    st.markdown("Powered by **Agentic RAG** and LangGraph")
    st.markdown("---")
    max_attempts = st.slider("Max retrieval attempts", 1, 5, 3)
    st.markdown("---")
    st.markdown("### Example Questions")
    st.markdown("""
    - Does COVID-19 qualify as force majeure under a supply contract?
    - Are non-compete clauses enforceable under California law?
    - GDPR Article 32 obligations for cloud processors?
    """)

st.title("⚖️ Legal Assistant — Agentic RAG")
st.markdown("Ask any legal question. The agent will decompose, search case law & contracts, evaluate, and generate a cited answer.")

query = st.text_area("Enter your legal question:", height=100, placeholder="e.g., 'Can the defendant claim force majeure?'")

if st.button("🔍 Ask the Legal Assistant", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        initial_state = {
            "query": query,
            "sub_queries": [],
            "retrieved_case_law": [],
            "retrieved_contracts": [],
            "context_sufficient": False,
            "retrieval_attempts": 0,
            "max_retrieval_attempts": max_attempts,
            "answer": None,
            "sources": [],
            "draft_clause": None
        }

        with st.spinner("Analyzing, retrieving, reasoning..."):
            start_time = time.perf_counter()
            final_state = legal_agent.invoke(initial_state)
            elapsed = time.perf_counter() - start_time

        st.success(f"Analysis complete in {elapsed:.2f} seconds")

        tab1, tab2, tab3 = st.tabs(["📋 Answer", "📚 Sources", "📝 Draft Clause"])

        with tab1:
            st.markdown("### Final Answer")
            st.markdown(final_state.get("answer", "No answer generated."))

        with tab2:
            st.markdown("### Cited Sources")
            sources = final_state.get("sources", [])
            if sources:
                for s in sources:
                    st.markdown(f"- {s}")
            else:
                st.markdown("No sources extracted.")

        with tab3:
            draft = final_state.get("draft_clause")
            if draft:
                st.markdown("### Draft Clause / Counter-Argument")
                st.markdown(draft)
            else:
                st.info("No draft clause generated for this query.")

        with st.expander("🔍 Retrieval Details"):
            st.markdown(f"**Retrieval attempts performed:** {final_state.get('retrieval_attempts', 0)}")
            st.markdown(f"**Context was sufficient:** {final_state.get('context_sufficient', False)}")
            sub_queries = final_state.get("sub_queries", [])
            if sub_queries:
                st.markdown("**Sub-queries used:**")
                for sq in sub_queries:
                    st.markdown(f"- {sq}")

else:
    st.info("ℹ️ Enter a legal question and click 'Ask the Legal Assistant' to begin.")