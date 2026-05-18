# Legal AI Assistant — Agentic RAG

A production-ready Legal AI system that retrieves, reasons over, and generates answers from case law and contracts. Built with LangGraph, FAISS, and Streamlit.

## 🚀 Quick Start (Fresh Device)

1. **Clone or download** this repository.
2. **Run the setup script**:
   - Windows: double‑click `setup.bat`
   - macOS/Linux: `bash setup.sh`
3. **Add your OpenAI API key** to the `.env` file.
4. **Build the vector store**: `python run.py build`
5. **Launch the app**: `python run.py ui`

Then open `http://localhost:8501` in your browser.

## 📋 Other Commands

| Command | Action |
|---------|--------|
| `python run.py build` | Build/update FAISS indexes |
| `python run.py query` | Interactive CLI query |
| `python run.py evaluate` | Run test suite & metrics |
| `python run.py ui` | Launch Streamlit UI |