# Document QA Bot - Progress Notes

## Status
Week 3 Day 1 - Basic RAG pipeline working

## Stack (pinned)
- Python 3.13
- gemini-2.5-flash-lite (LLM)
- gemini-embedding-001 (embeddings)
- langchain + langchain-text-splitters (modular imports)
- ChromaDB (local, ignored in git)
- LCEL pattern (not RetrievalQA)

## Completed
- [x] hello_gemini.py - basic API call
- [x] Prompt comparison tool (llm-experiments repo)
- [x] RAG pipeline with text file

## TODO
- [ ] Replace TextLoader with PyPDFLoader
- [ ] Add source citations
- [ ] Streamlit frontend
- [ ] Deploy + GitHub push

## Gotchas we hit
- text-embedding-004 deprecated → use gemini-embedding-001
- google-generativeai deprecated → use google-genai
- langchain.text_splitter moved → use langchain-text-splitters