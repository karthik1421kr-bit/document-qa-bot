import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# ------------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Q&A Bot")
st.caption("Upload a PDF, ask questions, get answers with citations.")


# ------------------------------------------------------------------
# Cache the expensive setup — only reprocess when PDF changes
# ------------------------------------------------------------------
@st.cache_resource
def build_rag_chain(pdf_path):
    """Load PDF, chunk it, embed, and return an LCEL chain + retriever."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided document.

Instructions:
- Answer using ONLY the context below
- If the answer isn't in the context, say "I don't have enough information in the document to answer that."
- Provide thorough, well-structured answers with specific details from the document
- When relevant, quote or reference specific phrases from the document
- Use bullet points for lists of items (skills, projects, responsibilities)
Context:
{context}

Question: {question}

Answer:
""")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    def format_docs(docs):
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page", "?")
            formatted.append(f"[Page {page + 1}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain, retriever, len(chunks), len(documents)


# ------------------------------------------------------------------
# Sidebar — upload PDF
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("Processing document..."):
            rag_chain, retriever, num_chunks, num_pages = build_rag_chain(pdf_path)

        st.success(f"✓ Loaded {num_pages} pages, {num_chunks} chunks")
        st.session_state["rag_ready"] = True
        st.session_state["rag_chain"] = rag_chain
        st.session_state["retriever"] = retriever
        st.session_state["filename"] = uploaded_file.name

    # Clear chat button
    if st.session_state.get("messages"):
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown("### About")
    st.markdown(
        "**Stack:** Gemini 2.5 Flash-Lite · LangChain LCEL · ChromaDB · Streamlit"
    )
    st.markdown(
        "**Repo:** [document-qa-bot](https://github.com/karthik1421kr-bit/document-qa-bot)"
    )

# ------------------------------------------------------------------
# Main chat area
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📚 Sources ({len(msg['citations'])})"):
                for i, cite in enumerate(msg["citations"], 1):
                    st.markdown(f"**[{i}] Page {cite['page']}**")
                    st.caption(cite["preview"])

# Chat input
if question := st.chat_input("Ask a question about the document..."):
    if not st.session_state.get("rag_ready"):
        st.warning("Please upload a PDF first.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state["rag_chain"].invoke(question)
            source_docs = st.session_state["retriever"].invoke(question)

        st.markdown(answer)

        # Format citations for display + storage
        citations = []
        for doc in source_docs:
            citations.append({
                "page": doc.metadata.get("page", 0) + 1,
                "preview": doc.page_content[:200].replace("\n", " ") + "..."
            })

        with st.expander(f"📚 Sources ({len(citations)})"):
            for i, cite in enumerate(citations, 1):
                st.markdown(f"**[{i}] Page {cite['page']}**")
                st.caption(cite["preview"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations
        })


