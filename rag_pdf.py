import os
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

PDF_FILE = "test.pdf"
CHROMA_DIR = "./chroma_pdf_db"

# ------------------------------------------------------------------
# STEP 1: Load the PDF
# ------------------------------------------------------------------
print(f"Step 1: Loading {PDF_FILE}...")
loader = PyPDFLoader(PDF_FILE)
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# ------------------------------------------------------------------
# STEP 2: Split into chunks
# ------------------------------------------------------------------
print("\nStep 2: Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,         # Bigger chunks for PDFs — they have more continuous text
    chunk_overlap=150,       # More overlap to preserve context across page breaks
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Show a sample chunk with its metadata
print(f"\nSample chunk metadata: {chunks[0].metadata}")
print(f"Sample chunk preview: {chunks[0].page_content[:150]}...")

# ------------------------------------------------------------------
# STEP 3: Create embeddings and store in vector DB
# ------------------------------------------------------------------
print("\nStep 3: Creating embeddings and storing in ChromaDB...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)
print(f"Stored {len(chunks)} chunks")

# ------------------------------------------------------------------
# STEP 4: Build the RAG chain with LCEL
# ------------------------------------------------------------------
print("\nStep 4: Building RAG chain...")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided document.

Instructions:
- Answer using ONLY information from the context below
- If the answer isn't in the context, say "I don't have enough information in the document to answer that."
- Be concise but complete
- When relevant, mention which section or topic the information is from

Context:
{context}

Question: {question}

Answer:
""")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)


def format_docs(docs):
    """Format retrieved chunks with their page numbers for the LLM."""
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

# ------------------------------------------------------------------
# STEP 5: Interactive Q&A loop
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"RAG Pipeline Ready! Ask questions about {PDF_FILE}.")
print("Type 'quit' to exit.")
print("=" * 60)

while True:
    question = input("\nQuestion: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue

    # Get the answer
    answer = rag_chain.invoke(question)

    # Also retrieve source docs separately so we can show citations
    source_docs = retriever.invoke(question)

    print(f"\nAnswer: {answer}")

    # Show citations with page numbers
    print(f"\n--- Citations ({len(source_docs)} chunks used) ---")
    seen_pages = set()
    for i, doc in enumerate(source_docs, 1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")
        filename = os.path.basename(source)

        preview = doc.page_content[:100].replace("\n", " ")
        print(f"[{i}] {filename}, Page {page + 1}")
        print(f"    \"{preview}...\"")
        seen_pages.add(page + 1)

    if seen_pages:
        pages_str = ", ".join(str(p) for p in sorted(seen_pages))
        print(f"\nUnique pages referenced: {pages_str}")