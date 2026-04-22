import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# ------------------------------------------------------------------
# STEP 1: Load the document
# ------------------------------------------------------------------
print("Step 1: Loading document...")
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# ------------------------------------------------------------------
# STEP 2: Split into chunks
# ------------------------------------------------------------------
print("\nStep 2: Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# ------------------------------------------------------------------
# STEP 3: Create embeddings and store in vector DB
# ------------------------------------------------------------------
print("\nStep 3: Creating embeddings and storing in ChromaDB...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"Stored {len(chunks)} chunks in vector database")

# ------------------------------------------------------------------
# STEP 4: Build the RAG chain using LCEL
# ------------------------------------------------------------------
print("\nStep 4: Building RAG chain with LCEL...")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the context below.
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:
""")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

def format_docs(docs):
    """Convert retrieved documents into a single string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

# The LCEL chain: each step pipes its output to the next
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ------------------------------------------------------------------
# STEP 5: Ask questions
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("RAG Pipeline Ready! Ask questions about the document.")
print("Type 'quit' to exit.")
print("=" * 60)

while True:
    question = input("\nQuestion: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue

    # Also fetch the source docs separately so we can display them
    source_docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)

    print(f"\nAnswer: {answer}")
    print(f"\n--- Sources used ({len(source_docs)} chunks) ---")
    for i, doc in enumerate(source_docs, 1):
        print(f"[{i}] {doc.page_content[:120]}...")