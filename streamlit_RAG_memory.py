"""
Streamlit RAG Application with LangGraph - PDF Version
A conversational RAG system with chat interface for PDF documents
"""

import os
from typing import TypedDict, List
import streamlit as st

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
import PyPDF2
import tempfile

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Constants
VECTOR_STORE_PATH = "faiss_index_pdf"
DOCUMENTS_FOLDER = "documents_pdf"


# ========================================
# STEP 1: DEFINE STATE
# ========================================
class RAGState(TypedDict):
    """State for the RAG workflow"""
    question: str
    context: str
    answer: str
    retrieved_docs: List[Document]
    chat_history: List


# ========================================
# STEP 2: DOCUMENT MANAGEMENT
# ========================================
def get_all_documents():
    """Get all .pdf files from documents folder"""
    if not os.path.exists(DOCUMENTS_FOLDER):
        os.makedirs(DOCUMENTS_FOLDER)
        return []

    files = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.endswith('.pdf')]
    return sorted(files)


def read_document(filename: str) -> List[Document]:
    """Read content of a PDF document file using PyPDF2"""
    filepath = os.path.join(DOCUMENTS_FOLDER, filename)

    documents = []
    with open(filepath, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            # Create a Document object for each page
            doc = Document(
                page_content=text,
                metadata={"source": filename, "page": page_num + 1}
            )
            documents.append(doc)

    return documents


def save_document(filename: str, content: bytes):
    """Save PDF content to a document file"""
    if not os.path.exists(DOCUMENTS_FOLDER):
        os.makedirs(DOCUMENTS_FOLDER)

    # Ensure .pdf extension
    if not filename.endswith('.pdf'):
        filename = filename + '.pdf'

    filepath = os.path.join(DOCUMENTS_FOLDER, filename)
    with open(filepath, 'wb') as f:
        f.write(content)


# ========================================
# STEP 3: INITIALIZE COMPONENTS
# ========================================
@st.cache_resource
def load_embeddings():
    """Load HuggingFace embeddings"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():
    """Load LLM"""
    return ChatOpenAI(model='gpt-4o-mini', temperature=0.2)


def get_vectorstore():
    """Load or create vector store - uses session state for caching"""
    # Check if vectorstore is already loaded in session state
    if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
        return st.session_state.vectorstore

    embeddings = load_embeddings()

    if os.path.exists(VECTOR_STORE_PATH):
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state.vectorstore = vectorstore
    else:
        st.warning("Vector store not found. Creating new one from documents folder...")

        # Get all document files
        doc_files = get_all_documents()

        if len(doc_files) == 0:
            st.error("❌ No documents found! Please upload .pdf files using the sidebar.")
            return None

        st.info(f"Found {len(doc_files)} document(s)")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )

        # Load and process all PDF documents
        all_docs = []
        for filename in doc_files:
            docs = read_document(filename)
            # Add filename to metadata
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)

        split_docs = text_splitter.split_documents(all_docs)

        st.info(f"Split into {len(split_docs)} chunks")

        # Create vector store
        vectorstore = FAISS.from_documents(split_docs, embeddings)

        # Save vector store
        vectorstore.save_local(VECTOR_STORE_PATH)
        st.success(f"✅ Vector store created and saved to {VECTOR_STORE_PATH}")

        # Cache in session state
        st.session_state.vectorstore = vectorstore

    return vectorstore


# ========================================
# STEP 3: DEFINE LANGGRAPH NODES
# ========================================
def retrieve_documents(state: RAGState) -> RAGState:
    """Retrieve relevant documents from vector store"""
    question = state["question"]

    vectorstore = get_vectorstore()

    # Retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=3)

    # Combine document contents
    context = "\n\n".join([doc.page_content for doc in docs])

    return {
        "context": context,
        "retrieved_docs": docs
    }


def generate_answer(state: RAGState) -> RAGState:
    """Generate answer using LLM with context and conversation history"""
    question = state["question"]
    context = state["context"]
    chat_history = state.get("chat_history", [])
    llm = load_llm()

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant who will strictly answer only based on the given context. 
        Use the following context to answer the user's question.
        If you cannot answer the question based on the context, say so.
        You can refer to the conversation history to provide more relevant answers.

        Context:
        {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # Generate response
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })

    return {
        "answer": response.content
    }


# ========================================
# STEP 4: CREATE LANGGRAPH
# ========================================
@st.cache_resource
def create_rag_graph():
    """Create the LangGraph workflow"""
    # Create the graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)

    # Add edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile the graph
    app = workflow.compile()
    return app


# ========================================
# STEP 5: STREAMLIT UI
# ========================================
def main():
    st.set_page_config(
        page_title="RAG Chat with LangGraph - PDF",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 RAG Chat Application with LangGraph - PDF Version")
    st.markdown("Ask questions about the PDF documents in your knowledge base!")

    # Sidebar
    with st.sidebar:
        st.header("📚 System Information")
        st.markdown(f"**Documents Folder:** `{DOCUMENTS_FOLDER}`")
        st.markdown(f"**Embeddings:** HuggingFace (all-MiniLM-L6-v2)")
        st.markdown(f"**LLM:** GPT-4o-mini")

        st.markdown("---")
        st.markdown("### 📤 Upload Documents")

        uploaded_files = st.file_uploader(
            "Choose .pdf files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload .pdf files to add to your knowledge base"
        )

        if uploaded_files:
            # Use a button to trigger the ingestion
            if st.button("📥 Ingest Documents"):
                with st.spinner("Ingesting documents..."):
                    for uploaded_file in uploaded_files:
                        filename = uploaded_file.name
                        content = uploaded_file.read()

                        # Save the document
                        save_document(filename, content)
                        st.success(f"✅ Saved: {filename}")

                    # Reingest after upload
                    import shutil
                    if os.path.exists(VECTOR_STORE_PATH):
                        shutil.rmtree(VECTOR_STORE_PATH)

                    # Clear caches
                    st.cache_resource.clear()
                    if "vectorstore" in st.session_state:
                        del st.session_state.vectorstore

                    st.success(f"✅ Successfully ingested {len(uploaded_files)} document(s) into vector store!")
                    st.info("Please refresh to start asking questions.")

                    # Small delay before rerun
                    import time
                    time.sleep(1)
                    st.rerun()

        st.markdown("---")
        st.markdown("### 💡 How it works")
        st.markdown("""
        1. Upload .pdf files with your documents
        2. Documents are chunked and embedded
        3. Ask questions in natural language
        4. AI retrieves relevant chunks and answers
        """)

    # Initialize vector store
    vectorstore = get_vectorstore()

    if vectorstore is None:
        st.error("❌ Failed to initialize vector store. Please check your documents file.")
        return

    # Initialize the graph
    graph = create_rag_graph()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show retrieved documents if available
            if message["role"] == "assistant" and "retrieved_docs" in message:
                with st.expander("📄 View Retrieved Context"):
                    for i, doc in enumerate(message["retrieved_docs"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.markdown("---")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Create initial state with hist
                initial_state = {
                    "question": prompt,
                    "context": "",
                    "answer": "",
                    "retrieved_docs": [],
                    "chat_history": st.session_state.chat_history
                }

                # Run the graph
                result = graph.invoke(initial_state)

                # Display answer
                st.markdown(result["answer"])

                # Show retrieved documents
                with st.expander("📄 View Retrieved Context"):
                    for i, doc in enumerate(result["retrieved_docs"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.markdown("---")

        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=result["answer"]))

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "retrieved_docs": result["retrieved_docs"]
        })


if __name__ == "__main__":
    main()