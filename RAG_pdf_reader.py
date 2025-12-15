'''
Company policies - Docs
Doc --> LLM >> Ask for answer : 2 Issues - 1. Token Limit Error , 2. Cost
--> Grounding : Results should be grounded
'''

import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

load_dotenv()  # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VECTOR_STORE_PATH = "faiss_index"
DOCUMENTS_FILE = "documents.pdf"


'''
INGESTION PART
--> 1. Read the docs
--> 2. Chunking 
--> 3. Create vector embeddings 
--> 4. Load to Vector Store
'''


# ------------------ MODIFY HERE FOR PDF ------------------
def read_documents():
    pdf_path = input("Enter path to PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print("❌ File not found")
        return None

    if not pdf_path.lower().endswith(".pdf"):
        print("❌ Only PDF files supported")
        return None


    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return [text]

# ---------------------------------------------------------



def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = [Document(page_content=doc) for doc in documents]
    return text_splitter.split_documents(docs)


def create_embeddings_and_vector_store(chunks, embeddings):
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store


def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        print('First do the ingestion')
        return None

    loaded_vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return loaded_vectorstore


def ingestion_pipeline():
    content = read_documents()
    if content:
        chunks = create_chunks(content)
        create_embeddings_and_vector_store(chunks, embeddings)
        print("Ingestion completed ✓")


def retrieval(vectorstore, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    # print("\nRetrieved Context:\n")
    for doc in docs:
        print(doc.page_content)
        print('\n-----------------------------------------------\n')


def retrieval_pipeline():
    vectorstore = load_vector_store()
    query = input('What is your query ? ')
    retrieval(vectorstore, query)


# Run ingestion once
ingestion_pipeline()

# Then retrieval
retrieval_pipeline()
