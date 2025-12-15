'''
Company policies - Docs
Doc --> LLM >> Ask for answer : 2 Issues - 1. Token Limit Error , 2. Cost
--> Grounding : Results should be grounded
'''

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()  # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VECTOR_STORE_PATH = "faiss_index"
DOCUMENTS_FILE = "documents"

'''

2 major components : 


1. Ingestion : Developers 
2. Retreival : End Users


'''

'''

INGESTION PART

--> 1. Read the docs
--> 2. Chunking 
--> 3. Create vector embeddings 
--> 4. Load to Vector Store

'''

'''
RETRIEVAL PART 

--> 1. Query - Convert into embeddings 
--> 2. Do a vector Search, get the results
--> 3. Send it to LLM as context
'''

'''
1. INGESTION : 
'''


def read_documents():
    if not os.path.exists(DOCUMENTS_FILE):
        print('No document found')
        return None

    with open(DOCUMENTS_FILE, 'r') as f:
        content = f.read()

    print('Content', content)

    return [content]


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = [Document(page_content=doc) for doc in documents]
    split_docs = text_splitter.split_documents(docs)
    print("Splitted Docs : \n\n")
    for doc in split_docs:
        print(doc)
        print('\n\n ----------------------------------------------- \n\n')
    return split_docs


def create_embeddings_and_vector_store(chunks, embeddings):
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print('Vector Store created and saved')

    return vector_store


def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        print('First do the ingestion')
        return None

    loaded_vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    print('Vector Store got loaded')
    return loaded_vectorstore


def ingestion_pipeline():
    content = read_documents()
    chunks = create_chunks(content)
    vectorestore = create_embeddings_and_vector_store(chunks, embeddings)


def retrieval(vectorstore, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


def llm_generation(query, context):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant who will strictly answer only based on the given context. Use the following context to answer the user's question. Do not make up your answer.
        If you cannot answer the question based on the given context, say so.

        Context : 
        {context}

         """),

        ("human", "{query}")

    ])

    chain = prompt | llm
    response = chain.invoke({"context": context,
                             "query": query})

    return response.content


def retreival_pipeline():
    vectorestore = load_vector_store()
    query = input('What is your query ? ')
    context = retrieval(vectorestore, query)
    answer = llm_generation(query, context)
    print('Answer : ', answer)

#
# ingestion_pipeline()

retreival_pipeline()
