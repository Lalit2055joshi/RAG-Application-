from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import asyncio
import streamlit as st

load_dotenv()

# Access the API key
# api_key = os.getenv("OPENAI_API_KEY")
# for streamlit production
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# Gemini api key
# gemini_api_key = os.getenv("GEMINI_API_KEY")

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = api_key

# Set gemini api
os.environ["GOOGLE_API_KEY"] = gemini_api_key

def setup_rag_pipeline(pdf_path):
    # Ensure event loop exists in this thread
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    # Step 1: Load PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2: Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # Step 3: Create embeddings and vector store
    # embeddings = OpenAIEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Step 4: Set up the language model
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Set up the language model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,

    )

    # Step 5: Define the prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Provide a concise and accurate answer.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Step 6: Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

def ask_question(qa_chain, question):
    # Run the question through the QA chain
    result = qa_chain.invoke({"query": question})
    return result["result"], result["source_documents"]

