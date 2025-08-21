import streamlit as st
from main import setup_rag_pipeline, ask_question
import tempfile
import os
import hashlib
import asyncio

# Streamlit UI
st.title("PDF Question Answering with RAG")
st.write("Upload a PDF and ask questions about its content.")

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'uploaded_file_id' not in st.session_state:
    st.session_state.uploaded_file_id = None

# PDF upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Process uploaded PDF only if a new file is uploaded
if uploaded_file is not None:
    # Create a unique identifier for the uploaded file (e.g., hash of file content)
    file_content = uploaded_file.read()
    file_id = hashlib.md5(file_content).hexdigest()
    uploaded_file.seek(0)  # Reset file pointer after reading

    # Check if the file is new or different from the previously processed one
    if file_id != st.session_state.uploaded_file_id:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        with st.spinner("Processing PDF..."):
            try:
                # Set up the RAG pipeline
                st.session_state.qa_chain = setup_rag_pipeline(tmp_file_path)
                st.session_state.uploaded_file_id = file_id
                st.success("PDF processed successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.session_state.qa_chain = None
                st.session_state.uploaded_file_id = None

        # Clean up temporary file
        os.unlink(tmp_file_path)
    # else:
    #     st.info("Using previously processed PDF.")
else:
    # Clear session state if no file is uploaded
    if st.session_state.qa_chain is not None:
        st.session_state.qa_chain = None
        st.session_state.uploaded_file_id = None

# Question input and answer display
if st.session_state.qa_chain is not None:
    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner("Generating answer..."):
            try:
                # Get answer and sources
                answer, sources = ask_question(st.session_state.qa_chain, question)
                st.write("**Answer:**")
                st.write(answer)
                st.write("**Sources:**")
                for i, doc in enumerate(sources, 1):
                    st.write(f"**Source {i} (Page {doc.metadata['page']+1}):**")
                    st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
            except Exception as e:
                st.error(f"Error answering question: {str(e)}")
else:
    st.info("Please upload a PDF to ask questions.")

