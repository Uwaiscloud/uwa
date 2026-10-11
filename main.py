import os
import tempfile
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit Page Setup
st.set_page_config(
    page_title="Lightning-Fast RAG with Groq & Streamlit",
    page_icon="🤖",
    layout="wide"
)

st.title("📄 AI Document Assistant (RAG)")
st.write("Upload a document (PDF, TXT, or DOCX) and instantly get answers based on its contents using Groq's high-speed inference engine.")

# Sidebar Configuration for API Key
st.sidebar.header("API Configuration")
default_api_key = "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"
groq_api_key = st.sidebar.text_input(
    "Groq API Key", 
    value=default_api_key, 
    type="password"
)

# Model selection (using the fast, free tier Llama 3.1 8B model)
model_name = st.sidebar.selectbox(
    "Groq Model",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    index=0
)

# File Upload Section
uploaded_file = st.file_uploader(
    "Upload your document here", 
    type=["pdf", "txt", "docx"]
)

if uploaded_file is not None:
    # Save uploaded file temporarily to process it
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Load document depending on its file extension
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            st.error("Unsupported file format!")
            st.stop()

        docs = loader.load()

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        # Initialize Free HuggingFace Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(final_documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        st.sidebar.success("Document processed and embedded successfully!")

        # Setup Chat Interface
        if not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar to proceed.")
            st.stop()

        # Initialize Groq LLM
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0.1)

        # Define RAG Prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Create chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # User Question Input
        user_query = st.text_input("Ask a question about your document:")

        if user_query:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_query})
                
                st.subheader("Answer:")
                st.write(response["answer"])

                with st.expander("View Retrieved Document Chunks"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.write(doc.page_content)
                        st.divider()

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
else:
    st.info("👆 Upload a PDF, TXT, or DOCX file to get started.")