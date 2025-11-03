import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
import os

# Constants
CHROMA_DIR = "chroma_db"
UPLOAD_DIR = "uploads"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"

# Initialize folders
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Streamlit UI
st.title("📚 Expert Call Query System (Offline) — Memory Optimized")
st.write("Upload expert call PDFs and ask questions about them. Optimized for low memory use.")

# File uploader
uploaded_files = st.file_uploader("Upload expert call PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.info("Uploading files...")
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully! You can now start querying.")

# Embedding Model and Vector DB (Load once)
@st.cache_resource
def load_embeddings_and_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vectordb

vectordb = load_embeddings_and_db()

# LLM Setup
@st.cache_resource
def load_llm():
    return Ollama(model=LLM_MODEL)

llm = load_llm()

# Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Prompt Template
template = """You are an AI assistant. Answer the question based only on the following context:

{context}

Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Chain Setup
chain = prompt | llm

# Question Input
with st.form("question_form"):
    question = st.text_input("Ask your question:")
    submitted = st.form_submit_button("Submit")

if submitted and question:
    # Retrieve docs
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Get LLM response
    response = chain.invoke({"context": context, "question": question})

    st.markdown("### Answer")
    st.write(response)

    st.markdown("### Source Documents")
    for doc in retrieved_docs:
        st.write(f"• {doc.metadata.get('source', 'Unknown Source')}")
