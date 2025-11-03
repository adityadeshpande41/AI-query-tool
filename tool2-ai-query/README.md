AI-Based Expert Call Query Tool
Overview
This is a local, offline Q&A tool built to answer questions based on expert call transcripts. Users can upload PDFs, ask questions, and get context-grounded answers. Powered by a local Mistral model running on Ollama and using local embeddings for semantic search.

Features
Upload multiple expert call PDFs.

Ask questions in natural language.

Get concise, source-grounded answers.

Fully offline — No internet required.

Easy-to-use Streamlit interface.

Setup Instructions
1. Create a virtual environment (optional but recommended)

Copy
Edit
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

2. Install requirements

Copy
Edit
pip install -r requirements.txt

3. Start Ollama server
Make sure Ollama is running:


Copy
Edit
ollama serve
and that you have Mistral model pulled locally:


Copy
Edit
ollama pull mistral

4. Run the application

Copy
Edit
streamlit run app.py
System Requirements
64 GB RAM

12 GB VRAM GPU

Python 3.10+

Ollama installed with Mistral model

How It Works
PDFs are converted into chunks.

Embeddings are generated using MiniLM-L6-v2.

ChromaDB is used for fast similarity search.

Mistral generates the answer based on retrieved documents.

Example Questions
What are the biggest risks highlighted by the expert?

How is AI being used in Uber Eats according to the expert?

