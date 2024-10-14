import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
import time
import requests


# Ensure the output directory exists
index_dir = "faiss_index"
os.makedirs(index_dir, exist_ok=True)

# Extract text from PDF function
def extract_text_from_pdf(pdf_path):
    """Extract Text from PDFs"""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

pdf_path = r"D:\vs\device-DR-chatbot\pdfs\Maintenance Manual for Laboratory Equipment - WHO.pdf"
manual_text = extract_text_from_pdf(pdf_path)

# Split text function
def split_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

chunks = split_text(manual_text)

# Use a larger model for embeddings
model = SentenceTransformer('all-mpnet-base-v2')

# Embed text chunks using the local model
embeddings = [model.encode(chunk) for chunk in chunks]
embeddings_array = np.vstack(embeddings)

# Create and save the FAISS index
dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)
faiss.write_index(index, f"{index_dir}/index.faiss")  # Save index for reuse

# Set up the HuggingFaceHub for QA
qa_model = HuggingFaceHub(
    repo_id="google/flan-ul2",  # Replace with your Hugging Face model repo_id
    huggingfacehub_api_token="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"  # Replace with your Hugging Face API token
)

# Create Hugging Face embeddings
huggingface_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Create the FAISS retriever
text_embeddings = [(text, emb.tolist()) for text, emb in zip(chunks, embeddings_array)]
faiss_store = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=huggingface_embeddings)

retriever = faiss_store.as_retriever()

qa_model = HuggingFaceEndpoint(
    repo_id="google/flan-ul2",
    huggingfacehub_api_token="YOUR_API_TOKEN"
)

# Retry logic for handling transient API errors
def ask_question(query, retries=3):
    for attempt in range(retries):
        try:
            response = qa_chain.invoke({"query": query})
            return response["result"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 504:  # Timeout error
                print(f"Timeout error on attempt {attempt + 1}, retrying...")
                time.sleep(2)  # Wait before retrying
            else:
                raise e  # Reraise other errors
    print("Failed to get a response after multiple retries.")
    return None