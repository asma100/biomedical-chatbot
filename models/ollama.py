import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Initialize the model
model = SentenceTransformer("all-mpnet-base-v2")

# Set up PDF processing and FAISS
pdf_path = r"app/pdfs/abc.pdf"

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    manual_text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            manual_text += page.extract_text()
    return manual_text

# Function to split text into chunks
def split_text(text, chunk_size=500, overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    print('done spliting the text')
    return text_splitter.split_text(text)

# Update load_and_embed_pdf to use these functions
def load_and_embed_pdf(pdf_path):
    if os.path.exists("faiss_index/index.faiss"):
        index = faiss.read_index("faiss_index/index.faiss")
        print ('done embedding')    
    else:
        manual_text = extract_text_from_pdf(pdf_path)
        chunks = split_text(manual_text)
        embeddings = [model.encode(chunk) for chunk in chunks]
        embeddings_array = np.vstack(embeddings)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        faiss.write_index(index, "faiss_index/index.faiss")
        print ('done embedding')
    
    return index

index = load_and_embed_pdf(pdf_path)

# Ensure chunks and embeddings_array are accessible
manual_text = extract_text_from_pdf(pdf_path)
chunks = split_text(manual_text)
embeddings_array = np.vstack([model.encode(chunk) for chunk in chunks])

text_embeddings = [(text, emb.tolist()) for text, emb in zip(chunks, embeddings_array)]

# Set up Ollama model for QA

qa_model = Ollama(model="mistral:7b") # Specify the Ollama model here
# Create Hugging Face embeddings (for retrieval)
huggingface_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
faiss_store = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=huggingface_embeddings)

retriever = faiss_store.as_retriever()
print('done withe retriever')

# Set up the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=qa_model,
    retriever=retriever,
    chain_type="map_reduce"
)

# Define a function to ask questions
def ask_question(query):
    print('starting the ask question function')
    print(query)
    response = qa_chain.invoke({"query": query})
    return response["result"]












