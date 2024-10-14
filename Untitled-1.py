
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint


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

def split_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

chunks = split_text(manual_text)

# Load a local transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed text chunks using the local model
embeddings = [model.encode(chunk) for chunk in chunks]
embeddings_array = np.vstack(embeddings)

# Create FAISS index from the numpy array
dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# Set up the Hugging Face model for question answering using HuggingFaceHub
qa_model = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # A model suitable for text2text-generation
    huggingfacehub_api_token="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"
)
# Create Hugging Face embeddings
huggingface_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the FAISS retriever
text_embeddings = [(text, emb.tolist()) for text, emb in zip(chunks, embeddings_array)]
faiss_store = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=huggingface_embeddings)

retriever = faiss_store.as_retriever()

# Set up the RetrievalQA chain (combining retrieval and generation)
qa_chain = RetrievalQA.from_chain_type(
    llm=qa_model,
    retriever=retriever,
    chain_type="stuff"  # Specify the chain type as "stuff" or "map_reduce" etc.
)

def ask_question(query):
    response = qa_chain({"query": query})
    return response["result"]

question = "How does a pH Meter work?"
answer = ask_question(question)
print("Answer:", answer)

























<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css">
    <link rel="stylesheet" href="static/style.css">
    <title>Chatbot - Brave Coder</title>
</head>
<body>
    <script>
        const chatForm = document.getElementById('chat-form');
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
    
        chatForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const query = userInput.value.trim();
            if (!query) return;
            
            // Display user message
            const userMessage = document.createElement('div');
            userMessage.classList.add('item', 'right');
            userMessage.innerHTML = `<div class="msg"><p>${query}</p></div>`;
            chatBox.appendChild(userMessage);
            
            // Reset input field
            userInput.value = '';
    
            // Fetch response from Flask backend
            try {
                const response = await fetch('/get_response', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                const answer = data.response;
    
                // Display chatbot response
                const botMessage = document.createElement('div');
                botMessage.classList.add('item');
                botMessage.innerHTML = `
                    <div class="icon"><i class="fa fa-user"></i></div>
                    <div class="msg"><p>${answer}</p></div>
                `;
                chatBox.appendChild(botMessage);
                
                // Scroll chat to the bottom
                chatBox.scrollTop = chatBox.scrollHeight;
    
            } catch (error) {
                console.error('Error fetching response:', error);
            }
        });
    </script>
    
    
</body>
</html>