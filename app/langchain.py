from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

 
pdf_paths = [r"D:\vs\device-DR-chatbot\app\pdfs\Gyrozen 416 Centrifuge - Service manual.pdf",
             r"D:\vs\device-DR-chatbot\app\pdfs\centrifuge2.pdf"]

all_documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    documents = loader.load()
    all_documents.extend(documents)


# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Load the embedding model and embed the text chunks
huggingface_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the FAISS index directly using the embeddings and text chunks
faiss_store = FAISS.from_documents(chunks, huggingface_embeddings)

api_key = os.getenv('huggingfacehub_api_token')
# Set up the QA chain with a retriever and an LLM

qa_model = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    huggingfacehub_api_token=api_key,
    model_kwargs={"max_length": 1024, "temperature": 0.7, "top_p": 0.9}
)

retriever = faiss_store.as_retriever(search_kwargs={"max_chunks": 5})


qa_chain = RetrievalQA.from_chain_type(
    llm=qa_model,
    retriever=retriever,
    chain_type="stuff"
)


def ask_question(query):
    response = qa_chain({"query": query})
    return response["result"]

