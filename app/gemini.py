from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import LLMResult
from typing import List
import google.generativeai as genai


# Custom Gemini Wrapper
class GeminiLLM:
    def __init__(self, model_name='gemini-pro'):
        self.model = genai.GenerativeModel(model_name)

    # This method ensures compatibility with LangChain
    def generate(self, prompts: List[str]) -> LLMResult:
        responses = [self.model.generate(prompt) for prompt in prompts]
        return LLMResult(generations=[[{"text": response.result}] for response in responses])


# Load the PDFs
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

# Use the custom Gemini model
qa_model = GeminiLLM()

# Create retriever from FAISS store
retriever = faiss_store.as_retriever(search_kwargs={"max_chunks": 5})

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=qa_model,
    retriever=retriever,
    chain_type="stuff"
)

# Function to ask questions
def ask_question(query):
    response = qa_chain({"query": query})
    return response["result"]
