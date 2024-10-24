from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai




pdf_paths = [r"D:\vs\device-DR-chatbot\app\pdfs\Gyrozen 416 Centrifuge - Service manual.pdf",
             r"D:\vs\device-DR-chatbot\app\pdfs\centrifuge2.pdf"]

# Load all PDF documents
all_documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    documents = loader.load()
    all_documents.extend(documents)

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(all_documents)

# Load the embedding model and embed the text chunks
huggingface_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the FAISS index directly using the embeddings and text chunks
faiss_store = FAISS.from_documents(chunks, huggingface_embeddings)

# Set up a retriever
retriever = faiss_store.as_retriever(search_kwargs={"max_chunks": 5})

# Conversation history to store previous exchanges
conversation_history = []

# Function to retrieve relevant text chunks
def get_relevant_docs(query):
    docs = retriever.get_relevant_documents(query)
    relevant_chunks = [doc.page_content for doc in docs]
    return relevant_chunks

# Function to create RAG prompt with conversation memory
def make_rag_prompt(query, relevant_passage, history):
    relevant_passage = ' '.join(relevant_passage)
    
    # Include conversation history in the prompt
    history_text = " ".join([f"User: {item['question']}\nBot: {item['answer']}" for item in history])
    
    prompt = (
        f"You are a helpful and informative chatbot that answers questions using text from the reference passage below. "
        f"Respond in a complete sentence and make sure that your response is easy to understand for everyone. "
        f"Maintain a friendly and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"QUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
        f"ANSWER:"
    )
    return prompt

# Function to generate a response using Gemini
def generate_response(user_prompt):
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(user_prompt)
    return answer.text

# Main function to ask a question and store the answer in history
def ask_question(query):
    relevant_text = get_relevant_docs(query)
    prompt = make_rag_prompt(query, relevant_passage=relevant_text, history=conversation_history)
    answer = generate_response(prompt)

    # Store the current query and answer in the conversation history
    conversation_history.append({'question': query, 'answer': answer})

    # Limit the history to a certain number of exchanges (e.g., last 5 interactions)
    if len(conversation_history) > 5:
        conversation_history.pop(0)  # Remove the oldest conversation

    return answer
