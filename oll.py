from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """
Answer the question below

Here is the conversation history: {context}

Question: {question}

Answer: {{answer}}
"""

# Consider using a chatbot-focused model
model = OllamaLLM(model="mistral:7b")  # or "alpaca-7b"

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model | StrOutputParser()  # Add StrOutputParser to get string output

def handle_conversation():
    context = ""
    print("welcome")
    while True:
        user_input = input("you: ")
        if user_input.lower() == "exit":
            break
        result = chain.invoke({"context": context, "question": user_input})
        print("bot: ", result)
        context += f"\nUser: {user_input}\nAI: {result}"

handle_conversation()