from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

response = llm.invoke("Привет! Объясни, что такое LangChain.")
print(response.content)
