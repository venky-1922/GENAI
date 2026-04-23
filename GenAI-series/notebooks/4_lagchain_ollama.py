from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma4:e2b",streaming = True)
response = llm.stream("What is Langchain?")
for chunk in response:
    print(chunk.content, end="")