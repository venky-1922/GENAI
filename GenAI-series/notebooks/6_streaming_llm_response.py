from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-20b",streaming=True)
question = "What is Langchain?"
response = llm.stream(question)
for chunk in response:
    print(chunk.content, end="")