from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_groq import ChatGroq
llm = ChatGroq(model="openai/gpt-oss-20b")
response = llm.invoke("What is Langchain?")
print(response.content)