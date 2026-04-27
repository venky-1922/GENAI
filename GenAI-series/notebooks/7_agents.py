from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

@tool
def add_numbers(a:int, b:int):
    """
    Add two numbers and return the result.
    Args:
        a : First number
        b : Second number
    """
    return a + b

@tool
def multiply_numbers(a:int, b:int):
    """
    multiply two numbers and return the result.
    Args:
        a : First number
        b : Second number
    """
    return a * b

llm = ChatGroq(model = "openai/gpt-oss-20b")
agent = create_agent(
    model= llm,
    tools=[add_numbers,multiply_numbers],
    system_prompt="You are a math teacher.ALWAYS use tools for calculations.Never solve math directly."    #please mention use tools so that agent should call tools or else sometimes it will directly solve math without calling tools
)
response = agent.invoke({"messages":[{"role":"user","content":"What is the output of 5+4*2"}]})
# print(response)
for res in response["messages"]:
    print(res +"\n")