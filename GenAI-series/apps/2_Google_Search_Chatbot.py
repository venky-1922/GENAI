from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0)
search = GoogleSerperAPIWrapper()
memory = InMemorySaver()

agent = create_agent(
    model = llm,
    tools = [search.run],
    system_prompt = "You are a helpful assistant always answer questions using Google Search.",
    checkpointer=memory
)
while True:
    query = input("User : ")
    if query.lower() == "exit" :
        print("GoodBye")
        break
    response = agent.invoke(
        {"messages":{"role": "user", "content":query}},
        {"configurable":{"thread_id":"venky"}}
        )
    print(response["messages"][-1].content)
    # for chunks in response["messages"]:
    #     print(chunks,end="\n")
