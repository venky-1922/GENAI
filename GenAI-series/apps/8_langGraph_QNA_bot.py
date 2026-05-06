from langgraph.checkpoint.memory import InMemorySaver
# from IPython.display import Image
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from langchain_groq import ChatGroq
from typing import List, Annotated

from dotenv import load_dotenv
load_dotenv()


class ChatState(BaseModel):
    messages : Annotated[List,add_messages]

llm = ChatGroq(model="openai/gpt-oss-20b")
memory = InMemorySaver()

def chatBotNode(state : ChatState) -> ChatState :
    res = llm.invoke(state.messages)
    state.messages = [res]
    return state 

graph = StateGraph(ChatState)
graph.add_node("chatBotNode", chatBotNode)
graph.add_edge(START, "chatBotNode")
graph.add_edge("chatBotNode", END)

final_graph = graph.compile(checkpointer=memory)
# print(Image(final_graph.get_graph().draw_mermaid_png()))
while True :
    query = input("User: ")
    if query.lower() in ["exit","quit"]:
        break
    res = final_graph.invoke({"messages":[{"role":"user","content":query}]},{"configurable":{"thread_id":"1"}})
    print(res["messages"][-1].content)

