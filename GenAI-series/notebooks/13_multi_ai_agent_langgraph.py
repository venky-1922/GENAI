from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal
from langchain_groq import ChatGroq 
from langchain_community.utilities import GoogleSerperAPIWrapper,OpenWeatherMapAPIWrapper
from langchain.agents import create_agent 
# from langchain.tools import tool
# from langchain_openai import ChatOpenAI

class FlowState(BaseModel):
    question : str = Field(default="")
    category : Literal['google_search', 'weather', 'coding'] = Field(default="google_search")
    answer : str = Field(default="")    

class CategoryState(BaseModel):
    category : Literal['google_search', 'weather', 'coding'] = Field(default="google_search")

llm = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
tools = [search.run]
weather = OpenWeatherMapAPIWrapper()
# llm = ChatOpenAI(model="gpt-5")

google_search_agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt ="you are a helpful assistant and you should use the google search tool to provide the answer don't provide answer based on your knowledge"
)

# @tool
# def weather_tool(city:str)-> str :
#     """
#     This tool provides weather information based on the user query.
#     """
#     res = f"the weather in the city is 30 degree celsius"
#     return res

weather_agent = create_agent(
    model = llm, 
    tools = [weather.run],
    system_prompt = "you are a helpful assistant and you should use the weather tool to provide the answer in understandable language don't provide answer based on your knowledge"
)

def QuestionCategory(state : FlowState) -> FlowState :
    result = llm.with_structured_output(CategoryState)
    res = result.invoke(f" I want to know the category of the question :{state.question} and if you are not sure return google_search ")
    # print("category:",res.category)
    state.category = res.category
    return state

def route(state:FlowState)-> Literal['google_search', 'weather', 'coding'] :
    return state.category

def google_search_node(state:FlowState)-> FlowState :
    res = google_search_agent.invoke({"messages":{"role":"user","content":state.question}})
    state.answer = res["messages"][-1].content
    return state

def weather_node(state:FlowState)-> FlowState :
    res = weather_agent.invoke({"messages":{"role":"user","content":state.question}})
    state.answer = res["messages"][-1].content
    return state

def coding_node(state:FlowState)-> FlowState :
    res = llm.invoke(f"you are a coding expert: {state.question}")
    state.answer = res.content
    return state

graph = StateGraph(FlowState)
graph.add_node("QuestionCategory", QuestionCategory)
graph.add_node("google_search", google_search_node)
graph.add_node("weather", weather_node)
graph.add_node("coding", coding_node)

graph.add_edge(START, "QuestionCategory")
graph.add_conditional_edges("QuestionCategory", route)
graph.add_edge("google_search", END)
graph.add_edge("weather", END)
graph.add_edge("coding", END)

final_graph = graph.compile()
res = final_graph.invoke({"question":"What is temp in india?"})
print(res)