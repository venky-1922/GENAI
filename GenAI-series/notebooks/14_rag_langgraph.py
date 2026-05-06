from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

# ------------------ LOAD + SPLIT ------------------
loader = PyPDFLoader("../data/Venkatesh_Vanjarapu_Resume_Final.pdf")
data = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splittedText = splitter.split_documents(data)

# ------------------ VECTOR STORE ------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore.from_documents(
    documents=splittedText,
    embedding=embeddings
)

# ------------------ LLM ------------------
llm = ChatGroq(model="openai/gpt-oss-20b")

# ------------------ TOOL ------------------
@tool
def context_retrieval(query: str) -> str:
    """Retrieve relevant info from PDF"""
    docs = vector_store.similarity_search(query=query, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])
    return context  # ✅ return string

# ------------------ AGENT ------------------
agent = create_agent(
    model=llm,
    tools=[context_retrieval],
    system_prompt=(
        "You are a helpful assistant. ALWAYS use the context_retrieval tool. "
        "Answer ONLY from retrieved context. If not found say 'I don't know'."
    )
)

# ------------------ STATE ------------------
class ChatRagState(BaseModel):
    question: str = ""
    context: List[str] = Field(default_factory=list)  # ✅ fixed
    answer: str = ""

# ------------------ NODES ------------------

# Step 1: Retrieve context
def QuestionNode(state: ChatRagState):
    docs = vector_store.similarity_search(state.question, k=4)

    context_list = [doc.page_content for doc in docs]

    return {
        "context": context_list  # ✅ no mutation
    }

# Step 2: Combine context
def ContextNode(state: ChatRagState):
    combined = "\n\n".join(state.context)

    return {
        "context": [combined]  # keep as list for consistency
    }

# Step 3: Generate answer
def AnswerNode(state: ChatRagState):
    prompt = f"""
    Answer the question using ONLY this context:

    {state.context[0]}

    Question: {state.question}
    """

    res = llm.invoke(prompt)

    return {
        "answer": res.content  # ✅ correct access
    }

# ------------------ GRAPH ------------------
graph = StateGraph(ChatRagState)

graph.add_node("QuestionNode", QuestionNode)
graph.add_node("ContextNode", ContextNode)
graph.add_node("AnswerNode", AnswerNode)

graph.add_edge(START, "QuestionNode")
graph.add_edge("QuestionNode", "ContextNode")
graph.add_edge("ContextNode", "AnswerNode")
graph.add_edge("AnswerNode", END)

final_graph = graph.compile()

# ------------------ RUN ------------------
result = final_graph.invoke({
    "question": "What is the name of the person in the PDF and what is his experience?"
})

print("Answer:", result["answer"])











# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import InMemoryVectorStore
# from langchain.agents import create_agent
# from langchain.tools import tool
# from langgraph.graph import StateGraph, START, END
# from pydantic import BaseModel, Field
# from typing import List


# from dotenv import load_dotenv
# load_dotenv()

# loader = PyPDFLoader("../data/Venkatesh_Vanjarapu_Resume_Final.pdf")
# data = loader.load()
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splittedText = splitter.split_documents(data)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# vector_store = InMemoryVectorStore.from_documents(
#     documents=splittedText,
#     embedding=embeddings
# )

# llm = ChatGroq(model="openai/gpt-oss-20b")

# @tool
# def context_retrieval(query:str):
#     """
#     This tool retrieves relevant information from the uploaded PDF based on the user query.
#     You MUST use this tool to answer ANY user question related to the PDF.
#     Do NOT answer without calling this tool.
#     """
#     data = vector_store.similarity_search(query=query,k=4)
#     # context = ""
#     # for text in data:
#     #     context += text.page_content + "\n\n" 

#     # return context
#     # print("data  :" , data)
#     return data


# agent = create_agent(
#     model = llm,
#     tools = [context_retrieval],
#     system_prompt=" you are a helpful assistant that answers questions based on the user query and retrieve context from the PDF using the context_retrieval tool and provide answer based on that and don't give answer based on your knowledge if the context is not related to query or if you don't know the answer return 'I don't know' "
# )

# # res = agent.invoke({"messages":[{"role":"user","content":"What is the name of the person in the PDF and what is his experience?"}]})

# # print(res["messages"][-1].content)


# class ChatRagState(BaseModel):
#     question : str = Field(default="")
#     documents : List =[]
#     context : str = ""
#     answer : str = Field(description="Answer which is extracted from the documents",default="")

# def QuestionNode(state:ChatRagState)-> ChatRagState :
#     chunks = agent.invoke({"messages":[{"role":"user","content":state.question}]})
#     # if chunks["messages"][-1].tool_calls[0].output:
#     # print("chunks   :",chunks["messages"][-1].tool_calls[0].output)
#     # state.documents = ["name of user is venky","experience of user is 5 years in software development"]
#     # print("chunks :",chunks["messages"])
#     state.documents = [chunks["messages"][-1].content, ""]
#     # print("Retrieved Chunks:", chunks)
#     # print(state.context)
#     return state

# def ContextNode(state:ChatRagState)-> ChatRagState :
#     context = ""
#     for text in state.documents:
#         context += text + "\n\n"
#     state.context = context
#     return state

# def AnswerNode(state:ChatRagState)-> ChatRagState :
#     answer = agent.invoke({"messages":[{"role":"user","content":state.question}]})
#     state.answer = answer["messages"][-1].content
#     return state

# graph = StateGraph(ChatRagState)
# graph.add_node("QuestionNode", QuestionNode)
# graph.add_node("ContextNode", ContextNode)
# graph.add_node("AnswerNode", AnswerNode)

# graph.add_edge(START, "QuestionNode")
# graph.add_edge("QuestionNode", "ContextNode")
# graph.add_edge("ContextNode", "AnswerNode")
# graph.add_edge("AnswerNode", END)

# final_graph = graph.compile()

# result = final_graph.invoke({"question":"What is the name of the person in the PDF and what is his experience?"})
# print("Answer:",result["answer"])
