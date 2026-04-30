from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
import streamlit as st
# from langchain_ollama import ChatOllama

from dotenv import load_dotenv
load_dotenv()

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "agent" not in st.session_state:
    st.session_state.agent = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages_history" not in st.session_state:
    st.session_state.messages_history =[]


def process_documents(path:str):
    loader = PyPDFDirectoryLoader(path)
    data = loader.load()
    print(len(data))
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
    splittedText = splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore.from_documents(
        documents=splittedText,
        embedding=embeddings
    )
    llm = ChatGroq(model = "openai/gpt-oss-20b")
    # llm = ChatOpenAI(model="gpt-5")
    # llm = ChatOllama(model="gemma4:e2b")
    memory = InMemorySaver()
    @tool
    def retrieval_data(query : str):
        """
        This tool contains ALL information from the uploaded PDFs.
        You MUST use this tool to answer ANY user question.
        Do NOT answer without calling this tool.
        """
        print("tool called :",query)
        data = vector_store.similarity_search(query=query,k=4)
        context="" 
        for text in data :
            context+=text.page_content
        print("context:",context)
        return context

    # systemPrompt = """
    # You are usefull ai assistant always call the retrieval_data tool to extract the context for the user query and give response from the context that is extracted from the tool only don't give response based on your knowledge if context is not related to the question say 'I don't know the answer' but don't give answer based on your knowledge
    # """
    systemPrompt = """
            STRICT RULES:
            1. You MUST call the retrieval_data tool before answering.
            2. You are allowed to answer with tool output.
            3. If tool returns empty or not related context related to question → say "I don't know the answer".
            4. Do NOT use your own knowledge.
            """
    agent = create_agent(
        model = llm,
        tools = [retrieval_data],
        system_prompt=systemPrompt,
        checkpointer=memory
    )
    st.session_state.agent = agent
    st.session_state.document_uploaded = True

####pdf upload UI
if not st.session_state.document_uploaded:
    print("document uploaded : ",st.session_state.document_uploaded)
    uploaded = st.file_uploader(label="Select PDF files", type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            path = "./doc_files/"
            for file in uploaded:
                with open(path+file.name,"wb") as f:
                    f.write(file.getvalue())
            process_documents(path)
            st.rerun()


#### Chat UI
if st.session_state.document_uploaded and st.session_state.agent:
    for data in st.session_state.messages_history:
        role = data["role"]
        content= data["content"]
        st.chat_message(role).markdown(content)
    query = st.chat_input("Ask me anything from the pdf files you uploaded....")
    if(query):
        st.chat_message("user").markdown(query)
        st.session_state.messages_history.append({"role":"user","content":query})      
        response = st.session_state.agent.invoke({"messages":[{"role":"user","content":query}]},{"configurable":{"thread_id":"1"}})
        result = response["messages"][-1].content
        st.chat_message("ai").markdown(result)
        st.session_state.messages_history.append({"role":"ai","content":result})