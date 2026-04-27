from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import GoogleSerperAPIWrapper
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
search = GoogleSerperAPIWrapper()
tools = [search.run]

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()
    st.session_state.history = []

st.subheader("Quick chat bot - Answer you question faster that Chat gpt")

agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt = "You are a helpful assistant can also search in google to answer questions.",
    checkpointer=st.session_state.memory
)

for chat in st.session_state.history:
    role = chat["role"]
    content = chat["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask me anything")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})
    response = agent.stream(
        {
            "messages":{"role": "user", "content":query}
        },
        {
            "configurable":{"thread_id":"1"}
        },
        stream_mode="messages"
    )
    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()
        msg = ""
        for chunk in response:
            msg += chunk[0].content
            space.write(msg)
        st.session_state.history.append({"role": "ai", "content": msg})