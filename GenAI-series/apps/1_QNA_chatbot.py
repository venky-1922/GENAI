from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

st.title("🤖 Trained Chat Bot")
st.markdown("This is a simple chatbot built using Google Gemini 2.5 Flash Lite model. It can answer your questions and have a conversation with you.")

if("messages" not in st.session_state) : 
    st.session_state.messages = []

for message in st.session_state.messages :
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content) 

history = []
query = st.chat_input("Ask me what you want to know...")
if query : 
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role":"user", "content": query})
    res = llm.invoke(st.session_state.messages)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({"role":"ai", "content":res.content})

