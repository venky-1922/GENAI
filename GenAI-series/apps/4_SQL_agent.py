# pip install langchain langchain-openai langchain-community langgraph streamlit python-dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
from dotenv import load_dotenv

# 🔹 Load env
load_dotenv()

# =========================
# 🔥 DATABASE SETUP
# =========================

db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

db.run("""
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK (status IN ('pending', 'in progress', 'completed')) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# =========================
# 🔥 STREAMLIT UI
# =========================

st.set_page_config(page_title="Task Manager AI", layout="wide")
st.title("🧠 Task Management Agent")

# =========================
# 🔥 LLM + TOOLS
# =========================

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# =========================
# 🔥 SYSTEM PROMPT
# =========================

system_prompt = """
Always use SQL tools for database operations.
Do not answer without executing queries.
Do not refuse valid operations.
"""

# =========================
# 🔥 AGENT (CACHED)
# =========================

@st.cache_resource
def get_agent():
    return create_agent(
        tools=tools,
        model=llm,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver()
    )

agent = get_agent()

# =========================
# 🔥 CHAT MEMORY
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for chat in st.session_state.history:
    st.chat_message(chat["role"]).markdown(chat["content"])

# =========================
# 🔥 USER INPUT
# =========================

query = st.chat_input("Ask me about your tasks...")

if query:
    # Show user message
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})

    # AI response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):

            response = agent.invoke(
                {
                    "messages": [
                        {"role": "user", "content": query}
                    ]
                },
                {
                    "configurable": {"thread_id": "1"}
                }
            )

            final_msg = response["messages"][-1].content

            st.markdown(final_msg)

            st.session_state.history.append({
                "role": "assistant",
                "content": final_msg
            })












# from langchain_groq import ChatGroq
# from langchain.agents import create_agent
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langgraph.checkpoint.memory import InMemorySaver
# import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()

# db = SQLDatabase.from_uri("sqlite:///my_tasks.db")
# db.run(
#     """
#     CREATE TABLE IF NOT EXISTS tasks (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         title TEXT NOT NULL,
#         description TEXT,
#         status TEXT CHECK (status IN ('pending', 'in progress', 'completed')) DEFAULT 'pending',
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#     )
#     """
# )
# # print("database table created successfully")

# st.subheader("Task Management Agent - Manage your tasks")

# llm = ChatGroq(model="openai/gpt-oss-20b")
# toolkit = SQLDatabaseToolkit(db = db , llm = llm)
# tools = toolkit.get_tools()
# system_prompt = """
# You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

# TASK RULES:
# 1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
# 2. After CREATE/UPDATE/DELETE, confirm with SELECT query
# 3. If the user requests a list of tasks, present the output in a structured table format

# CRUD OPERATIONS:
#     CREATE: INSERT INTO tasks(title, description, status)
#     READ: SELECT * FROM tasks WHERE ... LIMIT 10
#     UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
#     DELETE: DELETE FROM tasks WHERE id=? OR title=?

# Table schema: id, title, description, status(pending/progress/completed), created_at.
# """
# @st.cache_resource
# def get_agent():
#     agent = create_agent(
#         tools=tools,
#         model=llm,
#         system_prompt=system_prompt,
#         checkpointer=InMemorySaver()
#     )
#     return agent
# agent = get_agent()
# if "history" not in st.session_state:
#     st.session_state.history = []

# for chat in st.session_state.history:
#     role = chat["role"]
#     content = chat["content"]
#     st.chat_message(role).markdown(content)

# query = st.chat_input("Ask me anything about your tasks: ")
# if query:
#     st.chat_message("user").markdown(query)
#     st.session_state.history.append({"role": "user", "content": query})
#     with st.chat_message("ai"):
#         with st.spinner("Processing..."):
#             response = agent.invoke(
#                 {
#                     "messages": { "role": "user", "content":query}
#                 },
#                 {
#                     "configurable": {"thread_id":"1"}
#                 }
#             )
#             st.markdown(response["messages"][-1].content)
#             st.session_state.history.append({"role": "ai", "content": response["messages"][-1].content})