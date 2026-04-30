import os,hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()


loader = PyPDFLoader("../data/Venkatesh_Vanjarapu_Resume_Final.pdf")
data = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(data)

def generate_id(doc):
    return hashlib.md5(doc.page_content.encode()).hexdigest()

ids = [generate_id(doc) for doc in chunks]

# for data in chunks:
#     print(data.page_content,"\n\n")

embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")

persist_dir = "./vector_db1"
if os.path.exists("./vector_db1"):
    vector_store = Chroma(
            # documents = chunks,
            embedding_function = embeddings,
            persist_directory=persist_dir
        )
else:
    vector_store = Chroma.from_documents(
            documents = chunks,
            embedding = embeddings,
            ids = ids,
            persist_directory=persist_dir
        )


llm = ChatOpenAI(model="gpt-5")

@tool 
def retrieve_chunks(query:str):
    """
    Retrieve relevant information from chunks from the vector store based on the query.
    """
    print("tool called with query:", query)
    result = vector_store.similarity_search(query = query,k=4)
    context =""
    for data in result:
        print("Retrieved chunk:\n", data.page_content, "\n")
        context += data.page_content + "\n"
    return context

# system_prompt = "You are a helpful assistant that answers questions based on the given context. Always use the provided context to answer the question. If you don't know the answer, say 'I don't know the answer'."
system_prompt = """
You are a helpful assistant.
You MUST ALWAYS call the tool tool before answering.
Do NOT answer from your own knowledge.
If answer is found, return it clearly.
If not found, say: "I don't know the answer."
"""

agent = create_agent(
    model = llm,
    tools = [retrieve_chunks],
    system_prompt = system_prompt
)
query = "email and mobile number of candidate"
response = agent.invoke({"messages":[{"role":"user","content":query}]})
print(response["messages"][-1].content)