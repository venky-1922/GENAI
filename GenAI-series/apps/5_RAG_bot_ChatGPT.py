import os
import hashlib
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

load_dotenv()

# -----------------------------
# 1. Load PDF
# -----------------------------
loader = PyPDFLoader("../data/GenAI_Full_Detailed_Pathway.pdf")
docs = loader.load()

# -----------------------------
# 2. Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# -----------------------------
# 3. Generate stable IDs (avoid duplicates)
# -----------------------------
def generate_id(doc):
    return hashlib.md5(doc.page_content.encode()).hexdigest()  # unique ID based on content and prevent duplicates

ids = [generate_id(doc) for doc in chunks]

# -----------------------------
# 4. Embeddings
# -----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# -----------------------------
# 5. Create or Load Vector DB
# -----------------------------
persist_dir = "./vector_db"

if os.path.exists(persist_dir):
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
else:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids,
        persist_directory=persist_dir
    )

# -----------------------------
# 6. Retriever
# -----------------------------
retriever = vector_store.as_retriever(
    search_type="mmr",  # better results
    search_kwargs={"k": 4}
)

# -----------------------------
# 7. LLM
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# -----------------------------
# 8. RAG Chain
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -----------------------------
# 9. Query Loop
# -----------------------------
while True:
    query = input("\nAsk something (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    response = qa_chain.invoke({"query": query})

    print("\n🧠 Answer:\n", response["result"])

    # print("\n📄 Sources:")
    # for doc in response["source_documents"]:
    #     print(doc.metadata)