from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader("../data/GenAI_Full_Detailed_Pathway.pdf")
data = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(data)
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings
)
query = "What is CPP?"

def get_context(query):
    result_data = vector_store.similarity_search(query = query)
    context_data = ""
    for data in result_data:
        context_data += data.page_content + "\n"
    return {
        "query": query,
        "context": context_data
    }

llm = ChatOpenAI(model="gpt-4")
prompts = PromptTemplate.from_template(
    "You are a helpful assistant that answers questions based on the given context. Always use the provided context to answer the question. If you don't know the answer, say 'I don't know the answer'. \n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
)
rag_chain = get_context | prompts | llm 
response  = rag_chain.invoke(query)
print(response.content)

