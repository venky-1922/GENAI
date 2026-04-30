from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

documents = [
    "Artificial Intelligence (AI) is a field of computer science focused on building systems that can perform tasks requiring human intelligence such as learning, reasoning, and problem solving.",
    
    "Machine Learning is a subset of AI that enables systems to learn from data without being explicitly programmed. It includes supervised, unsupervised, and reinforcement learning.",
    
    "Deep Learning is a specialized area of machine learning that uses neural networks with many layers to analyze complex patterns in large datasets such as images, audio, and text.",
    
    "Natural Language Processing (NLP) allows machines to understand and generate human language. It powers applications like chatbots, translation systems, and sentiment analysis.",
    
    "LangChain is a framework for building applications powered by large language models (LLMs). It helps manage prompts, memory, chains, and integrations with external tools.",
    
    "In LangChain, a Document is a data structure that contains text (page_content) and optional metadata, which is useful for storing and retrieving contextual information.",
    
    "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation, allowing models to answer questions using external knowledge sources.",
    
    "Vector databases store embeddings (numerical representations of text) and enable similarity search, which is crucial for building semantic search and RAG systems.",
    
    "Embeddings convert text into numerical vectors so that machines can understand semantic meaning and compare similarity between different pieces of content.",
    
    "Large Language Models (LLMs) like GPT are trained on massive datasets and can perform tasks such as text generation, summarization, question answering, and coding assistance."
]

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
vector = embeddings.embed_documents(documents)
ids = [f"doc_{i}" for i in range(len(documents))]

vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    ids=ids,
    persist_directory="./vector_db_1",
)
query = "AI"
result = vector_store.similarity_search(query  = query,k = 2)
for data in result:
    print("\n",data.page_content, "\n")