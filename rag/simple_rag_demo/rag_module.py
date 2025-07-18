# Updated imports for latest LangChain versions
from langchain.vectorstores import Milvus
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
import os
from pymilvus import MilvusClient, exceptions
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

from langchain_core.retrievers import BaseRetriever
from typing import List, Any
from langchain_core.documents import Document

class MilvusRetriever(BaseRetriever):
    lc_namespace: str = "custom"
    tags: List[str] = []
    metadata: dict = {}

    client: Any
    collection_name: str
    embedding: HuggingFaceEmbeddings
    top_k: int = 3

    def __init__(self, client: MilvusClient, collection_name: str, embedding: HuggingFaceEmbeddings, top_k: int = 3):
        super().__init__()
        object.__setattr__(self, "client", client)
        object.__setattr__(self, "collection_name", collection_name)
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "top_k", top_k)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embedding.embed_query(query)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=self.top_k,
            output_fields=["content", "page_num"]
        )

        documents = []
        for hit in results[0]:
            content = hit.get("content", "")
            metadata = {"page_num": hit.get("page_num", -1)}
            documents.append(Document(page_content=content, metadata=metadata))
        return documents
    
# Initialize embeddings
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "mps"}  # Use "cpu" if you don't have Apple Silicon
)

# Load Milvus vector store
# vectorstore = Milvus(
#     embedding_function=embedding,
#     collection_name="rag_demo_collection",
#     connection_args={"uri": "http://localhost:19530", "token": "root:Milvus"}
# )



client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)


# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 自定义 retriever
retriever = MilvusRetriever(
    client=client,
    collection_name="rag_demo_collection",
    embedding=embedding,
    top_k=3
)

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",  # Use 'model' instead of 'model_name' for newer versions
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def answer_question(question: str):
    """Answer a question using the RAG system."""
    try:
        result = qa_chain({"query": question})  # Use dict format for newer versions
        return result["result"], result["source_documents"]
    except Exception as e:
        return f"Error: {str(e)}", []