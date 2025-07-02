from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from transformers import Tool

import logging

logging.basicConfig(level=logging.INFO)

from search_demo.loaders.pdf_loader import MyPdfLoader
logging.info("Loading PDF")
pdf_loader = MyPdfLoader("/Users/yee/vscode/study/test_demo/search_demo/data/deepseek-v2-tech-report.pdf")
documents = pdf_loader.load()
logging.info(f"Found {len(documents)} documents")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunked_documents = splitter.split_documents(documents)
logging.info(f"Splitted {len(chunked_documents)} documents")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(
    documents=chunked_documents,
    embedding=embedding,
)
logging.info(f"Loaded  embeddings")


#建立工具
# 1. Tavily搜索工具
# 2. 向量存储检索工具
import os
tools = [] # init tools list
tavily_tool = TavilySearch(tavily_api_key=os.getenv("TAVILY_API_KEY"), max_results=5)

tools.append(Tool(
    name="web_search",
    func=tavily_tool.run,
    description="使用Tavily搜索引擎获取实时信息"
))

doc_retriever = vector_store.as_retriever()
@tool
def document_search_func(q: str) -> str:
    """使用上传的PDF/CSV文档回答问题"""
    docs = doc_retriever.get_relevant_documents(q)
    last_references = [d.metadata for d in docs]
    return "\n\n".join([d.page_content for d in docs])

tools.append(Tool(
    name="document_search",
    func=document_search_func,
    description="使用上传的PDF/CSV文档回答问题"
))
prompt = """
你是一个聪明的助手，善于使用工具解决问题。

你可以使用以下工具来获取信息：{tools}
{tool_names}

说明：
- 使用 `document_search` 工具来查询本地上传的 PDF 或 CSV 文档内容。
- 如果 `document_search` 得不到答案，才考虑使用 `web_search` 获取实时信息。

请严格按照以下格式进行推理：

Question: 用户的问题  
Thought: 你对此问题的思考  
Action: 要使用的工具名（document_search 或 web_search）  
Action Input: 给工具的输入内容  

Observation: 工具返回的结果  
... （可重复多次 Thought/Action/Observation）

Final Answer: 整合后的最终答案，回答用户问题，并说明查询结束。

现在请开始回答下面的问题：

Question: {input}
{agent_scratchpad}
"""
prompt_template = PromptTemplate.from_template(template=prompt)
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
)
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template,
)
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 执行Agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

while True:
    user_input = input("请输入查询内容（输入 q 退出）：")
    if user_input.strip().lower() == "q":
        break

    response = agent_executor.invoke({"input": user_input})
    print(f"AI：{response['output']}")