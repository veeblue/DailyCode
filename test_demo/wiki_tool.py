from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from langchain_tavily import TavilySearch

from langchain_openai import ChatOpenAI
import os
model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    )
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=100,
    description="使用维基百科搜索事实类知识")
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = TavilySearch(
    max_results=1,
    api_key=os.getenv("TAVILY_API_KEY"),
    description="使用搜索引擎获取最新实时信息，如天气、新闻等")

tools = [wiki, search]



# 使用推荐的 tool-calling prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个聪明的助手，可以调用工具来帮助用户解决问题。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # 必加
])
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 执行
response = agent_executor.invoke({"input": "请告诉我关于猫的知识？今天上海天气？", "agent_scratchpad": []})
print(response["output"])

