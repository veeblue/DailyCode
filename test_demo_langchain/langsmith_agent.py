from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import JsonOutputParser
openai_api_key=os.getenv("DEEPSEEK_API_KEY")
from langchain_deepseek import ChatDeepSeek
import asyncio
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_tavily import TavilySearch
from langchain.globals import set_debug
print(f"key ==> {openai_api_key}")
llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0.7, 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base="https://api.deepseek.com/v1",
    streaming=True,
    callbacks=[]
    )

# print(llm.invoke("今天九江的天气怎么样？"))
# json = JsonOutputParser()

# chain = llm | json
print(f"tavily key ==> {os.getenv('TAVILY_API_KEY')}")
tools = [TavilySearch(max_results=1, api_key=os.getenv("TAVILY_API_KEY"))]

prompt = ChatPromptTemplate.from_messages([
    
        (
            "system",
            "你是一位得力的助手。",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    
])

agent = create_tool_calling_agent(llm, tools, prompt)
set_debug(True)
agent_executor = AgentExecutor(agent=agent, tools=tools)

response = agent_executor.invoke(
    {"input": "谁执导了2023年的电影《奥本海默》，他多少岁了？"}
)
print(response)
# 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 