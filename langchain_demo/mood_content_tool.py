
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import os

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def mood_tool(mood: Literal["开心的😄", "生气的😠", "沮丧的😢"]) -> None:
    """Describe the weather"""
    pass

llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0.7, 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base="https://api.deepseek.com/v1",
    )

llm_with_tools = llm.bind_tools([mood_tool])

# prompt = HumanMessage(
#     content=[
#         {"type": "text", "text": "用中文描述当前文本的情绪"},
#         {"type": "text", "text": "哎，今天考试又没考好。"},
#         {"type": "text", "text": "耶，今天买彩票中奖了。"},
#     ],
# )
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "用中文描述当前文本的情绪",
    ),
    ("human", "{input1}"),
    MessagesPlaceholder("agent_scratchpad"),
])
msg = prompt.format_messages(input1="哎，今天考试又没考好。", agent_scratchpad=[])
response = llm_with_tools.invoke(msg)
print(response.tool_calls)