import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph

from typing import Literal


@tool
def search_tool(query: str) -> str:
    """根据用户的查询返回当前天气信息。"""
    if '上海' in query.lower() or 'shanghai' in query.lower():
        return "上海的天气是晴朗的，温度在25°C左右。"
    return f"现在是35°C，天气炎热。请注意防暑！"


tools = [search_tool]

# 创建工具节点
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools=tools)

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
)

# 绑定工具到LLM
llm_with_tools = llm.bind_tools(tools)

from langchain_core.messages import AIMessage


def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    # 修复：返回的键应该是 "messages"，不是 "message"
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")  # 设置入口点为agent节点

workflow.add_conditional_edges(
    "agent",
    should_continue
)

# 修复：添加从tools回到agent的边
workflow.add_edge("tools", "agent")

checkpoint = MemorySaver()

app = workflow.compile(checkpointer=checkpoint)

# 第一次调用
final_state = app.invoke(
    {"messages": [HumanMessage(content="上海的天气怎么样？")]},
    config={"configurable": {"thread_id": "12345"}}
)

# 修复：访问最后一条消息的方式
print("第一次回答:")
print(final_state["messages"][-1].content)
print()

# 第二次调用
final_state = app.invoke(
    {"messages": [HumanMessage(content="我问哪个城市？")]},
    config={"configurable": {"thread_id": "12345"}}
)

print("第二次回答:")
print(final_state["messages"][-1].content)