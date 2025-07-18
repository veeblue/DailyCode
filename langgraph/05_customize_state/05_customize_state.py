from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt 
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolCallId, tool
# 环境设置
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# 状态定义
class State(TypedDict):
    messages: Annotated[List[dict], add_messages]
    name: str  # 添加名称字段   <-- 新增
    birthday: str  # 添加生日字段   <-- 新增

graph_builder = StateGraph(State)

# =======================================
#               工具设置
# =======================================
# 1. 搜索工具
from langchain_tavily import TavilySearch

search_tool = TavilySearch(max_results=2)

# =======================================
import json
@tool
def human_assistance(name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """Request assistance from a human."""
    print(f"\n[需要人工协助] 名称: {name}, 生日: {birthday}")
    content = {
         "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
    }
    raw_input = input(f"请提供协助（输入JSON）: {content}\n")
    try:
        human_response = json.loads(raw_input)
    except json.JSONDecodeError:
        print("输入格式错误，请输入 JSON 格式的字符串。")
        return "Invalid input."
     # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return f"Human expert provided: {state_update}"
# =======================================

tools = [search_tool, human_assistance]

# 2. 初始化LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1"
)

# 3. 绑定工具
llm_with_tools = llm.bind_tools(tools)
# =======================================
#               节点定义
# =======================================
from langgraph.prebuilt import ToolNode
tool_node = ToolNode(tools=tools)

def chatbot(state: State):
    """聊天机器人节点 - 处理用户输入和工具结果"""
    # 获取完整对话历史
    messages = state["messages"]
    # 调用LLM
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# =======================================
#               路由逻辑
# =======================================
def route_tool(state: State):
    """路由决策：判断是否需要调用工具"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # 检查最后一条消息是否包含工具调用
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"

# =======================================
#               构建工作流
# =======================================
# 添加节点
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# 设置入口点
graph_builder.add_edge(START, "chatbot")

# 设置条件路由
graph_builder.add_conditional_edges(
    "chatbot",
    route_tool,
    {
        "tools": "tools",  # 需要调用工具
        "end": END  # 直接结束
    }
)

# 工具执行后返回给chatbot
graph_builder.add_edge("tools", "chatbot")

# 增加存储器
memory = MemorySaver() #在生产应用程序中，您可能会将其更改为使用SqliteSaver或PostgresSaver连接数据库。
# 编译工作流
graph = graph_builder.compile(checkpointer=memory) # <------- 添加内存检查点

# =======================================
#               可视化 & 执行
# =======================================
def save_graph_visualization(graph, filename="05.png"):
    """保存工作流可视化"""
    try:
        graph_png = graph.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(graph_png)
        print(f"工作流图已保存: {filename}")
    except Exception as e:
        print(f"保存失败: {e}")

save_graph_visualization(graph)

config = {"configurable": {"thread_id": "1"}} 

def stream_graph_updates(user_input: str):
    """流式处理用户输入"""
    # 初始化对话状态
    state = {"messages": [HumanMessage(content=user_input)]}
    
    for event in graph.stream(state, config=config): # <------- 添加配置参数
        for node, output in event.items():
            messages = output["messages"]
            last_msg = messages[-1]
            
            if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                print("AI:", last_msg.content)
            elif isinstance(last_msg, ToolMessage):
                print(f"[工具执行] {last_msg.content}")

# 主聊天循环
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("对话结束")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print(f"错误: {e}")
        continue