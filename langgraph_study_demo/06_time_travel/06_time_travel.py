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
#           修复后的时间旅行工具
# =======================================
@tool
def show_history(tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """显示对话历史状态快照"""
    try:
        # 获取历史状态
        history = list(graph.get_state_history(config))
        
        if not history:
            return "没有找到历史状态"
        
        # 反转历史顺序，使最旧的在前面
        history = list(reversed(history))
        
        history_info = []
        for i, snapshot in enumerate(history):
            # 处理时间戳，可能是字符串或datetime对象
            if hasattr(snapshot.created_at, 'strftime'):
                timestamp = snapshot.created_at.strftime("%H:%M:%S")
            else:
                timestamp = str(snapshot.created_at)
            
            # 显示步骤信息包括checkpoint_id
            step_info = f"步骤 {i}: {timestamp} (ID: {snapshot.config.get('configurable', {}).get('checkpoint_id', 'N/A')[:8]}...)"
            
            # 获取状态信息
            state_info = []
            if snapshot.values.get("name"):
                state_info.append(f"名称: {snapshot.values['name']}")
            if snapshot.values.get("birthday"):
                state_info.append(f"生日: {snapshot.values['birthday']}")
            
            # 获取最后一条消息
            if snapshot.values.get("messages"):
                last_msg = snapshot.values["messages"][-1]
                if isinstance(last_msg, HumanMessage):
                    state_info.append(f"用户: {last_msg.content[:50]}...")
                elif isinstance(last_msg, AIMessage):
                    state_info.append(f"AI: {last_msg.content[:50]}...")
                elif isinstance(last_msg, ToolMessage):
                    state_info.append(f"工具: {last_msg.content[:50]}...")
            
            if state_info:
                step_info += f" - {', '.join(state_info)}"
            
            history_info.append(step_info)
        
        result = "对话历史状态 (按时间顺序):\n" + "\n".join(history_info)
        return result
        
    except Exception as e:
        return f"获取历史状态时出错: {str(e)}"

# 全局变量存储选中的检查点配置
selected_checkpoint_config = None

@tool
def select_checkpoint(step_number: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """选择要回退到的检查点"""
    global selected_checkpoint_config
    
    try:
        # 获取历史状态
        history = list(graph.get_state_history(config))
        
        if not history:
            return "没有找到历史状态"
        
        # 反转历史顺序，使最旧的在前面
        history = list(reversed(history))
        
        if step_number < 0 or step_number >= len(history):
            return f"无效的步骤号。可用范围: 0-{len(history)-1}"
        
        # 获取指定步骤的快照
        target_snapshot = history[step_number]
        
        # 保存检查点配置供后续使用
        selected_checkpoint_config = target_snapshot.config
        
        # 获取检查点信息
        state_info = []
        if target_snapshot.values.get("name"):
            state_info.append(f"名称: {target_snapshot.values['name']}")
        if target_snapshot.values.get("birthday"):
            state_info.append(f"生日: {target_snapshot.values['birthday']}")
        
        checkpoint_id = target_snapshot.config.get('configurable', {}).get('checkpoint_id', 'N/A')
        
        result = f"已选择检查点 {step_number} (ID: {checkpoint_id[:8]}...)"
        if state_info:
            result += f" - {', '.join(state_info)}"
        result += "\n现在可以使用 'continue' 命令从这个检查点继续对话。"
        
        return result
        
    except Exception as e:
        return f"选择检查点时出错: {str(e)}"

# =======================================

tools = [search_tool, human_assistance, show_history, select_checkpoint]

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
def save_graph_visualization(graph, filename="06.png"):
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

def stream_graph_updates(user_input: str, custom_config=None):
    """流式处理用户输入"""
    # 使用自定义配置（用于时间旅行）或默认配置
    current_config = custom_config if custom_config else config
    
    # 初始化对话状态
    state = {"messages": [HumanMessage(content=user_input)]}
    
    for event in graph.stream(state, config=current_config):
        for node, output in event.items():
            messages = output["messages"]
            last_msg = messages[-1]
            
            if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                print("AI:", last_msg.content)
            elif isinstance(last_msg, ToolMessage):
                print(f"[工具执行] {last_msg.content}")

def continue_from_checkpoint():
    """从选中的检查点继续对话"""
    global selected_checkpoint_config
    
    if not selected_checkpoint_config:
        print("没有选中的检查点。请先使用 'select <步骤号>' 选择一个检查点。")
        return
    
    print(f"从检查点继续对话...")
    
    # 从检查点恢复并继续执行
    try:
        for event in graph.stream(None, selected_checkpoint_config, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                    print("AI:", last_msg.content)
                elif isinstance(last_msg, ToolMessage):
                    print(f"[工具执行] {last_msg.content}")
                    
        # 重置选中的检查点
        selected_checkpoint_config = None
        print("\n已从检查点恢复完成。")
    except Exception as e:
        print(f"从检查点恢复时出错: {str(e)}")

# =======================================
#               交互式命令
# =======================================
def show_commands():
    """显示可用命令"""
    print("\n可用命令:")
    print("- 'history' 或 'h': 显示对话历史状态")
    print("- 'select <数字>' 或 's <数字>': 选择要回退的检查点")
    print("- 'continue' 或 'c': 从选中的检查点继续对话")
    print("- 'exit', 'quit', 'q': 退出程序")
    print("- 'help': 显示此帮助信息")
    print()

# 主聊天循环
print("欢迎使用带时间旅行功能的聊天机器人!")
show_commands()

while True:
    try:
        user_input = input("User: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("对话结束")
            break
        elif user_input.lower() in ["help"]:
            show_commands()
            continue
        elif user_input.lower() in ["history", "h"]:
            # 直接调用历史显示功能
            try:
                history = list(graph.get_state_history(config))
                if not history:
                    print("没有找到历史状态")
                    continue
                
                # 反转历史顺序，使最旧的在前面
                history = list(reversed(history))
                
                print("\n对话历史状态 (按时间顺序):")
                for i, snapshot in enumerate(history):
                    # 处理时间戳，可能是字符串或datetime对象
                    if hasattr(snapshot.created_at, 'strftime'):
                        timestamp = snapshot.created_at.strftime("%H:%M:%S")
                    else:
                        timestamp = str(snapshot.created_at)
                    
                    checkpoint_id = snapshot.config.get('configurable', {}).get('checkpoint_id', 'N/A')
                    step_info = f"步骤 {i}: {timestamp} (ID: {checkpoint_id[:8]}...)"
                    
                    # 获取状态信息
                    state_info = []
                    if snapshot.values.get("name"):
                        state_info.append(f"名称: {snapshot.values['name']}")
                    if snapshot.values.get("birthday"):
                        state_info.append(f"生日: {snapshot.values['birthday']}")
                    
                    # 获取最后一条消息
                    if snapshot.values.get("messages"):
                        last_msg = snapshot.values["messages"][-1]
                        if isinstance(last_msg, HumanMessage):
                            state_info.append(f"用户: {last_msg.content[:50]}...")
                        elif isinstance(last_msg, AIMessage):
                            state_info.append(f"AI: {last_msg.content[:50]}...")
                        elif isinstance(last_msg, ToolMessage):
                            state_info.append(f"工具: {last_msg.content[:50]}...")
                    
                    if state_info:
                        step_info += f" - {', '.join(state_info)}"
                    
                    print(step_info)
                print()
            except Exception as e:
                print(f"获取历史状态时出错: {str(e)}")
            continue
        elif user_input.lower().startswith("select ") or user_input.lower().startswith("s "):
            # 选择检查点
            try:
                parts = user_input.split()
                if len(parts) < 2:
                    print("请指定要选择的步骤号，例如: select 0")
                    continue
                
                step_number = int(parts[1])
                
                history = list(graph.get_state_history(config))
                if not history:
                    print("没有找到历史状态")
                    continue
                
                # 反转历史顺序，使最旧的在前面
                history = list(reversed(history))
                
                if step_number < 0 or step_number >= len(history):
                    print(f"无效的步骤号。可用范围: 0-{len(history)-1}")
                    continue
                
                # 获取指定步骤的快照
                target_snapshot = history[step_number]
                
                # 保存检查点配置供后续使用
                selected_checkpoint_config = target_snapshot.config
                
                # 获取检查点信息
                state_info = []
                if target_snapshot.values.get("name"):
                    state_info.append(f"名称: {target_snapshot.values['name']}")
                if target_snapshot.values.get("birthday"):
                    state_info.append(f"生日: {target_snapshot.values['birthday']}")
                
                checkpoint_id = target_snapshot.config.get('configurable', {}).get('checkpoint_id', 'N/A')
                
                result = f"已选择检查点 {step_number} (ID: {checkpoint_id[:8]}...)"
                if state_info:
                    result += f" - {', '.join(state_info)}"
                result += "\n现在可以使用 'continue' 命令从这个检查点继续对话。"
                
                print(result)
                
            except ValueError:
                print("请输入有效的步骤号")
            except Exception as e:
                print(f"选择检查点时出错: {str(e)}")
            continue
        elif user_input.lower() in ["continue", "c"]:
            # 从选中的检查点继续
            continue_from_checkpoint()
            continue
        
        # 正常处理用户输入
        stream_graph_updates(user_input)
        
    except Exception as e:
        print(f"错误: {e}")
        continue