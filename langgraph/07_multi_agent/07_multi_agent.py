#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多代理协作系统
使用LangGraph构建的研究员和图表生成器协作系统
"""

import os
import operator
import functools
from typing import Annotated, Sequence, TypedDict, Literal

# LangChain核心组件
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# LangGraph组件
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

# 工具
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL


# ===== 配置和工具设置 =====
def setup_matplotlib():
    """在主线程中预先配置matplotlib"""
    import os
    import matplotlib
    
    # 设置环境变量
    os.environ['MPLBACKEND'] = 'Agg'
    
    # 强制使用非交互式后端
    matplotlib.use('Agg', force=True)
    
    # 导入pyplot并关闭交互模式
    import matplotlib.pyplot as plt
    plt.ioff()
    
    print("✅ Matplotlib已配置为非交互式模式")


def setup_tools():
    """设置工具"""
    # Tavily搜索工具
    tavily_tool = TavilySearchResults(max_results=5)
    
    # Python REPL工具
    repl = PythonREPL()
    
    @tool
    def python_repl(code: Annotated[str, "要执行以生成图表的Python代码。"]):
        """使用这个工具来执行Python代码。如果你想查看某个值的输出，
        应该使用print(...)。这个输出对用户可见。
        
        对于matplotlib图表，会自动保存为图片文件。
        """
        try:
            # 强制设置matplotlib后端以避免GUI相关问题
            setup_code = """
import os
import sys
print(f"当前工作目录: {os.getcwd()}")
print(f"Python版本: {sys.version}")

# 设置matplotlib
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()
print("✅ Matplotlib配置完成")
"""
            
            # 检查是否是matplotlib相关代码
            is_matplotlib_code = any(keyword in code for keyword in ['plt.', 'matplotlib', 'pyplot'])
            
            if is_matplotlib_code:
                # 生成唯一的文件名
                import time
                timestamp = int(time.time() * 1000) % 100000
                chart_filename = f"07_chart_{timestamp}.png"
                
                # 构建完整的代码
                full_code = setup_code + "\n" + code
                
                # 确保代码以保存图表结束
                if 'plt.show()' in full_code:
                    full_code = full_code.replace('plt.show()', f'''
# 保存图表
plt.savefig("{chart_filename}", dpi=300, bbox_inches="tight")
plt.close('all')
print(f"📊 图表已保存为: {chart_filename}")
print(f"📍 完整路径: {{os.path.abspath('{chart_filename}')}}")
print("✅ 图表生成完成!")
''')
                elif any(plot_func in full_code for plot_func in ['plt.plot', 'plt.bar', 'plt.scatter', 'plt.figure']):
                    # 如果有绘图函数但没有plt.show()，添加保存代码
                    full_code += f'''

# 自动保存图表
plt.savefig("{chart_filename}", dpi=300, bbox_inches="tight")
plt.close('all')
print(f"📊 图表已保存为: {chart_filename}")
print(f"📍 完整路径: {{os.path.abspath('{chart_filename}')}}")
print("✅ 图表生成完成!")
'''
                
                # 添加文件存在检查
                full_code += f'''

# 检查文件是否成功保存
if os.path.exists("{chart_filename}"):
    file_size = os.path.getsize("{chart_filename}")
    print(f"✅ 确认文件已保存: {chart_filename} (大小: {{file_size}} bytes)")
else:
    print(f"❌ 警告: 文件 {chart_filename} 未找到")
'''
                
                result = repl.run(full_code)
            else:
                # 非matplotlib代码，直接执行
                result = repl.run(code)
            
        except BaseException as e:
            return f"❌ 执行失败。错误: {repr(e)}"
        
        result_str = f"✅ 成功执行:\n```python\n{code}\n```\n📋 输出: {result}"
        return result_str + "\n\n💡 如果你已完成所有任务，请回复FINAL ANSWER。"
    
    return tavily_tool, python_repl


# ===== 状态定义 =====
class AgentState(TypedDict):
    """代理状态类型定义"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# ===== 代理创建函数 =====
def create_agent(llm, tools, system_message: str):
    """创建一个代理"""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个有帮助的AI助手，与其他助手合作。"
            " 使用提供的工具来推进问题的回答。"
            " 如果你不能完全回答，没关系，另一个拥有不同工具的助手"
            " 会接着你的位置继续帮助。执行你能做的以取得进展。"
            " 如果你或其他助手有最终答案或交付物，"
            " 在你的回答前加上FINAL ANSWER，以便团队知道停止。"
            " 你可以使用以下工具: {tool_names}。\n{system_message}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    
    return prompt | llm.bind_tools(tools)


def agent_node(state, agent, name):
    """代理节点处理函数"""
    result = agent.invoke(state)
    
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    
    return {
        "messages": [result],
        "sender": name,
    }


# ===== 路由器函数 =====
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    """路由器函数，决定下一步行为"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "call_tool"
    
    if "FINAL ANSWER" in last_message.content:
        return "__end__"
    
    return "continue"


# ===== 主要工作流类 =====
class MultiAgentWorkflow:
    """多代理协作工作流"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="deepseek-chat", 
            api_key=os.getenv("DEEPSEEK_API_KEY"), 
            base_url="https://api.deepseek.com/v1/"
        )
        
        # 设置工具
        self.tavily_tool, self.python_repl = setup_tools()
        self.tools = [self.tavily_tool, self.python_repl]
        
        # 创建代理
        self.research_agent = create_agent(
            self.llm,
            [self.tavily_tool],
            system_message="你应该提供准确的数据供chart_generator使用。",
        )
        
        self.chart_agent = create_agent(
            self.llm,
            [self.python_repl],
            system_message="""你是一个专业的图表生成专家。
            当生成matplotlib图表时，请注意：
            1. 不要使用plt.show()，图表会自动保存为PNG文件
            2. 确保图表有清晰的标题、轴标签和网格
            3. 使用合适的颜色和样式提高可读性
            4. 设置合适的图表大小，例如plt.figure(figsize=(10, 6))
            5. 完成图表后，告诉用户图表已生成并说明图表的内容
            6. 示例代码结构：
               - 导入必要的库
               - 准备数据
               - 创建图表plt.figure(figsize=(10, 6))
               - 绘制数据plt.plot(...)
               - 添加标题和标签
               - 添加网格plt.grid(True)
               - 结束（系统会自动保存）
            """,
        )
        
        # 创建节点
        self.research_node = functools.partial(agent_node, agent=self.research_agent, name="Researcher")
        self.chart_node = functools.partial(agent_node, agent=self.chart_agent, name="chart_generator")
        self.tool_node = ToolNode(self.tools)
        
        # 构建工作流
        self.graph = self._build_workflow()
    
    def _build_workflow(self):
        """构建工作流图"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("Researcher", self.research_node)
        workflow.add_node("chart_generator", self.chart_node)
        workflow.add_node("call_tool", self.tool_node)
        
        # 添加条件边
        workflow.add_conditional_edges(
            "Researcher",
            router,
            {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
        )
        
        workflow.add_conditional_edges(
            "chart_generator",
            router,
            {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
        )
        
        workflow.add_conditional_edges(
            "call_tool",
            lambda x: x["sender"],
            {
                "Researcher": "Researcher",
                "chart_generator": "chart_generator",
            },
        )
        
        # 添加起始边
        workflow.add_edge(START, "Researcher")
        
        return workflow.compile()
    
    def save_graph_visualization(self, filename="07.png"):
        """保存工作流图可视化"""
        try:
            graph_png = self.graph.get_graph().draw_mermaid_png()
            with open(filename, "wb") as f:
                f.write(graph_png)
            print(f"工作流图已保存为: {filename}")
        except Exception as e:
            print(f"保存图片失败: {e}")
    
    def run(self, query: str, recursion_limit: int = 150):
        """运行工作流"""
        print(f"开始处理查询: {query}")
        print("=" * 50)
        
        events = self.graph.stream(
            {
                "messages": [HumanMessage(content=query)],
            },
            {"recursion_limit": recursion_limit},
        )
        
        for event in events:
            print(event)
            print("-" * 30)
        
        print("工作流完成!")


# ===== 主程序 =====
def main():
    """主程序入口"""
    # 首先配置matplotlib
    setup_matplotlib()
    
    # 创建工作流实例
    workflow = MultiAgentWorkflow()
    
    # 保存工作流图
    workflow.save_graph_visualization()
    
    # 运行示例查询
    query = (
        "获取过去5年AI软件市场规模，"
        "然后绘制一条折线图。"
        "一旦你编写好代码，完成任务。"
    )
    
    workflow.run(query)


if __name__ == "__main__":
    main()