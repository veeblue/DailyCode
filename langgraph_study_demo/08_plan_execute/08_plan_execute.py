from langchain_community.tools.tavily_search import TavilySearchResults
import os
import operator
import asyncio
from typing import Annotated, List, Tuple, TypedDict, Union, Literal
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# 禁用LangSmith追踪
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 创建TavilySearchResults工具，设置最大结果数为3
search_tool = TavilySearchResults(max_results=3)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain import hub

# 选择驱动代理的LLM，使用DeepSeek模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1"
)

# 定义一个TypedDict类PlanExecute，用于存储输入、计划、过去的步骤和响应
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

# 创建一个简单的搜索执行函数
async def execute_search(query: str) -> str:
    """执行搜索查询"""
    try:
        results = search_tool.run(query)
        return str(results)
    except Exception as e:
        return f"搜索失败: {str(e)}"

# 创建一个直接使用LLM的执行函数
async def execute_with_llm(task: str) -> str:
    """使用LLM执行任务"""
    try:
        # 首先尝试搜索相关信息
        search_query = f"搜索关于: {task}"
        search_results = await execute_search(search_query)
        
        # 然后使用LLM分析结果
        analysis_prompt = f"""
任务: {task}

搜索结果: {search_results}

请基于搜索结果回答任务中的问题。如果搜索结果不够充分，请明确指出需要更多信息。
"""
        
        response = await llm.ainvoke(analysis_prompt)
        return response.content
    except Exception as e:
        return f"任务执行失败: {str(e)}"

# 创建计划生成的提示模板
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "你是一个计划助手。为给定的目标创建一个简单的逐步计划。"
     "计划应包含独立的任务，如果正确执行将得出正确答案。"
     "不要添加多余的步骤。最后一步的结果应该是最终答案。"
     "确保每一步都有所有必要的信息。\n\n"
     "返回格式为JSON对象:\n"
     "{{\n"
     '  "steps": ["步骤1", "步骤2", "步骤3"]\n'
     "}}\n\n"
     "只返回JSON对象，不要其他内容。"),
    ("user", "{input}")
])

# 创建JSON输出解析器
plan_parser = JsonOutputParser()

# 使用指定的提示模板创建一个计划生成器
planner = planner_prompt | llm | plan_parser

# 创建重新计划的提示模板
replanner_prompt = ChatPromptTemplate.from_template(
    """你是一个重新计划助手。基于原始目标、原始计划和已完成的步骤，决定下一步要做什么。

你的目标: {input}

你的原始计划: {plan}

目前已完成的步骤: {past_steps}

如果你有足够的信息提供最终答案，返回JSON对象:
{{{{
  "action_type": "response",
  "content": "你的最终答案"
}}}}

如果你需要继续更多步骤，返回JSON对象:
{{{{
  "action_type": "plan",
  "content": ["剩余步骤1", "剩余步骤2"]
}}}}

只返回JSON对象，不要其他内容。"""
)

# 使用指定的提示模板创建一个重新计划生成器
replanner = replanner_prompt | llm | plan_parser

# 定义一个异步函数，用于生成计划步骤
async def plan_step(state: PlanExecute):
    try:
        plan_result = await planner.ainvoke({"input": state["input"]})
        steps = plan_result.get("steps", [])
        print(f"生成的计划: {steps}")
        return {"plan": steps}
    except Exception as e:
        print(f"生成计划时出错: {e}")
        # fallback计划
        return {"plan": ["搜索相关信息", "分析结果", "提供答案"]}

# 定义一个异步函数，用于执行步骤
async def execute_step(state: PlanExecute):
    plan = state["plan"]
    if not plan:
        return {"past_steps": state["past_steps"]}
    
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    
    print(f"正在执行任务: {task}")
    
    try:
        # 使用我们自定义的执行函数
        result = await execute_with_llm(task)
        print(f"执行步骤 '{task}' 的结果: {result[:200]}...")
        
        return {
            "past_steps": state["past_steps"] + [(task, result)],
        }
    except Exception as e:
        print(f"执行步骤时出错: {e}")
        return {
            "past_steps": state["past_steps"] + [(task, f"执行失败: {str(e)}")],
        }

# 定义一个异步函数，用于重新计划步骤
async def replan_step(state: PlanExecute):
    try:
        # 格式化过去的步骤
        past_steps_str = "\n".join([f"- {step}: {result[:200]}..." for step, result in state["past_steps"]])
        
        output = await replanner.ainvoke({
            "input": state["input"],
            "plan": state["plan"],
            "past_steps": past_steps_str
        })
        
        if output.get("action_type") == "response":
            return {"response": output.get("content", "处理完成")}
        else:
            new_steps = output.get("content", [])
            print(f"重新计划的步骤: {new_steps}")
            return {"plan": new_steps}
    except Exception as e:
        print(f"重新计划时出错: {e}")
        # 如果已经有一些步骤完成，尝试生成响应
        if state["past_steps"]:
            last_result = state["past_steps"][-1][1]
            return {"response": f"基于已完成的步骤，找到的信息: {last_result[:500]}..."}
        else:
            return {"response": f"处理过程中出现错误: {str(e)}"}

# 定义一个函数，用于判断是否结束
def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
    if "response" in state and state["response"]:
        return "__end__"
    else:
        return "agent"

# 定义主函数
async def main():
    # 创建一个状态图，初始化PlanExecute
    workflow = StateGraph(PlanExecute)

    # 添加计划节点
    workflow.add_node("planner", plan_step)

    # 添加执行步骤节点
    workflow.add_node("agent", execute_step)

    # 添加重新计划节点
    workflow.add_node("replan", replan_step)

    # 设置从开始到计划节点的边
    workflow.add_edge(START, "planner")

    # 设置从计划到代理节点的边
    workflow.add_edge("planner", "agent")

    # 设置从代理到重新计划节点的边
    workflow.add_edge("agent", "replan")

    # 添加条件边，用于判断下一步操作
    workflow.add_conditional_edges(
        "replan",
        # 传入判断函数，确定下一个节点
        should_end,
    )

    # 编译状态图，生成LangChain可运行对象
    app = workflow.compile()

    # 将生成的图片保存到文件
    try:
        graph_png = app.get_graph().draw_mermaid_png()
        with open("08_plan_execute.png", "wb") as f:
            f.write(graph_png)
        print("工作流图已保存: 08_plan_execute.png")
    except Exception as e:
        print(f"保存图片失败: {e}")

    # 设置配置，递归限制为50
    config = {"recursion_limit": 50}
    # 输入数据
    inputs = {"input": "2024年巴黎奥运会100米自由泳决赛冠军的家乡是哪里?请用中文答复"}
    
    # 初始化状态
    if "past_steps" not in inputs:
        inputs["past_steps"] = []
    
    # 异步执行状态图，输出结果
    try:
        print("开始执行计划...")
        async for event in app.astream(inputs, config=config):
            for k, v in event.items():
                if k != "__end__":
                    print(f"节点 {k}: {v}")
        print("执行完成！")
    except Exception as e:
        print(f"执行过程中出错: {e}")

# 运行异步函数
if __name__ == "__main__":
    asyncio.run(main())