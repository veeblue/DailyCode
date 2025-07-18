from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain_openai import ChatOpenAI

from langchain.schema import AIMessage
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)


llm = ChatOpenAI(
    model="deepseek-chat", 
    api_key=os.environ["DEEPSEEK_API_KEY"], 
    base_url="https://api.deepseek.com/v1")

# print(f"llm msg: {llm.invoke('Hello!')}")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node('chatbot', chatbot)
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('chatbot', END)
graph = graph_builder.compile()

def save_graph_visualization(graph, filename="01.png"):
    """保存工作流图可视化"""
    try:
        graph_png = graph.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(graph_png)
        print(f"工作流图已保存为: {filename}")
    except Exception as e:
        print(f"保存图片失败: {e}")

save_graph_visualization(graph, "01.png")

# 运行聊天机器人
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages":[{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("AI:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting chatbot.")
            break
        stream_graph_updates(user_input)
    except EOFError:
        # fallback if input() is not available (like in Jupyter or online terminals)
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
    except Exception as e:
        print(f"Unexpected error: {e}")
        continue

