from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable

# 调用你本地服务中的 deepseek endpoint
deepseek = RemoteRunnable("http://localhost:8000/deepseek/")

# 正确的 prompt 模板（system + user）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个喜欢写故事的助手。"),
    ("user", "写一个故事，主题是：{topic}")
])

# 串联：prompt 输出 → RemoteRunnable 输入
chain = prompt | RunnableMap({"output": deepseek})

# 使用 batch 调用
response = chain.batch([{"topic": "猫"}])
print(response)