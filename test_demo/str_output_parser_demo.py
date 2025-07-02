from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

# 设置API密钥
os.environ["DEEPSEEK_API_KEY"] = "your-api-key-here"

# 初始化LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
)

print("=== 1. 基本用法 ===")
# 不使用解析器 - 返回AIMessage对象
response_without_parser = llm.invoke("请简单介绍一下Python")
print("不使用StrOutputParser:")
print(f"类型: {type(response_without_parser)}")
print(f"内容: {response_without_parser}")
print()

# 使用StrOutputParser - 返回字符串
str_parser = StrOutputParser()
chain_with_str = llm | str_parser
response_with_str = chain_with_str.invoke("请简单介绍一下Python")
print("使用StrOutputParser:")
print(f"类型: {type(response_with_str)}")
print(f"内容: {response_with_str}")
print()

print("=== 2. 流式处理 ===")
# 流式输出
print("流式输出:")
for chunk in chain_with_str.stream("请写一首关于春天的短诗"):
    print(chunk, end="", flush=True)
print("\n")

print("=== 3. 与JsonOutputParser对比 ===")
# 创建JSON格式的提示
json_prompt = ChatPromptTemplate.from_template(
    "请以JSON格式返回以下信息：{topic}。格式：{{\"name\": \"值\", \"description\": \"描述\"}}"
)

# 使用JsonOutputParser
json_parser = JsonOutputParser()
json_chain = json_prompt | llm | json_parser
json_response = json_chain.invoke({"topic": "Python编程语言"})
print("JsonOutputParser输出:")
print(f"类型: {type(json_response)}")
print(f"内容: {json_response}")
print()

# 使用StrOutputParser（不解析JSON）
str_chain = json_prompt | llm | str_parser
str_response = str_chain.invoke({"topic": "Python编程语言"})
print("StrOutputParser输出（JSON字符串）:")
print(f"类型: {type(str_response)}")
print(f"内容: {str_response}")
print()

print("=== 4. 在复杂链中使用 ===")
# 创建多步骤链
complex_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的翻译助手，请将用户输入翻译成英文。"),
    ("human", "{input}")
])

# 构建链：提示模板 -> LLM -> 字符串解析器
translation_chain = complex_prompt | llm | str_parser

# 执行翻译
translation = translation_chain.invoke({"input": "你好，世界！"})
print("翻译结果:")
print(translation)
print()

print("=== 5. 错误处理 ===")
# StrOutputParser 的错误处理
try:
    # 正常情况
    normal_response = str_parser.invoke(llm.invoke("Hello"))
    print("正常解析:", normal_response)
    
    # 如果输入不是AIMessage，会抛出错误
    # str_parser.invoke("直接字符串")  # 这会报错
    
except Exception as e:
    print(f"解析错误: {e}")

print("\n=== 6. 性能对比 ===")
import time

# 测试不使用解析器的性能
start_time = time.time()
for _ in range(3):
    llm.invoke("测试消息")
end_time = time.time()
print(f"不使用解析器耗时: {end_time - start_time:.4f}秒")

# 测试使用StrOutputParser的性能
start_time = time.time()
for _ in range(3):
    chain_with_str.invoke("测试消息")
end_time = time.time()
print(f"使用StrOutputParser耗时: {end_time - start_time:.4f}秒")

print("\n=== 总结 ===")
print("StrOutputParser的主要特点:")
print("1. 简单易用 - 将AIMessage转换为字符串")
print("2. 流式支持 - 支持.stream()和.astream()方法")
print("3. 轻量级 - 几乎没有性能开销")
print("4. 通用性 - 适用于大多数文本输出场景")
print("5. 链式操作 - 可以轻松集成到LCEL链中") 