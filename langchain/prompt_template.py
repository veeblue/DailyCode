
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
# 加载环境变量
load_dotenv()
# 禁用 LangSmith 跟踪以避免 403 错误
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0.7, 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com/v1",
    streaming=True,
    callbacks=[]
    )
def basic_prompt_template_example():
    """基础提示词模板示例"""
    print("=== 基础提示词模板示例 ===")
    
    # 创建简单的提示词模板
    template = "请告诉我关于{topic}的信息，用{language}回答。"
    prompt = PromptTemplate.from_template(template)
    
    # 格式化提示词
    formatted_prompt = prompt.format(topic="人工智能", language="中文")
    print(f"格式化后的提示词: {formatted_prompt}")
    
    # 使用LLM生成回答
    
    response = llm.invoke(formatted_prompt)
    print(f"LLM回答: {response.content}\n")


def chat_prompt_template_example():
    """聊天提示词模板示例"""
    print("=== 聊天提示词模板示例 ===")
    
    # 创建聊天提示词模板
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{role}，请用{language}回答问题。"),
        ("human", "请解释一下{topic}。")
    ])
    
    # 格式化消息
    messages = chat_template.format_messages(
        role="医生",
        language="中文",
        topic="高血压的症状"
    )
    
    # 使用LLM生成回答
    
    response = llm.invoke(messages)
    print(f"LLM回答: {response.content}\n")

def medical_diagnosis_example():
    """医疗诊断提示词模板示例"""
    print("=== 医疗诊断提示词模板示例 ===")
    
    # 创建医疗诊断的提示词模板
    diagnosis_template = ChatPromptTemplate.from_messages([
        ("system", """你是一个经验丰富的医生。请根据患者的症状进行初步分析。
        请提供：
        1. 可能的诊断
        2. 建议的检查
        3. 注意事项
        请用中文回答。"""),
        ("human", """
        患者信息：
        年龄: {age}
        性别: {gender}
        症状: {symptoms}
        病史: {medical_history}
        """)
    ])
    
    # 格式化消息
    messages = diagnosis_template.format_messages(
        age="45岁",
        gender="男性",
        symptoms="胸痛、呼吸困难、出汗",
        medical_history="高血压病史5年"
    )
    
    # 使用LLM生成回答
    
    response = llm.invoke(messages)
    print(f"医疗诊断建议: {response.content}\n")

def chain_example():
    """链式提示词模板示例"""
    print("=== 链式提示词模板示例 ===")
    
    # 创建多个提示词模板
    analysis_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个数据分析师。请分析以下数据并提取关键信息。"),
        ("human", "数据: {data}")
    ])
    
    summary_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个总结专家。请将以下分析结果进行简洁总结。"),
        ("human", "分析结果: {analysis}")
    ])
    
    # 示例数据
    sample_data = "销售额: 100万, 成本: 60万, 利润: 40万, 增长率: 15%"
    
    # 第一步：分析数据
    
    analysis_messages = analysis_template.format_messages(data=sample_data)
    analysis_response = llm.invoke(analysis_messages)
    
    # 第二步：总结分析结果
    summary_messages = summary_template.format_messages(analysis=analysis_response.content)
    summary_response = llm.invoke(summary_messages)
    
    print(f"原始数据: {sample_data}")
    print(f"分析结果: {analysis_response.content}")
    print(f"总结: {summary_response.content}\n")

def variable_validation_example():
    """变量验证示例"""
    print("=== 变量验证示例 ===")
    
    # 创建带验证的提示词模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}。"),
        ("human", "请用{language}回答关于{topic}的问题。")
    ])
    
    # 定义必需的变量
    required_variables = ["role", "language", "topic"]
    
    try:
        # 测试缺少变量的情况
        messages = template.format_messages(role="教师", language="中文")
        print("缺少变量时应该报错")
    except KeyError as e:
        print(f"缺少变量错误: {e}")
    
    # 正确的使用方式
    messages = template.format_messages(
        role="教师",
        language="中文",
        topic="数学"
    )
    print(f"正确的消息格式: {messages}\n")

def you_are_a_250():
    """你是一个250"""
    template = ChatPromptTemplate.from_messages([
        ("system", "你叫250， 如果有人问你，你就说你是一个250。"),
        ("ai", "我是一个250。"),
        ("human", "你是谁？"),
        ("human", "请用{language}回答关于{topic}的问题。"),
    ])

    # output_parser = StrOutputParser()
    chain = template | llm 
    # chain = template | llm | output_parser
    # response = chain.invoke({"language": "中文", "topic": "数学"})
    # print(response)
    for s in chain.stream({"language": "中文", "topic": "数学"}):
        print(s.content, end="", flush=True)


if __name__ == "__main__":
    # 运行所有示例
    # basic_prompt_template_example()
    # chat_prompt_template_example()
    # medical_diagnosis_example()
    # chain_example()
    # variable_validation_example()
    you_are_a_250()