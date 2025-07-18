from langchain_openai import ChatOpenAI
# pip install -qU langchain langchain-openai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import XMLOutputParser
import os
model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    )
query = "生成周星驰的简化电影作品列表，按照最新的时间降序"

parser = XMLOutputParser()

instructions = """请以如下 XML 格式回答：

<movies>
  <movie>
    <title>电影名</title>
    <year>年份</year>
  </movie>
</movies>
"""
prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions":instructions},
)

# chain = prompt | model | parser
chain = prompt | model

response = chain.invoke({"query": query})

print(response.content)