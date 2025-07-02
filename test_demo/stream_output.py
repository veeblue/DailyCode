from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import JsonOutputParser
openai_api_key=os.getenv("DEEPSEEK_API_KEY")
from langchain_deepseek import ChatDeepSeek
import asyncio
from langchain_core.output_parsers import StrOutputParser
# llm = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0.7,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com/v1",
#     streaming=True,
#     callbacks=[]
#     )

# print(llm.invoke("Hello, how are you?"))


print(f"key ==> {openai_api_key}")
llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0.7, 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base="https://api.deepseek.com/v1",
    streaming=True,
    callbacks=[]
    )

# print(llm.invoke("今天九江的天气怎么样？"))
# json = JsonOutputParser()

# chain = llm | json

str = StrOutputParser()
async def async_stream():
    async for text in chain.astream(
            "以JSON 格式输出法国、西班牙和日本的国家及其人口列表。"
            '使用一个带有“countries”外部键的字典，其中包含国家列表。'
            "每个国家都应该有键`name`和`population`"
    ):
        print(text, flush=True)


# 运行异步流处理
# asyncio.run(async_stream())

chain = llm | str
print("start")
for text in chain.stream(
            "出师表原文"
    ):
        print(text, end="", flush=True)