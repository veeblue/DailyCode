# 示例：callback_process.py
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
import os
class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")


prompt = ChatPromptTemplate.from_messages(["给我讲个关于{animal}的笑话，限制20个字"])
# 为启用流式处理，我们在ChatModel构造函数中传入`streaming=True`
# 另外，我们将自定义处理程序作为回调参数的列表传入
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    streaming=True,
    callbacks=[MyCustomHandler()],
    )
chain = prompt | llm
response = chain.invoke({"animal": "猫"})
# print(response.content)