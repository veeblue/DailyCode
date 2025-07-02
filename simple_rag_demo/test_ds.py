
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",  # 替换为 deepseek 的 API 地址
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model_name="deepseek-chat",             # 例如 deepseek-chat / deepseek-coder
)

messages = [HumanMessage(content="请介绍一下强化学习")]
print(llm(messages).content)