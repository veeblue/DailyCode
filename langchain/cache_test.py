from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一个助理，擅长{ability},回答在20词及以上。"),
         MessagesPlaceholder(variable_name="history"),
         ("human","{input}")
    ]
)

llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0.7, 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base="https://api.deepseek.com/v1",
    )
str_out = StrOutputParser()
result = prompt | llm | str_out

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

response = result.invoke({
    "ability": "数学",
    "input": "什么是余弦相似度？",
    "history": []  # 如果用了 MessagesPlaceholder，一定要提供 history
})

print(response)

response1 = result.invoke({
    "ability": "数学",
    "input": "什么是余弦相似度？",
    "history": []  # 如果用了 MessagesPlaceholder，一定要提供 history
})
print(response1)