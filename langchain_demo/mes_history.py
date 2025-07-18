from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_huggingface import HuggingFaceEmbeddings
import os
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

runnable = prompt | llm

store = {}

# def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
#     if (user_id, conversation_id) not in store:
#         store[(user_id, conversation_id)] = ChatMessageHistory()
#     return store[(user_id, conversation_id)]

# with_message_history = RunnableWithMessageHistory(
#     runnable,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="history",
#     history_factory_config=[
#         ConfigurableFieldSpec(
#             id="user_id",
#             annotation=str,
#             name="User ID",
#             description="用户的唯一标识符。",
#             default="",
#             is_shared=True,
#         ),
#         ConfigurableFieldSpec(
#             id="conversation_id",
#             annotation=str,
#             name="Conversation ID",
#             description="对话的唯一标识符。",
#             default="",
#             is_shared=True,
#         ),
#     ],
# )

# print("=====================[ 1 ]=========================")
# response = with_message_history.invoke(
#     {"ability": "math", "input": "余弦是什么意思？"},
#     config={"configurable": {"user_id": "123", "conversation_id": "1"}},
# )
# print(response)
# #content='余弦是一个三角函数，它表示直角三角形的邻边长度和斜边长度的比值。' response_metadata={'token_usage': {'completion_tokens': 33, 'prompt_tokens': 38, 'total_tokens': 71}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-2d1eba02-4709-4db5-ab6b-0fd03ab4c68a-0' usage_metadata={'input_tokens': 38, 'output_tokens': 33, 'total_tokens': 71}

# print("======================[ 2 ]========================")
# # 记住
# response = with_message_history.invoke(
#     {"ability": "math", "input": "什么?"},
#     config={"configurable": {"user_id": "123", "conversation_id": "1"}},
# )
# print(response)
# #content='余弦是一个数学术语，代表在一个角度下的邻边和斜边的比例。' response_metadata={'token_usage': {'completion_tokens': 32, 'prompt_tokens': 83, 'total_tokens': 115}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-99368d03-c2ed-4dda-a32f-677c036ad676-0' usage_metadata={'input_tokens': 83, 'output_tokens': 32, 'total_tokens': 115}

# print("======================[ 3 ]========================")
# # 新的 user_id --> 不记得了。
# response = with_message_history.invoke(
#     {"ability": "math", "input": "什么?"},
#     config={"configurable": {"user_id": "123", "conversation_id": "2"}},
# )
# print(response)

print("======================[ session_id ]======================")

# store_session_id = {}

# def get_session_history_session_id(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store_session_id[session_id] = ChatMessageHistory()
#     return store_session_id[session_id]

# with_message_history_session_id = RunnableWithMessageHistory(
#     runnable,
#     get_session_history_session_id,
#     input_messages_key="input",
#     history_messages_key="history",
# )

# response = with_message_history_session_id.invoke(
#     {"ability": "math", "input": "余弦是什么意思？"},
#     config={"configurable": {"session_id": "abc123"}},
# )
# print(response)
# # content="Cosine is a trigonometric function comparing the ratio of an angle's adjacent side to its hypotenuse in a right triangle." response_metadata={'token_usage': {'completion_tokens': 27, 'prompt_tokens': 33, 'total_tokens': 60}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-c383660d-7195-4b36-9175-992f05739ece-0' usage_metadata={'input_tokens': 33, 'output_tokens': 27, 'total_tokens': 60}

# # 记住
# response = with_message_history_session_id.invoke(
#     {"ability": "math", "input": "什么?"},
#     config={"configurable": {"session_id": "abc123"}},
# )
# print(response)

# # 新的 session_id --> 不记得了。
# response = with_message_history_session_id.invoke(
#     {"ability": "math", "input": "什么?"},
#     config={"configurable": {"session_id": "def234"}},
# )
# print(response)


print("======================[ redis ]======================")
from langchain_community.chat_message_histories import RedisChatMessageHistory
REDIS_URL = "redis://localhost:6379/0"

def get_message_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)
response = with_message_history.invoke(
    {"ability": "math", "input": "余弦是什么意思？"},
    config={"configurable": {"session_id": "abc123"}},
)
print(response)
# content="Cosine is a trigonometric function comparing the ratio of an angle's adjacent side to its hypotenuse in a right triangle." response_metadata={'token_usage': {'completion_tokens': 27, 'prompt_tokens': 33, 'total_tokens': 60}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-c383660d-7195-4b36-9175-992f05739ece-0' usage_metadata={'input_tokens': 33, 'output_tokens': 27, 'total_tokens': 60}

# 记住
response = with_message_history.invoke(
    {"ability": "math", "input": "我提了什么问题?"},
    config={"configurable": {"session_id": "abc123"}},
)
print(response)

# 新的 session_id --> 不记得了。
response = with_message_history.invoke(
    {"ability": "math", "input": "什么?"},
    config={"configurable": {"session_id": "def234"}},
)
print(response)