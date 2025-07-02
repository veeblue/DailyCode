"""
Streamlit RAG Demo
==================
基于已有的 PDF→Milvus 向量化流程，提供一个 Web UI，
支持：
- 在 Milvus 中检索文档片段
- 结合 DeepSeek LLM 进行问答
- 展示答案及引用来源
运行：
$ streamlit run rag_webui.py
环境变量：
DEEPSEEK_API_KEY   # DeepSeek 密钥
DEEPSEEK_API_BASE  # 非必填，默认 https://api.deepseek.com/v1
"""

import os
import logging
import streamlit as st
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings  # 建议安装 langchain-huggingface
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ---------- 配置 ---------- #
DEFAULT_COLLECTION_NAME = "rag_demo_collection"
DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_MILVUS_TOKEN = "root:Milvus"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

PROMPT_TEMPLATE = (
    """已知信息:\n{context}\n\n"
    "问题: {question}\n\n"
    "请基于以上已知信息，用中文简洁回答问题；"
    "如无法从中得到答案，请回答\"无法从所给文档中找到答案\"。"""
)

# ---------- 日志 ---------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RAGWebUI")

# ---------- 工具函数 ---------- #
@st.cache_resource(show_spinner="♻️ 正在初始化向量检索器 …")
def get_vectorstore(collection_name: str, uri: str, token: str):
    """连接 Milvus 并返回 VectorStore 对象"""
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "mps"})
    vs = Milvus(
        embedding_function=embedding,
        collection_name=collection_name,
        connection_args={"uri": uri, "token": token},
        consistency_level="Strong",

    )
    return vs

@st.cache_resource(show_spinner="🤖 正在加载 LLM …")
def get_llm():
    """初始化 DeepSeek ChatOpenAI"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("请设置环境变量 DEEPSEEK_API_KEY！")
        st.stop()
    llm = ChatOpenAI(
        model_name="deepseek-chat",
        temperature=0.2,
        streaming=True,
        openai_api_key=api_key,
        openai_api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
    )
    return llm

# ---------- Streamlit UI ---------- #
st.set_page_config(page_title="📚 RAG Demo", page_icon="📚", layout="wide")
st.title("📚 RAG 问答演示")

with st.sidebar:
    st.header("🔧 Milvus 配置")
    collection_name = st.text_input("Collection 名称", value=DEFAULT_COLLECTION_NAME)
    milvus_uri = st.text_input("Milvus URI", value=DEFAULT_MILVUS_URI)
    milvus_token = st.text_input("Milvus Token", value=DEFAULT_MILVUS_TOKEN, type="password")
    st.divider()
    st.header("🛠️ DeepSeek LLM")
    if "DEEPSEEK_API_KEY" not in os.environ:
        st.text_input("DeepSeek API Key", key="ds_api_key", type="password", on_change=lambda: os.environ.update({"DEEPSEEK_API_KEY": st.session_state.ds_api_key}))
    st.caption("✴️ 如果未填写，则使用系统环境变量")

# 初始化向量检索器和 LLM（带缓存）
vectorstore = get_vectorstore(collection_name, milvus_uri, milvus_token)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
llm = get_llm()
prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

# 聊天输入
if "messages" not in st.session_state:
    st.session_state.messages = []  # 存储 (role, content, sources)

for msg in st.session_state.messages:
    with st.chat_message("user") if msg[0] == "user" else st.chat_message("assistant"):
        st.markdown(msg[1])
        if msg[2]:
            with st.expander("引用文档"):
                for i, doc in enumerate(msg[2]):
                    st.markdown(f"**片段 {i+1} (page {doc.metadata.get('page_num', '-')})**:\n\n{doc.page_content}")

query = st.chat_input("输入你的问题，回车发送 …")
if query:
    # 记录用户消息
    st.session_state.messages.append(("user", query, None))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            result = qa_chain({"query": query})
            answer = result["result"]
            sources = result.get("source_documents", [])
            placeholder.markdown(answer)
        except Exception as e:
            logger.exception("问答异常")
            placeholder.error(f"💥 出错: {e}")
            answer, sources = f"出错: {e}", []

    # 保存到历史
    st.session_state.messages.append(("assistant", answer, sources))
