import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import tempfile
import os
from langchain_tavily import TavilySearch
from loaders.csv_loader import MyCSVLoader
from loaders.pdf_loader import MyPdfLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 初始化设置
st.set_page_config(page_title="智能研究助手", layout="wide")
st.title("🔍 智能研究助手 - PDF/CSV分析 + 实时搜索")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "last_references" not in st.session_state:
    st.session_state.last_references = []

# 侧边栏设置
with st.sidebar:
    st.header("📁 文档上传")
    uploaded_files = st.file_uploader(
        "上传PDF或CSV文件",
        type=["pdf", "csv"],
        accept_multiple_files=True
    )

    # 处理上传文件
    if uploaded_files:
        documents = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name

                try:
                    if file.name.endswith('.pdf'):
                        loader = MyPdfLoader(tmp_path)
                        documents.extend(loader.load())
                    elif file.name.endswith('.csv'):
                        loader = MyCSVLoader(tmp_path)
                        documents.extend(loader.load())
                except Exception as e:
                    st.error(f"处理 {file.name} 时出错: {str(e)}")
                finally:
                    os.unlink(tmp_path)

        if documents:
            # 文档切割
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunked_documents = splitter.split_documents(documents)
            # 创建向量存储
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_store = FAISS.from_documents(chunked_documents, embeddings)
            st.success(f"成功加载 {len(chunked_documents)} 个文档片段!")

    st.divider()
    st.markdown("### 配置选项")
    search_enabled = st.checkbox("启用Tavily搜索", value=True)
    chunk_size = st.number_input("分块大小 (chunk_size)", min_value=100, max_value=2000, value=500, step=100)
    chunk_overlap = st.number_input("分块重叠 (chunk_overlap)", min_value=0, max_value=500, value=50, step=10)
    st.divider()
    st.caption("提示：上传文档后可直接提问相关内容，或启用搜索获取实时信息")

# 聊天历史显示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入处理
if prompt := st.chat_input("输入您的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 初始化工具列表
    tools = []

    # 1. 文档检索工具
    if st.session_state.vector_store:
        doc_retriever = st.session_state.vector_store.as_retriever()
        def document_search_func(q):
            docs = doc_retriever.get_relevant_documents(q)
            st.session_state.last_references = [d.metadata for d in docs]
            return "\n\n".join([d.page_content for d in docs])
        tools.append(Tool(
            name="document_search",
            func=document_search_func,
            description="使用上传的PDF/CSV文档回答问题"
        ))

    # 2. Tavily搜索工具
    if search_enabled:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            tavily_tool = TavilySearch(api_key=tavily_api_key, max_results=5)
            tools.append(Tool(
                name="web_search",
                func=tavily_tool.run,
                description="使用Tavily搜索引擎获取实时信息"
            ))
        else:
            st.warning("未设置Tavily API密钥")

    # 无可用工具时的处理
    if not tools:
        response = "请至少上传文档或启用搜索功能"
    else:
        try:
            # 创建Agent
            llm = ChatOpenAI(
                model="deepseek-chat",
                temperature=0.7,
                openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                openai_api_base="https://api.deepseek.com/v1",
                streaming=True,
            )

            # Agent提示模板
            # Agent提示模板部分需要修改为：
            template = """你是一个专业的助手。使用提供的工具回答问题。

            你拥有以下工具：
            {tools}{tool_names}

            使用以下格式：

            Question: 需要回答的输入问题
            Thought: 思考下一步该做什么
            Action: 要执行的动作，优先使用document_search，如果内容与document_search无关，则使用web_search
            Action Input: 动作的输入内容
            Observation: 动作的结果
            ... (这个思考/动作/输入/观察的过程可以重复多次)
            Thought: 我现在知道最终答案了
            Final Answer: 用简洁的语言回答原始问题

            记住：
            1. 回答必须严格按照上述格式
            2. 所有中文内容都要用中文回答
            3. Final Answer必须简洁明了

            Question: {input}
            {agent_scratchpad}"""

            prompt_template = PromptTemplate.from_template(template)  # 使用from_template方法

            # 然后创建agent时的代码保持不变
            agent = create_react_agent(llm, tools, prompt_template)
            # 修改agent_executor的创建部分
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True  # 添加此参数
            )

            response = agent_executor.invoke({"input": prompt})["output"]
        except Exception as e:
            response = f"处理请求时出错: {str(e)}"

    # 显示AI响应
    with st.chat_message("assistant"):
        st.markdown(response)
        # 追加参考文档来源
        if st.session_state.last_references:
            st.markdown("\n---\n**参考文档来源：**")
            for ref in st.session_state.last_references:
                file_name = ref.get('file_name') or ref.get('source') or '未知来源'
                page = ref.get('page_number')
                if page:
                    st.markdown(f"- {file_name} (第{page}页)")
                else:
                    st.markdown(f"- {file_name}")
    st.session_state.messages.append({"role": "assistant", "content": response})