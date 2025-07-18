from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from langchain.vectorstores import Milvus
from pymilvus import connections
print("--------------- 文档加载并清洗 -----------------")

def clean_page_content(text):
    """
    清洗 PDF 页面内容：去除页眉页脚、空行等
    可按需添加其他清洗规则
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        # 跳过页码、卷号、页眉等典型格式
        if re.match(r'^第?\d+页$', line): continue
        if re.search(r'第\d+卷|Vol|No|计算机科学', line): continue
        if len(line) < 5: continue  # 忽略太短的行
        cleaned.append(line)
    return "\n".join(cleaned)

def restore_english_spacing(text):
    """
    检测英文长句中没有空格的部分，尝试加回空格。
    使用粗略词典规则+空格预测
    """
    # 简单的处理方式：找出全是小写字母的长单词串进行分词（用 spacymodel 效果更好）
    from wordninja import split  # pip install wordninja
    tokens = []
    for word in text.split():
        if re.match(r"^[a-z]{10,}$", word):  # 10个以上连续小写字母，可能是英文句子粘连
            tokens += split(word)
        else:
            tokens.append(word)
    return ' '.join(tokens)

# 初始化 PDF 加载器
def load_and_clean_pdf(file_path):
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()
    # 清洗每页的内容
    cleaned_documents = []
    for doc in raw_documents:
        doc.page_content = clean_page_content(doc.page_content)
        # doc.page_content = restore_english_spacing(doc.page_content)
        cleaned_documents.append(doc)
    return cleaned_documents

pdf_path = "/Users/yee/vscode/study/simple_rag_demo/document/test.pdf"
documents = load_and_clean_pdf(pdf_path)

print(f"原始页面数：{len(documents)}")

print("--------------- 语义切割 -----------------")

# 使用语义优先级的分隔符切割：更自然的段落结构
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", ".", "！", "？"],
    add_start_index=True
)

chunks = text_splitter.split_documents(documents)

# print(f"分割后文档数量: {len(chunks)}")
# for i, doc in enumerate(chunks):  # 只展示前3段内容示例
#     print(f"\n--- 分割段 {i} ---")
#     print(doc.page_content)  # 截取前200字预览

print("# ========= 3. 嵌入模型（本地 HuggingFace 模型）=========")

from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "mps"}
)

# ========= 4. 连接 Milvus =========
connections.connect("default", uri="http://localhost:19530", token="root:Milvus")

# ========= 5. 向量存入 Milvus =========
collection_name = "rag_demo_collection"

vectorstore = Milvus.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name=collection_name,
    connection_args={"uri": "http://localhost:19530", "token": "root:Milvus"},
    consistency_level="Strong"
)
print(f"✓ 已成功将 {len(chunks)} 个文档片段嵌入并存入 Milvus。")