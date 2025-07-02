import logging
import re
from dataclasses import dataclass
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from pymilvus import connections, utility, MilvusClient, exceptions, FieldSchema, CollectionSchema, DataType, Collection
from tqdm import tqdm

# 推荐用 langchain_huggingface 的新实现
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    logging.warning("建议 pip install -U langchain-huggingface 并用新版 HuggingFaceEmbeddings")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CleanedDocument:
    page_content: str
    page_num: Optional[int] = None

def clean_page_content(text: str) -> str:
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

def load_and_clean_pdf(file_path: str) -> List[CleanedDocument]:
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()
    cleaned_documents = []
    logger.info(f"开始清洗 {len(raw_documents)} 页 PDF ...")
    for i, doc in enumerate(tqdm(raw_documents, desc="清洗PDF")):
        cleaned_text = clean_page_content(doc.page_content)
        cleaned_documents.append(CleanedDocument(page_content=cleaned_text, page_num=i+1))
    logger.info(f"完成清洗，共 {len(cleaned_documents)} 页")
    return cleaned_documents

def split_documents(documents: List[CleanedDocument]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", "！", "？"],
        add_start_index=True
    )
    from langchain.schema import Document as LCDocument
    lc_docs = [
        LCDocument(
            page_content=doc.page_content,
            metadata={
                "page_num": doc.page_num,
                "content": doc.page_content  # 把原文内容也放到 metadata
            }
        )
        for doc in documents
    ]
    logger.info("开始分块 ...")
    chunks = text_splitter.split_documents(lc_docs)
    logger.info(f"分块完成，共 {len(chunks)} 个片段")
    return chunks

def ensure_clean_collection(collection_name: str, uri: str, token: str):
    client = MilvusClient(
        uri=uri,
        token=token
    )
    logger.info(f"✓ 已连接 Milvus 接口")

    try:
        client.create_database(db_name="my_database_1")
        print("✓ my_database_1 创建成功")
    except exceptions.MilvusException as e:
        if "already exist" in str(e).lower():
            print("ℹ my_database_1 已存在")
        else:
            raise

    # ✅ 构建 schema（不要混用 pymilvus）
    schema = client.create_schema(enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=512)
    schema.add_field(field_name="page_num", datatype=DataType.INT64)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)  # 增大最大长度

    # ✅ 用 MilvusClient 判断 collection 存在
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
        logger.info(f"已删除旧的 collection: {collection_name}")

    client.create_collection(collection_name=collection_name, schema=schema)
    logger.info(f"已创建 collection: {collection_name}")

    info = client.describe_collection(collection_name=collection_name)
    logger.info(f"Collection详情：{info}")

def main():
    try:
        pdf_path = "/Users/yee/vscode/study/simple_rag_demo/document/doc.pdf"
        collection_name = "rag_demo_collection"
        milvus_uri = "http://localhost:19530"
        milvus_token = "root:Milvus"

        # 1. 清理并创建 collection（带 schema）
        ensure_clean_collection(collection_name, milvus_uri, milvus_token)

        # 2. 加载并清洗 PDF
        documents = load_and_clean_pdf(pdf_path)
        print(f"原始页面数：{len(documents)}")

        # 3. 分块
        chunks = split_documents(documents)

        # 4. 嵌入模型
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "mps"}
        )

        # 5. 存入 Milvus（直接用已存在的 collection）
        logger.info("开始写入 Milvus ...")
        vectorstore = Milvus.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=collection_name,
            connection_args={"uri": milvus_uri, "token": milvus_token},
            consistency_level="Strong"
        )
        print(f"✓ 已成功将 {len(chunks)} 个文档片段嵌入并存入 Milvus。")
    except Exception as e:
        logger.error(f"主流程发生异常: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()