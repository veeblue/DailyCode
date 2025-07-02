# PDF Processing Configuration
PDF_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 100,
    "min_line_length": 5,
    "separators": ["\n\n", "\n", "。", ".", "！", "？"]
}

# Milvus Configuration
MILVUS_CONFIG = {
    "uri": "http://localhost:19530",
    "token": "root:Milvus",
    "collection_name": "rag_demo_collection",
    "consistency_level": "Strong"
}

# Embedding Model Configuration
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-small-zh-v1.5",
    "device": "mps"
}

# File Paths
FILE_PATHS = {
    "pdf_path": "/Users/yee/vscode/study/simple_rag_demo/document/test.pdf"
} 