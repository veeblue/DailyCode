from langchain_core.documents import Document
import pandas as pd

from typing import List
import os

class MyCSVLoader:
    """ CSV File Loader."""

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Load CSV file and convert it to a list of Document objects."""
        try:
            df = pd.read_csv(self.file_path, encoding=self.encoding)
            documents = []

            for index, row in df.iterrows():
                content_parts = []

                for column, value in row.items():
                    if pd.notna(value):
                        content_parts.append(f"{column}: {value}")
                content = "\n".join(content_parts)

                metadata = {
                    "source": self.file_path,
                    "source_type": "csv",
                    "row_index": index,
                    "file_name": os.path.basename(self.file_path)
                }
                # 添加每列作为元数据
                for column, value in row.items():
                    if pd.notna(value):
                        metadata[f"field_{column}"] = str(value)

                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))

            return documents
        except Exception as e:
            raise Exception(f"加载CSV文件失败: {str(e)}")

if __name__ == '__main__':
    # Example usage
    loader = MyCSVLoader("/Users/yee/vscode/study/test_demo/search_demo/data/brand_category.csv")
    documents = loader.load()
    for doc in documents:
        print(f"doc.page_content -> {doc.page_content}")
        print(f"doc.metadata -> {doc.metadata}")
        print("-------------------------------------------")