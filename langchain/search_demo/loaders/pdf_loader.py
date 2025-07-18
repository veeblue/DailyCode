from langchain_core.documents import Document

from typing import List
import PyPDF2
from langchain.schema import Document
from typing import List
import os
from io import BytesIO

class MyPdfLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        """加载PDF文件并转换为Document对象"""
        try:
            documents = []

            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    # 提取页面文本
                    text = page.extract_text()

                    if text.strip():  # 只处理非空页面
                        metadata = {
                            "source": self.file_path,
                            "source_type": "pdf",
                            "page_number": page_num + 1,
                            "file_name": os.path.basename(self.file_path),
                            "total_pages": len(pdf_reader.pages)
                        }

                        documents.append(Document(
                            page_content=text,
                            metadata=metadata
                        ))

            return documents

        except Exception as e:
            raise Exception(f"加载PDF文件失败: {str(e)}")
if __name__ == '__main__':
    # Example usage
    loader = MyPdfLoader("/Users/yee/vscode/study/test_demo/search_demo/data/deepseek-v2-tech-report.pdf")
    documents = loader.load()
    for doc in documents:
        print(f"doc.page_content -> {doc.page_content}...")  # Print first 100 characters
        print(f"doc.metadata -> {doc.metadata}")
        print("-------------------------------------------")

