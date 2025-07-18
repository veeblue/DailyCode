"""Web文档加载器"""
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from typing import List, Optional
from urllib.parse import urljoin, urlparse
import re
import time


class WebLoader:
    """网页文档加载器"""

    def __init__(self, url: str, timeout: int = 10):
        self.url = url
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def load(self) -> List[Document]:
        """加载单个网页"""
        try:
            response = requests.get(self.url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # 移除脚本和样式标签
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # 提取标题
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "无标题"

            # 提取主要内容
            content = self._extract_main_content(soup)

            # 清理文本
            content = self._clean_text(content)

            if not content.strip():
                raise Exception("未能提取到有效内容")

            metadata = {
                "source": self.url,
                "source_type": "web",
                "title": title_text,
                "url": self.url,
                "domain": urlparse(self.url).netloc,
                "content_length": len(content)
            }

            # 尝试提取更多元数据
            meta_description = soup.find('meta', attrs={'name': 'description'})
            if meta_description:
                metadata['description'] = meta_description.get('content', '')

            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                metadata['keywords'] = meta_keywords.get('content', '')

            return [Document(page_content=content, metadata=metadata)]

        except Exception as e:
            raise Exception(f"加载网页失败 ({self.url}): {str(e)}")

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """提取主要内容"""
        # 尝试找到主要内容区域
        main_selectors = [
            'main', 'article', '[role="main"]',
            '.content', '.main-content', '.post-content',
            '#content', '#main-content', '#post-content'
        ]

        for selector in main_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text()

        # 如果没找到主要内容区域，提取body内容
        body = soup.find('body')
        if body:
            return body.get_text()

        # 最后备选方案
        return soup.get_text()

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除多余的换行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def load_multiple(self, urls: List[str], delay: float = 1.0) -> List[Document]:
        """加载多个网页"""
        documents = []

        for i, url in enumerate(urls):
            try:
                loader = WebLoader(url, self.timeout)
                docs = loader.load()
                documents.extend(docs)

                # 添加延迟避免过于频繁的请求
                if i < len(urls) - 1:
                    time.sleep(delay)

            except Exception as e:
                print(f"跳过URL {url}: {str(e)}")
                continue

        return documents

    def get_page_info(self) -> dict:
        """获取网页基本信息"""
        try:
            response = requests.get(self.url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # 提取基本信息
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "无标题"

            info = {
                "url": self.url,
                "title": title_text,
                "domain": urlparse(self.url).netloc,
                "status_code": response.status_code,
                "content_type": response.headers.get('content-type', ''),
                "content_length": len(response.content)
            }

            # 提取meta信息
            meta_description = soup.find('meta', attrs={'name': 'description'})
            if meta_description:
                info['description'] = meta_description.get('content', '')

            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                info['keywords'] = meta_keywords.get('content', '')

            # 统计链接数量
            links = soup.find_all('a', href=True)
            info['link_count'] = len(links)

            # 统计图片数量
            images = soup.find_all('img', src=True)
            info['image_count'] = len(images)

            return info

        except Exception as e:
            raise Exception(f"获取网页信息失败: {str(e)}")

    def extract_links(self, base_url: Optional[str] = None) -> List[str]:
        """提取网页中的所有链接"""
        try:
            response = requests.get(self.url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            links = []

            for link in soup.find_all('a', href=True):
                href = link['href']
                # 转换相对链接为绝对链接
                if base_url:
                    href = urljoin(base_url, href)
                elif not href.startswith('http'):
                    href = urljoin(self.url, href)

                links.append(href)

            return list(set(links))  # 去重

        except Exception as e:
            raise Exception(f"提取链接失败: {str(e)}")

if __name__ == '__main__':

    from langchain.schema import Document
    from langchain_tavily import TavilySearch
    import os

    search = TavilySearch(max_results=1, api_key=os.getenv("TAVILY_API_KEY"))

    results = search.invoke("介绍一下LangChain的核心模块")

    # 转换为 Document 对象（适配 RAG）
    docs = [
        Document(
            page_content=result['content'],
            metadata={"source": result['url'], "title": result['title']}
        )
        for result in results
    ]

    # 你可以用于 vectorstore 或直接显示
    print(docs)
