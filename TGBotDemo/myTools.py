from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import  Qdrant
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import requests
serpapi_api_key = '27cca680e124202702860e6cc374b59067b0c8ede23a6b448c2881ff891a2a83'

@tool
def test():
    '''Test tool'''
    return "Hello"

@tool
def search(query:str):
    '''当需要搜索的时候，使用这个工具''' 
    serpapi = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    result = serpapi.run(query)
    print("实时信息：", result)
    return result

@tool
def get_info_from_local_db(query:str):
    '''只有当回答本年运势的时候，才使用这个工具'''
    client = Qdrant(
        QdrantClient( path="/local_qdrant_db" ),
        "local_documents",
        OpenAIEmbeddings()
    )
    retriever = client.as_retriever(search_type="mmr")
    result = retriever.get_relevant_documents(query)
    return result


@tool
def bazi_analysis(query: str):
    '''只有做八字排盘的时候才使用这个工具，需要姓名和出生年月日，缺一不可！'''
    url = "https://api.yuanfenju.com/index.php/v1/Bazi/cesuan"
    
    try:
        print(f"开始八字分析，查询内容: {query}")
        
        # Create a prompt template to extract parameters
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一个参数助手，根据用户的输入的内容找出相关的参数并按照json的格式返回。 
            json字段如下：
            "api_key": "vKNMX1DOSUZ4ZXowLOAtR9aaM",
            "name": "名字 例如 张三",
            "sex": "1",
            "type": "1",
            "year": "1990",
            "month": "9",
            "day": "9",
            "hours": "9",
            "minute": "00"
            
            如果没有找到相关的参数，告诉用户缺少哪些参数。
            只返回数据结构，不要做其他评论。
            """)
        ])
        
        parser = JsonOutputParser()
        chain = prompt | ChatOpenAI(
                model='deepseek-chat', 
                openai_api_key='sk-15af6285962b4449a66e1a06ee3f0464', 
                openai_api_base='https://api.deepseek.com',
                temperature=0,
            )  | parser
        
        # Invoke the chain to extract parameters
        data = chain.invoke({"input": query})
        print(f"八字查询参数: {data}")
        
        if not data:
            print("Error: No data returned from parameter extraction")
            return "參數解析失敗，請確保提供完整的姓名和出生年月日時分信息"
        
        # Send request to the API
        print("发送API请求...")
        result = requests.post(url, data=data)
        print(f"API响应状态码: {result.status_code}")
        print(f"API响应内容: {result.text}")
        
        if result.status_code == 200:
            try:
                json_result = result.json()
                if not json_result or 'data' not in json_result or 'bazi_info' not in json_result['data']:
                    print("Error: Invalid API response structure")
                    return "八字分析結果格式異常，請稍後重試"
                
                resultStr = f"八字為：{json_result['data']['bazi_info']['bazi']}"
                return resultStr
            except Exception as e:
                print(f"Error parsing result: {e}")
                return "八字分析失敗，解析結果時出錯"
        else:
            print(f"API error: {result.text}")
            return "系統處理出錯，請稍後重試"
    
    except Exception as e:
        print(f"Exception in bazi_analysis: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        return "系統處理出錯，請稍後重試"