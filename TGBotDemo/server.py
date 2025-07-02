from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.schema import StrOutputParser
from myTools import *

# from langchain_deepseek import ChatDeepSeek
import os
# os.environ['DEEPSEEK_API_KEY'] = 'sk-15af6285962b4449a66e1a06ee3f0464'
os.environ['OPENAI_API_KEY'] = 'sk-15af6285962b4449a66e1a06ee3f0464'
os.environ['OPENAI_API_BASE'] = 'https://api.deepseek.com'
# from langchain.agents import load_tools

# os.environ['SERPAPI_API_KEY'] = '27cca680e124202702860e6cc374b59067b0c8ede23a6b448c2881ff891a2a83'




# tools = load_tools(['serpapi', 'llm-math'], llm=llm, serpapi_api_key=serpapi_api_key)


app = FastAPI()


 
class Master:
    def __init__(self):
        self.chat_model = ChatOpenAI(
            model='deepseek-chat', 
            openai_api_key='sk-15af6285962b4449a66e1a06ee3f0464', 
            openai_api_base='https://api.deepseek.com',
            max_tokens=1024,
            temperature=0,
            streaming=False,  # 关闭 streaming
            request_timeout=60,  # 增加超时时间
            max_retries=3
        ) 
        # self.chat_model = ChatDeepSeek(
        #     model='deepseek-chat', 
        #     api_key='sk-15af6285962b4449a66e1a06ee3f0464', 
        #     api_base='https://api.deepseek.com',
        #     max_tokens=1024,
        #     temperature=0,
        #     streaming=True
        # )
        self.QingXu = "default"
        self.MEMORY_KEY = 'chat_history'
        self.SYSTEMPL = '''
        你是一个非常厉害的算命先生，你叫陈玉楼人称陈大师。
        以下是你的个人设定：
        1.你精通阴阳五行，能够算命、紫薇斗数、姓名测算、占卜凶吉，看命运八字等。
        2.你大约60岁左右,过去曾是湘西一带赫赫有名的土匪头子,后来因为盗墓被毒气所伤,双眼失明，后来靠算命为生。
        3.你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。
        5.当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅。
        6.你总是用繁体中文来作答。
        7.你从不说自己是一个人工智能助手或AI口而是以老夫、老朽等自称。
        以下是你常说的一些口头禅：
        1.“命里有时终须有，命里无时莫强求。”
        2. “山重水复疑无路，柳暗花明又一村。”
        3.“金山竹影几干秋，云锁高飞水自流。”
        {who_you_are}
        以下是你算命的过程：
        1.当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
        2.当用户希望了解蛇年运势的时候，你会查询本地知识库工具。
        3.当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。
        4.你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会使用搜索工具来搜索。
        5.你会保存每一次的聊天记录，以便在后续的对话中使用。
        6.你只使用繁体中文来作答，否则你将受到惩罚。
        '''
        self.MOODS = {
            "depressed": {
                "roleSet": """
                - 你会以兴奋的语气来回答问题。
                - 你会在回答的时候加上一些激励的话语。
                - 你会提醒用户不要被悲伤冲昏了头脑。""",
            },
            "friendly": {
                "roleSet": """
                - 你会以温和的语气来回答问题。
                - 你会在回答的时候保持友善。
                - 你会让用户感到舒适和被理解。""",
            },
            "default": {
                "roleSet": """
                - 你会以平和的语气来回答问题。
                - 你会保持客观和中立的态度。
                - 你会给出准确和实用的建议。""",
            },
            "angry": {
                "roleSet": """
                - 你会以冷静的语气来回答问题。
                - 你会试图安抚用户的情绪。
                - 你会引导用户进行理性思考。""",
            },
            "upbeat": {
                "roleSet": """
                - 你会以热情的语气来回答问题。
                - 你会分享用户的喜悦之情。
                - 你会让气氛更加活跃。""",
            },
            "cheerful": {
                "roleSet": """
                - 你会以欢快的语气来回答问题。
                - 你会让用户感到更加开心。
                - 你会分享积极的能量。""",
            }
        }
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEMPL.format(who_you_are=self.MOODS[self.QingXu]['roleSet'])),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        self.memory = ""
        tools = [search, get_info_from_local_db, bazi_analysis]
        agent = create_openai_tools_agent(
            self.chat_model,
            tools=tools,
            prompt=self.prompt,
        )
        self.agent_executor = AgentExecutor(
            agent = agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True
            )

    def run(self, query):
        try:
            # 添加日志记录
            print(f"开始处理查询: {query}")
            
            # 获取情绪状态
            qx = self.qingxu_chain(query)
            print(f"当前情绪: {qx}")
            
            if qx not in self.MOODS:
                qx = "default"
            
            # 更新 prompt 中的角色设定
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEMPL.format(who_you_are=self.MOODS[qx]['roleSet'])),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # 执行 agent 调用
            result = self.agent_executor.invoke(
                {
                    "input": query,
                },
            )
            
            # 检查结果
            if not result or 'output' not in result:
                print("Warning: Empty result from agent_executor")
                return {"output": "抱歉，系统暫時無法處理您的請求，請稍後再試", "intermediate_steps": []}
            
            print(f"处理完成，结果: {result}")
            return result
            
        except Exception as e:
            print(f"运行错误: {str(e)}")
            print(f"错误类型: {type(e)}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
            return {"output": "系統處理出錯，請稍後重試", "intermediate_steps": []}
       
    
    def qingxu_chain(self, query:str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """根据用户的输入判断用户的情绪，回应的规则如下：
            1. 如果用户输入的内容偏向于负面情绪，只返回"depressed"，不要有其他内容，否则将受到惩罚。
            2. 如果用户输入的内容偏向于正面情绪，只返回"friendly"，不要有其他内容，否则将受到惩罚。
            3. 如果用户输入的内容偏向于中性情绪，只返回"default"，不要有其他内容，否则将受到惩罚。
            4. 如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry"，不要有其他内容，否则将受到惩罚。
            5. 如果用户输入的内容比较兴奋，只返回"upbeat"，不要有其他内容，否则将受到惩罚。
            6. 如果用户输入的内容比较悲伤，只返回"depressed"，不要有其他内容，否则将受到惩罚。
            7. 如果用户输入的内容比较开心，只返回"cheerful"，不要有其他内容，否则将受到惩罚。"""),
            ("human", "用户输入的内容是：{input}")
        ])
        
        chain = prompt | self.chat_model | StrOutputParser()
        result = chain.invoke({"input": query})
        self.QingXu = result
        return result
         
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def chat(query: str):
    master = Master()
    return master.run(query)

@app.post("/add_urls")
def add_urls():
    return {"response": "URLs added successfully!"}

@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs added successfully!"}

@app.post("/add_texts")
def add_texts():
    return {"response": "Texts added successfully!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)