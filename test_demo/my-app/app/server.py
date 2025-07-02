from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_openai import ChatOpenAI
import os
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
app = FastAPI(
    title="My App",
    description="This is a test app",
    version="0.1.0",
    openapi_url=None
)

try:
    add_routes(
        app,
        ChatDeepSeek(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base="https://api.deepseek.com/v1"
        ),
        path="/deepseek"
    )
except Exception as e:
    print("❌ LLM 加载失败：", e)

prompt = ChatPromptTemplate.from_template("告诉我一个关于 {topic} 的笑话")
add_routes(
    app,
    prompt |    ChatDeepSeek(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base="https://api.deepseek.com/v1"
        ),
    path="/txt",
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# 设置所有启用 CORS 的来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
