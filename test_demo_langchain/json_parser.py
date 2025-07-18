import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0.7, 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base="https://api.deepseek.com/v1",
    )

class Joke(BaseModel):
    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解决笑话的答案"
    )

joke_query = "告诉我一个笑话。"
parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
# print(parser.get_format_instructions())
chain = prompt | llm | parser
response = chain.invoke({"query": joke_query})
print(response)
