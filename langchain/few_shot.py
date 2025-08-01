from langchain_core.prompts import PromptTemplate
example_prompt = PromptTemplate.from_template("问题：{question}\n{answer}")

examples = [
    {
        "question": "谁活得更长，穆罕默德·阿里还是艾伦·图灵？",
        "answer": """
是否需要后续问题：是的。
后续问题：穆罕默德·阿里去世时多大年纪？
中间答案：穆罕默德·阿里去世时74岁。
后续问题：艾伦·图灵去世时多大年纪？
中间答案：艾伦·图灵去世时41岁。
所以最终答案是：穆罕默德·阿里
""",
    },
    {
        "question": "克雷格斯列表的创始人是什么时候出生的？",
        "answer": """
是否需要后续问题：是的。
后续问题：克雷格斯列表的创始人是谁？
中间答案：克雷格斯列表的创始人是克雷格·纽马克。
后续问题：克雷格·纽马克是什么时候出生的？
中间答案：克雷格·纽马克于1952年12月6日出生。
所以最终答案是：1952年12月6日
""",
    },
    {
        "question": "乔治·华盛顿的外祖父是谁？",
        "answer": """
是否需要后续问题：是的。
后续问题：乔治·华盛顿的母亲是谁？
中间答案：乔治·华盛顿的母亲是玛丽·波尔·华盛顿。
后续问题：玛丽·波尔·华盛顿的父亲是谁？
中间答案：玛丽·波尔·华盛顿的父亲是约瑟夫·波尔。
所以最终答案是：约瑟夫·波尔
""",
    },
    {
        "question": "《大白鲨》和《皇家赌场》的导演都来自同一个国家吗？",
        "answer": """
是否需要后续问题：是的。
后续问题：《大白鲨》的导演是谁？
中间答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
后续问题：史蒂文·斯皮尔伯格来自哪个国家？
中间答案：美国。
后续问题：《皇家赌场》的导演是谁？
中间答案：《皇家赌场》的导演是马丁·坎贝尔。
后续问题：马丁·坎贝尔来自哪个国家？
中间答案：新西兰。
所以最终答案是：不是
""",
    },
]

print(example_prompt.invoke(examples[0]).to_string())
print("--------------------------------")
from langchain_core.prompts import FewShotPromptTemplate
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="问题：{input}",
    input_variables=["input"],
)
print(
    prompt.invoke({"input": "乔治·华盛顿的父亲是谁？"}).to_string()
)

print("--------------------------------")
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 这是可供选择的示例列表。
    examples,
    # 这是用于生成嵌入的嵌入类，用于衡量语义相似性。
    HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
    # 这是用于存储嵌入并进行相似性搜索的 VectorStore 类。
    Chroma,
    # 这是要生成的示例数量。
    k=1,
)
# 选择与输入最相似的示例。
question = "玛丽·波尔·华盛顿的父亲是谁？"
selected_examples = example_selector.select_examples({"question": question})
print(f"与输入最相似的示例: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")