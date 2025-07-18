# LangGraph聊天机器人实现笔记

## 📋 项目概述

这是一个基于LangGraph的智能聊天机器人，具有时间旅行功能、人工协助机制和工具集成能力。该项目展示了如何使用LangGraph构建复杂的对话系统。

## 🏗️ 系统架构

### 核心组件

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    用户输入      │───▶│   聊天机器人     │───▶│    工具节点     │
│   (HumanMessage)│    │   (chatbot)     │    │  (tool_node)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   路由决策      │    │   工具执行      │
                       │ (route_tool)    │    │   结果返回      │
                       └─────────────────┘    └─────────────────┘
```

### 状态管理

```python
class State(TypedDict):
    messages: Annotated[List[dict], add_messages]
    name: str      # 用户名称
    birthday: str  # 用户生日
```

## 🛠️ 工具集成

### 1. 搜索工具
- **工具**: `TavilySearch`
- **功能**: 网络搜索能力
- **配置**: 最大结果数为2

### 2. 人工协助工具
```python
@tool
def human_assistance(name: str, birthday: str, tool_call_id: str) -> str:
    """请求人工协助的工具"""
    # 实现人工验证和修正功能
```

**特点**:
- 支持信息验证
- JSON格式输入/输出
- 状态更新机制

### 3. 时间旅行工具

#### 历史查看
```python
@tool
def show_history(tool_call_id: str) -> str:
    """显示对话历史状态快照"""
```

#### 检查点选择
```python
@tool
def select_checkpoint(step_number: int, tool_call_id: str) -> str:
    """选择要回退到的检查点"""
```

**功能**:
- 查看完整对话历史
- 选择特定检查点
- 从检查点恢复对话

## 🔄 工作流程

### 1. 消息处理流程
```
START ──▶ chatbot ──▶ route_tool ──┐
           ▲                       │
           │                       ▼
        tools ◀─────────────── [判断是否需要工具]
           │                       │
           ▼                       ▼
      [工具执行]                  END
```

### 2. 路由决策
```python
def route_tool(state: State):
    """路由决策逻辑"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"  # 需要调用工具
    return "end"        # 直接结束
```

## 💾 持久化存储

### 内存检查点
```python
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

### 配置管理
```python
config = {"configurable": {"thread_id": "1"}}
```

## 🎮 交互式命令

### 基本命令
| 命令 | 简写 | 功能 |
|------|------|------|
| `history` | `h` | 显示对话历史 |
| `select <数字>` | `s <数字>` | 选择检查点 |
| `continue` | `c` | 从检查点继续 |
| `exit/quit` | `q` | 退出程序 |
| `help` | - | 显示帮助信息 |

### 使用示例
```bash
User: 我的名字是张三，生日是1990-01-01
AI: 你好张三！很高兴认识你...

User: history
对话历史状态 (按时间顺序):
步骤 0: 14:30:15 (ID: abc12345...) - 名称: 张三, 生日: 1990-01-01

User: select 0
已选择检查点 0 (ID: abc12345...) - 名称: 张三, 生日: 1990-01-01

User: continue
从检查点继续对话...
```

## 🔧 配置说明

### 环境变量
```python
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["DEEPSEEK_API_KEY"] = "your_api_key_here"
```

### LLM配置
```python
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1"
)
```

## 🚀 核心特性

### 1. 状态管理
- ✅ 对话历史持久化
- ✅ 用户信息跟踪
- ✅ 检查点机制

### 2. 工具集成
- ✅ 网络搜索
- ✅ 人工协助
- ✅ 时间旅行

### 3. 交互体验
- ✅ 流式输出
- ✅ 命令行界面
- ✅ 错误处理

## 📊 数据流

### 输入处理
```
用户输入 → HumanMessage → State.messages → LLM处理 → AIMessage
```

### 工具调用
```
AIMessage.tool_calls → 工具执行 → ToolMessage → State更新
```

### 状态更新
```python
state_update = {
    "name": verified_name,
    "birthday": verified_birthday,
    "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
}
```

## 🐛 错误处理

### 异常捕获
```python
try:
    # 核心逻辑
except Exception as e:
    print(f"错误: {e}")
    continue
```

### 输入验证
```python
try:
    human_response = json.loads(raw_input)
except json.JSONDecodeError:
    print("输入格式错误，请输入 JSON 格式的字符串。")
    return "Invalid input."
```

## 🔮 扩展建议

### 1. 数据库集成
```python
# 替换MemorySaver为持久化存储
from langgraph.checkpoint.postgres import PostgresSaver
# 或
from langgraph.checkpoint.sqlite import SqliteSaver
```

### 2. 更多工具
- 文件操作工具
- 邮件发送工具
- 数据分析工具

### 3. 增强功能
- 用户认证
- 权限管理
- 审计日志

## 📈 性能优化

### 1. 内存管理
- 定期清理历史记录
- 限制状态大小

### 2. 并发处理
- 异步工具调用
- 线程池管理

### 3. 缓存机制
- 搜索结果缓存
- 状态快照缓存

## 🧪 测试建议

### 单元测试
```python
def test_chatbot_node():
    """测试聊天机器人节点"""
    pass

def test_tool_routing():
    """测试工具路由逻辑"""
    pass
```

### 集成测试
```python
def test_full_workflow():
    """测试完整工作流程"""
    pass
```

## 📚 参考资源

- [LangGraph官方文档](https://langchain-ai.github.io/langgraph/)
- [LangChain工具集成](https://python.langchain.com/docs/integrations/tools/)
- [OpenAI API文档](https://platform.openai.com/docs/api-reference)


## 📄 许可证

该项目采用MIT许可证。

---

**注意**: 这是一个学习示例项目，在生产环境中使用前请进行充分的测试和安全审查。