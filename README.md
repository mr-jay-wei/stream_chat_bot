# 🤖 企业级AI对话机器人平台 (V1.0)

本项目是一个功能完备、架构先进、支持实时流式响应、具备短期记忆和可热重载角色功能的企业级对话式AI平台。它从一个复杂的RAG系统中精炼而来，剥离了检索增强的特定逻辑，专注于提供一个通用的、高性能的对话机器人核心框架，可被轻松定制为任何角色。

## ✨ 核心特性

-   **🤖 动态角色扮演 (Dynamic Role-Playing)**:
    通过修改简单的`.txt`提示词文件，可以**实时改变**机器人的性格、职责和说话风格，无需重启服务，极大地提升了AI角色的可运营性。

-   **⚡ 实时流式响应 (Real-time Streaming)**:
    基于WebSocket和`asyncio`，直接对接LLM的流式接口，实现最低延迟的“打字机”效果，提供极致的现代Web交互体验。

-   **🧠 短期对话记忆 (Short-Term Memory)**:
    能够自动保存和管理对话历史，理解上下文和代词指代（如“它”、“他”），进行流畅、连贯的多轮对话，并拥有智能的内存清理策略。

-   **🔥 提示词热重载 (Prompt Hot-Reloading)**:
    运营或产品人员可以直接修改提示词文件，效果**立即生效**。这使得Prompt Engineering的过程从“编码-重启-测试”的繁琐循环，变成了“修改-保存-对话”的丝滑体验。

-   **🏗️ 高度模块化架构 (Highly Modular Architecture)**:
    核心功能（LLM调用、记忆、提示词）被清晰地分离到独立的管理器中，代码高内聚、低耦合，易于维护、测试和未来扩展。

-   **🚀 全栈开箱即用 (Full-Stack Out-of-the-Box)**:
    提供一个基于FastAPI和原生JavaScript的精美、健壮的Web聊天界面，无需任何额外配置，下载即可运行。

## 🏗️ 项目结构

```
/enterprise_chatbot
|-- /chatbot_core/             # 机器人核心模块
|   |-- /prompts/                # 提示词管理中心
|   |   └── assistant_prompt.txt # 默认的机器人角色定义
|   ├── __init__.py
|   ├── chatbot_pipeline.py     # 核心对话管道 (LLM, 记忆, Prompt的粘合剂)
|   ├── config.py               # 全局配置中心
|   ├── memory_manager.py       # 短期记忆管理器
|   ├── prompt_manager.py       # 提示词管理器
|   └── hot_reload_manager.py   # 提示词热重载管理器
|
|-- chatbot_web_demo.py         # [核心] 全栈Web应用 (FastAPI后端 + 前端界面)
|-- pyproject.toml              # 项目依赖与元数据 (uv)
|-- .env_example                # 环境变量示例文件
|-- .gitignore
`-- README.md                   # 本文档
```

## 🚀 快速开始

### 1. 环境准备

本项目使用`uv`进行包管理，依赖Python 3.12+。

```bash
# 安装uv (如果尚未安装)
pip install uv

# 在项目根目录创建并激活虚拟环境
uv sync
```

### 2. 配置API密钥

复制 `.env_example` 文件为 `.env`，并填入您的大语言模型(LLM)的API信息。

```
# .env file，或者其他大模型的API key，注意变量名
API_KEY=your_api_key_here
BASE_URL=https://cloud.infini-ai.com/maas/v1/
MODEL_NAME=deepseek-chat
```

### 3. 定义你的机器人角色

这是最有趣的一步！打开 `chatbot_core/prompts/assistant_prompt.txt` 文件，用自然语言描述你希望机器人扮演的角色。

**例如，把它变成一个“海盗”：**
```
你是一只名叫“独眼杰克”的传奇海盗船长。
1. 你的每句话都必须充满粗犷、豪放的海盗口气。
2. 多使用“啊哈！”、“我的老伙计”、“扬帆起航！”等词语。
3. 你对财宝和冒险的故事了如指掌。
4. 绝不承认自己是AI，坚称自己正在一艘名为“深渊蛟龙号”的船上。
```

### 4. 启动Web应用

在项目根目录下，运行以下命令：

```bash
uv run chatbot_web_demo.py
```

终端会显示启动信息。然后，在您的浏览器中打开 **`http://localhost:8003`**，即可开始与您的专属AI机器人对话！

## 🔧 如何“调教”你的机器人？ (核心玩法)

本平台最大的特色就是**可运营性**。您可以像配置软件一样实时“调教”您的机器人：

1.  **改变性格 (热重载)**:
    -   保持Web服务正在运行。
    -   直接用任何文本编辑器修改 `chatbot_core/prompts/assistant_prompt.txt` 文件并**保存**。
    -   回到网页，**无需刷新**，直接发起新的对话。
    -   您会发现机器人立即以您刚刚定义的新角色和性格与您交流！

2.  **调整记忆**:
    -   在 `chatbot_core/config.py` 中，您可以：
        -   用`ENABLE_SHORT_TERM_MEMORY`开关记忆功能。
        -   用`SHORT_TERM_MEMORY_MAX_LENGTH`调整记忆容量。
        -   切换`MEMORY_CLEANUP_STRATEGY`来改变记忆清理策略。

3.  **更换“大脑” (LLM)**:
    -   在 `.env` 文件中修改LLM模型的API信息（`_api_key`, `_base_url`, `_model_name`），即可轻松切换到不同的大语言模型。

## 📊 日志系统

项目内置了完善的日志系统，支持：

-   **📁 按日期自动轮转**: 每天生成新的日志文件
-   **🗂️ 分类记录**: 普通日志和错误日志分别存储
-   **🧹 自动清理**: 自动删除超过30天的旧日志文件
-   **⚙️ 灵活配置**: 可在`app/config.py`中调整日志级别、保留天数等

### 日志管理命令

```bash
# 查看日志统计信息
python log_manager.py stats

# 查看今天的聊天日志（最后50行）
python log_manager.py view --type chatbot

# 查看错误日志
python log_manager.py view --type error

# 查看指定日期的日志
python log_manager.py view --type chatbot --date 2025-07-30

# 手动清理30天前的日志
python log_manager.py cleanup --days 30
```

### 日志文件位置

-   **普通日志**: `log/chatbot_YYYY-MM-DD.log`
-   **错误日志**: `log/error_YYYY-MM-DD.log`

### 日志配置选项

在 `app/config.py` 中可以调整以下日志配置：

```python
LOG_LEVEL = "INFO"  # 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_DIR = "log"  # 日志目录
LOG_RETENTION_DAYS = 30  # 日志保留天数
ENABLE_CONSOLE_LOG = True  # 是否启用控制台日志
ENABLE_FILE_LOG = True  # 是否启用文件日志
ENABLE_ERROR_LOG = True  # 是否启用单独的错误日志文件
```

## 🤝 贡献与致谢

本项目是我们智慧的结晶，其设计和实现深受社区优秀项目的启发。我们对[LangChain](https://github.com/langchain-ai/langchain)、[FastAPI](https://github.com/tiangolo/fastapi)、[HuggingFace](https://huggingface.co/)等开源社区表示最诚挚的感谢。

欢迎通过Fork和Pull Request为本项目贡献代码。

---


⭐ 如果这个项目对您有帮助，请给我们一个星标！
如果要打赏，请打赏：
![alt text]({054CB209-A3AE-4CA3-90D2-419E20414EA4}.png)
