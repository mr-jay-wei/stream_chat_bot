
# 🤖 企业级AI对话机器人平台 (V1.2)

本项目是一个功能完备、架构先进的企业级对话式AI平台。它采用FastAPI构建，支持实时流式响应、多用户会话管理、短期记忆，并拥有独特的提示词（Prompt）热重载功能。后端服务（PostgreSQL, Redis）通过Docker Compose进行管理，实现了开发环境的一键部署。

## ✨ 核心特性

- **🤖 动态角色扮演 (Dynamic Role-Playing)**:
  通过修改简单的`.txt`提示词文件，可以**实时改变**机器人的性格、职责和说话风格，无需重启服务，极大地提升了AI角色的可运营性。

- **⚡ 实时流式响应 (Real-time Streaming)**:
  基于WebSocket和`asyncio`，直接对接LLM的流式接口，实现最低延迟的“打字机”效果，提供极致的现代Web交互体验。

- **🔐 多用户与会话管理 (Multi-User & Session Management)**:
  内置完整的用户认证（JWT）、注册、登录系统。每个用户拥有独立的、可持久化的多轮对话会话，确保数据隔离与安全。

- **🔥 提示词热重载 (Prompt Hot-Reloading)**:
  运营或产品人员可以直接修改提示词文件，效果**立即生效**。这使得Prompt Engineering的过程从“编码-重启-测试”的繁琐循环，变成了“修改-保存-对话”的丝滑体验。

- **🏗️ 高度模块化架构 (Highly Modular Architecture)**:
  核心功能（用户认证、会话管理、LLM调用、提示词管理）被清晰地分离到独立的模块中，代码高内聚、低耦合，易于维护、测试和未来扩展。

- **🚀 全栈开箱即用 (Full-Stack Out-of-the-Box)**:
  提供一个基于FastAPI后端和原生JavaScript的精美、健壮的Web聊天界面，并使用Docker Compose管理数据库和缓存，实现真正的“一键启动”。

## 🏗️ 项目结构

```
stream_chat_bot/
├── app/
│   ├── core/
│   ├── prompts/
│   │   └── assistant_prompt.txt
│   ├── __init__.py
│   ├── api_routes.py
│   ├── auth.py
│   ├── chatbot_pipeline.py
│   ├── config.py
│   ├── database.py
│   ├── hot_reload_manager.py
│   ├── limiter.py
│   ├── logger_config.py
│   ├── main.py
│   ├── models.py
│   └── session_manager.py
├── frontend/
├── log/
├── scripts/
├── test/
│   └── test_port.py
├── .env_example
├── .gitignore
├── .python-version
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## 🚀 快速开始

### 1. 环境准备

- **Docker**: 确保你已经安装并启动了 [Docker Desktop](https://www.docker.com/products/docker-desktop/)。
- **Python**: 需要 Python 3.12+ 版本。
- **uv**: 本项目使用`uv`进行包管理。如果尚未安装，请运行 `pip install uv`。

### 2. 配置项目

首先，克隆本项目到你的本地。

```bash
# 复制环境变量文件
cp .env_example .env
```

然后，打开`.env`文件，填入你的配置信息。**至少需要填写LLM的`API_KEY`、`BASE_URL`和`MODEL_NAME`**。数据库和Redis的配置可以使用默认值。

```ini
# .env file
# LLM配置
API_KEY='your_llm_api_key_here'
BASE_URL="your_llm_base_url_here"
MODEL_NAME="your_model_name_here"

# 数据库配置 (可使用默认值)
DB_HOST=localhost
DB_PORT=5432
DB_USER=chatbot_user
DB_PASSWORD=052756
DB_NAME=chatbot_db

# Redis配置 (可使用默认值)
REDIS_HOST=localhost
REDIS_PORT=6379

# JWT密钥配置 (建议修改为一个复杂的随机字符串)
SECRET_KEY="a_very_secret_key_for_jwt"
```

### 3. 启动后端服务

这是最关键的一步。在项目根目录下，运行以下命令来启动PostgreSQL数据库和Redis缓存服务：

```bash
docker-compose up -d
```

- `d`参数表示在后台运行。你可以随时使用`docker-compose down`来停止并移除这些服务容器。
- 首次运行时，Docker会自动下载所需的镜像，请耐心等待。

### 4. 安装依赖并启动Web应用

打开一个新的终端窗口，确保仍处于项目根目录。

```bash
# 使用uv安装所有Python依赖
uv sync

# 启动FastAPI Web应用
uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 28501 --reload
```

终端会显示应用启动信息。现在，在你的浏览器中打开 **`http://localhost:28501`**，即可开始与你的专属AI机器人进行交互！

## 🔧 如何“调教”你的机器人？

本平台最大的特色就是**可运营性**。您可以像配置软件一样实时“调教”您的机器人：

1.  **改变性格 (热重载)**:
    - 保持Web服务正在运行。
    - 直接用任何文本编辑器修改 `app/prompts/assistant_prompt.txt` 文件并**保存**。
    - 回到网页，**无需刷新**，直接发起新的对话。
    - 你会发现机器人立即以你刚刚定义的新角色和性格与你交流！

2.  **调整配置**:
    - 在 `app/config.py` 文件中，你可以调整日志级别、是否开启短期记忆等核心配置。

3.  **更换“大脑” (LLM)**:
    - 在 `.env` 文件中修改LLM模型的API信息，然后重启Web应用即可。

## 🛠️ 实用工具脚本

项目在`scripts/`目录下提供了两个非常方便的命令行工具。

### 日志管理 (`log_manager.py`)

```bash
# 查看日志统计信息
uv run python scripts/log_manager.py stats

# 查看今天的聊天日志（最后50行）
uv run python scripts/log_manager.py view

# 查看今天的错误日志
uv run python scripts/log_manager.py view --type error

# 手动清理30天前的日志
uv run python scripts/log_manager.py cleanup --days 30
```

### 数据库查看器 (`view_database.py`)

这是一个为不熟悉数据库的开发者设计的交互式工具，可以让你轻松查看数据库中的内容。

```bash
uv run python scripts/view_database.py
```

运行后，你会进入一个菜单驱动的界面，可以查看所有表的信息、用户列表、会话和消息内容，甚至执行简单的`SELECT`查询。

## 🤝 贡献与致谢

本项目的设计和实现深受社区优秀项目的启发。我们对[LangChain](https://github.com/langchain-ai/langchain)、[FastAPI](https://github.com/tiangolo/fastapi)等开源社区表示最诚挚的感谢。

欢迎通过 Fork 和 Pull Request 为本项目贡献代码。

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！

---