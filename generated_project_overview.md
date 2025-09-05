# 项目概览: stream_chat_bot

本文档由`generate_project_overview.py`自动生成，包含了项目的结构树和所有可读文件的内容。

## 项目结构

```
stream_chat_bot/
├── app
│   ├── core
│   ├── prompts
│   │   └── assistant_prompt.txt
│   ├── __init__.py
│   ├── api_routes.py
│   ├── auth.py
│   ├── chatbot_pipeline.py
│   ├── config.py
│   ├── database.py
│   ├── hot_reload_manager.py
│   ├── logger_config.py
│   ├── models.py
│   ├── prompt_manager.py
│   └── session_manager.py
├── log
├── static
│   ├── index.html
│   ├── main.js
│   └── style.css
├── test
├── .env_example
├── .gitignore
├── .python-version
├── chatbot_web_demo.py
├── log_manager.py
├── pyproject.toml
├── README.md
├── stream_chat_bot.db
└── view_database.py
```

---

# 文件内容

## `.env_example`

```
CLOUD_INFINI_API_KEY = ""
CLOUD_BASE_URL = ""
CLOUD_MODEL_NAME = ""
DeepSeek_api_key = ""
DeepSeek_base_url = ""
DeepSeek_model_name = ""
```

## `.gitignore`

```
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info
.vscode
.env
# Virtual environments
.venv

```

## `.python-version`

```
3.12

```

## `app/__init__.py`

[文件为空]

## `app/api_routes.py`

```python
# app/api_routes.py

from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from .models import User

from .database import get_db
from .auth import (
    authenticate_user, create_user, create_access_token, 
    verify_token, ACCESS_TOKEN_EXPIRE_MINUTES
)
# from .user_service import UserService  # 已移动到test文件夹
from .logger_config import get_logger

logger = get_logger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)

# 请求模型
class UserRegister(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_email: str

class UserInfo(BaseModel):
    id: int
    email: str
    is_active: bool

# 依赖函数：获取当前用户
def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token: Optional[str] = Cookie(None, alias="access_token"),
    db: Session = Depends(get_db)
):
    """获取当前认证用户"""
    # 优先使用Authorization header中的token
    auth_token = None
    if credentials:
        auth_token = credentials.credentials
    elif token:
        auth_token = token
    
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    email = verify_token(auth_token)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    from .auth import get_user_by_email
    user = get_user_by_email(db, email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在"
        )
    
    return user

@router.post("/register", response_model=TokenResponse)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """用户注册"""
    try:
        # 创建用户
        user = create_user(db, user_data.email, user_data.password)
        
        # 创建访问令牌
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        
        logger.info(f"用户注册成功: {user.email}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user_email=user.email
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户注册失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注册失败，请稍后重试"
        )

@router.post("/login", response_model=TokenResponse)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """用户登录"""
    try:
        # 验证用户
        user = authenticate_user(db, user_data.email, user_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="邮箱或密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 创建访问令牌
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        
        logger.info(f"用户登录成功: {user.email}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user_email=user.email
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户登录失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录失败，请稍后重试"
        )

@router.get("/me", response_model=UserInfo)
def get_current_user_info(current_user = Depends(get_current_user)):
    """获取当前用户信息"""
    return UserInfo(
        id=current_user.id,
        email=current_user.email,
        is_active=current_user.is_active
    )

@router.post("/logout")
def logout():
    """用户登出（客户端需要删除token）"""
    return {"message": "登出成功"}

# 新的会话管理API
@router.get("/chat-sessions")
def get_user_chat_sessions(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取用户的会话列表"""
    try:
        from .session_manager import session_manager
        sessions = session_manager.get_user_sessions(db, current_user.id)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"获取用户会话失败: {e}")
        return {"sessions": []}

@router.get("/chat-sessions/{session_id}/messages")
def get_session_messages(
    session_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取指定会话的所有消息"""
    try:
        from .session_manager import session_manager
        messages = session_manager.get_session_messages(db, session_id, current_user.id)
        return {"messages": messages}
    except Exception as e:
        logger.error(f"获取会话消息失败: {e}")
        return {"messages": []}

# 保留旧的API以兼容现有代码
@router.get("/conversations")
def get_user_conversations(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取用户的聊天记录列表（兼容旧版本）"""
    try:
        from .session_manager import session_manager
        
        # 优先返回新的会话数据
        sessions = session_manager.get_user_sessions(db, current_user.id)
        if sessions:
            # 转换为旧格式以兼容前端
            conversations = []
            for session in sessions:
                conversations.append({
                    "id": session["id"],
                    "title": session["title"],
                    "preview": session["preview"],
                    "created_at": session["created_at"],
                    "session_type": "chat_session"  # 标记这是新的会话类型
                })
            return {"conversations": conversations}
        
        # 如果没有新会话，返回旧的对话记录
        legacy_conversations = session_manager.get_legacy_conversations(db, current_user.id)
        return {"conversations": legacy_conversations}
        
    except Exception as e:
        logger.error(f"获取用户对话记录失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取对话记录失败"
        )

@router.delete("/chat-sessions/{session_id}")
def delete_chat_session(
    session_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """删除指定的对话会话"""
    try:
        # 使用新的会话管理系统进行删除
        from .session_manager import session_manager
        
        # 尝试删除新的会话
        success = session_manager.delete_session(db, current_user.id, session_id)
        
        if not success:
            # 如果新会话删除失败，尝试删除旧的对话记录（兼容性）
            from .models import Conversation
            conversation = db.query(Conversation).filter(
                Conversation.id == session_id,
                Conversation.user_id == current_user.id
            ).first()
            
            if conversation:
                db.delete(conversation)
                db.commit()
                success = True
                logger.info(f"用户 {current_user.email} 删除旧对话记录 {session_id}")
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="对话记录不存在或不属于当前用户"
                )
        else:
            logger.info(f"用户 {current_user.email} 删除会话 {session_id}")
        
        return {"message": "对话会话删除成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除对话会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除对话会话失败"
        )

@router.delete("/messages/{message_id}")
def delete_message(
    message_id: int,
    current_user: User = Depends(get_current_user), # 明确类型提示
    db: Session = Depends(get_db)
):
    """删除指定的消息"""
    try:
        from .models import Message, ChatSession # 局部导入模型
        
        # 查询消息，并联结ChatSession以验证用户所有权
        message_to_delete = db.query(Message).join(ChatSession).filter(
            Message.id == message_id,
            ChatSession.user_id == current_user.id
        ).first()

        if message_to_delete:
            db.delete(message_to_delete)
            db.commit()
            logger.info(f"用户 {current_user.email} 成功删除消息 {message_id}")
            return {"message": "消息删除成功"}
        else:
            # 如果消息不存在，或不属于当前用户，则引发404错误
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="消息不存在或无权限删除"
            )
        
    except HTTPException:
        raise # 重新抛出已知的HTTP异常
    except Exception as e:
        logger.error(f"删除消息 {message_id} 时发生内部错误: {e}")
        db.rollback() # 发生未知错误时回滚
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除消息失败"
        )
```

## `app/auth.py`

```python
# app/auth.py

from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from .models import User
from .logger_config import get_logger

logger = get_logger(__name__)

# 密码加密配置
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT配置
import os
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """生成密码哈希"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """根据邮箱获取用户"""
    try:
        return db.query(User).filter(User.email == email).first()
    except Exception as e:
        logger.error(f"获取用户失败: {e}")
        return None

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """验证用户"""
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_user(db: Session, email: str, password: str) -> User:
    """创建新用户"""
    try:
        # 检查用户是否已存在
        existing_user = get_user_by_email(db, email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已被注册"
            )
        
        # 创建新用户
        hashed_password = get_password_hash(password)
        db_user = User(
            email=email,
            hashed_password=hashed_password
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"新用户注册成功: {email}")
        return db_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建用户失败: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="用户创建失败"
        )

def verify_token(token: str) -> Optional[str]:
    """验证JWT令牌并返回用户邮箱"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return email
    except JWTError:
        return None
```

## `app/chatbot_pipeline.py`

```python
# app/chatbot_pipeline.py

import asyncio
import time
import os
from typing import AsyncGenerator, Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

# 导入我们的模块化组件
from . import config
from .prompt_manager import prompt_manager
from .logger_config import get_logger

# 配置日志
logger = get_logger(__name__)

# 导入LangChain核心组件
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 导入流式事件定义
from dataclasses import dataclass
from enum import Enum

class StreamEventType(Enum):
    PROCESSING = "processing"
    GENERATION_START = "generation_start"
    GENERATION_CHUNK = "generation_chunk"
    GENERATION_END = "generation_end"
    ERROR = "error"
    COMPLETE = "complete"

@dataclass
class StreamEvent:
    type: StreamEventType
    data: Any
    timestamp: float

class ChatbotPipeline:
    """
    企业级对话机器人核心管道 (V1.1 - 支持热重载回调)
    """
    def __init__(self):
        logger.info("正在初始化企业级对话机器人...")
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self._setup_llm()
        logger.info("企业级对话机器人初始化完成。")

    def _setup_llm(self):
        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL")
        model_name = os.getenv("MODEL_NAME")

        if not all([api_key, base_url, model_name]):
            logger.error("API密钥或模型配置未找到。请检查.env文件。")
            raise ValueError("API密钥或模型配置未找到。请检查.env文件。")
        
        logger.info(f"配置大语言模型: {model_name}")
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.001,
            streaming=True,
            max_tokens=4000,
            request_timeout=60
        )
            
    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def ask_stream(self, question: str, db, user_id: int, session_id: Optional[int] = None) -> AsyncGenerator[StreamEvent, None]:
        """
        核心的流式对话方法 - 支持用户数据隔离和会话管理
        """
        try:
            yield StreamEvent(type=StreamEventType.PROCESSING, data={"message": "思考中..."}, timestamp=time.time())

            from .session_manager import session_manager
            
            current_session_id = session_id
            user_message_id = None

            if current_session_id is None:
                current_session_id = session_manager.create_new_session(db, user_id, question)
                if current_session_id:
                    yield StreamEvent(type=StreamEventType.PROCESSING, data={"message": "开始新对话", "session_id": current_session_id}, timestamp=time.time())
                else:
                    raise Exception("创建新会话失败")
            else:
                yield StreamEvent(type=StreamEventType.PROCESSING, data={"message": "继续对话", "session_id": current_session_id}, timestamp=time.time())

            # [FIX] 在调用AI之前，立刻保存用户的提问
            if current_session_id:
                user_message_id = session_manager.add_message_to_session(db, current_session_id, "user", question)
                if user_message_id:
                    logger.info(f"用户提问已保存到会话 {current_session_id}: 消息ID {user_message_id}")
                else:
                    logger.error(f"保存用户提问失败: 会话 {current_session_id}")

            system_prompt_template = prompt_manager.get_template(config.SYSTEM_PROMPT_NAME)
            system_message_content = system_prompt_template.format()

            chat_history = []
            if config.ENABLE_SHORT_TERM_MEMORY and current_session_id:
                try:
                    # [FIX] 在保存完用户问题后获取历史，确保上下文包含当前问题
                    session_messages = session_manager.get_session_context_for_ai(db, current_session_id, user_id, max_messages=10)
                    for msg in session_messages:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history.append(AIMessage(content=msg["content"]))
                    logger.debug(f"加载了会话 {current_session_id} 的 {len(session_messages)} 条历史消息")
                except Exception as e:
                    logger.error(f"获取会话历史失败: {e}")
            
            messages = [SystemMessage(content=system_message_content)]
            messages.extend(chat_history)

            yield StreamEvent(type=StreamEventType.GENERATION_START, data={"message": "开始生成回答"}, timestamp=time.time())

            complete_answer = ""
            api_key = os.getenv("API_KEY", "")
            if not api_key or api_key.endswith(".com") or "invalid" in api_key.lower():
                logger.warning("API密钥无效，使用模拟回复进行测试")
                mock_response = f"这是一个模拟回复。你的问题是：{question}。"
                for char in mock_response:
                    complete_answer += char
                    yield StreamEvent(type=StreamEventType.GENERATION_CHUNK, data={"chunk": char}, timestamp=time.time())
                    await asyncio.sleep(0.02)
            else:
                try:
                    if hasattr(self.llm, 'astream'):
                        async for chunk in self.llm.astream(messages):
                            chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                            if chunk_content:
                                complete_answer += chunk_content
                                yield StreamEvent(type=StreamEventType.GENERATION_CHUNK, data={"chunk": chunk_content}, timestamp=time.time())
                    else:
                        response = await self._run_in_executor(self.llm.invoke, messages)
                        answer = response.content if hasattr(response, 'content') else str(response)
                        complete_answer = answer.strip()
                        for char in complete_answer:
                            yield StreamEvent(type=StreamEventType.GENERATION_CHUNK, data={"chunk": char}, timestamp=time.time())
                            await asyncio.sleep(0.02)
                except Exception as api_error:
                    logger.error(f"LLM API调用失败: {api_error}")
                    fallback_response = f"抱歉，AI服务暂时不可用。你的问题是：{question}。请稍后重试。"
                    for char in fallback_response:
                        complete_answer += char
                        yield StreamEvent(type=StreamEventType.GENERATION_CHUNK, data={"chunk": char}, timestamp=time.time())
                        await asyncio.sleep(0.02)

            ai_message_id = None
            try:
                if current_session_id:
                    ai_message_id = session_manager.add_message_to_session(db, current_session_id, "assistant", complete_answer.strip())
                    if ai_message_id:
                        logger.info(f"AI回答已保存到会话 {current_session_id}: 消息ID {ai_message_id}")
                    else:
                        logger.error(f"保存AI回答失败: 会话 {current_session_id}")
                else:
                    logger.error("无效的会话ID，无法保存AI回答")
            except Exception as save_error:
                logger.error(f"保存AI回答失败: {save_error}")
                db.rollback()

            yield StreamEvent(type=StreamEventType.GENERATION_END, data={"message": "生成完成"}, timestamp=time.time())
            yield StreamEvent(
                type=StreamEventType.COMPLETE, 
                data={
                    "message": "对话完成", 
                    "session_id": current_session_id,
                    "user_message_id": user_message_id,
                    "ai_message_id": ai_message_id
                }, 
                timestamp=time.time()
            )

        except Exception as e:
            yield StreamEvent(type=StreamEventType.ERROR, data={"error": str(e)}, timestamp=time.time())
            
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
```

## `app/config.py`

```python
# chatbot_core/config.py

from typing import Dict, Any

# --- 模型配置 ---
# (此部分配置从.env文件中读取)

# --- 核心对话配置 ---
# 定义机器人默认使用的系统角色提示词文件名 (不含.txt)
SYSTEM_PROMPT_NAME: str = "assistant_prompt"

# --- 短期记忆配置 ---
ENABLE_SHORT_TERM_MEMORY: bool = True
SHORT_TERM_MEMORY_MAX_LENGTH: int = 100_000 # 最大字符长度
SINGLE_CONVERSATION_MAX_LENGTH: int = 20_000 # 单轮对话最大长度
MIN_CONVERSATION_ROUNDS: int = 1 # 最小保留轮数
MEMORY_CLEANUP_STRATEGY: str = "auto" # "auto"或"sliding_window"
SLIDING_WINDOW_SIZE: int = 20 # 滑动窗口大小时使用

# --- 提示词热重载配置 ---
ENABLE_HOT_RELOAD: bool = True
HOT_RELOAD_DEBOUNCE_TIME: float = 0.5 # 防抖时间（秒）

# --- 日志配置 ---
LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_DIR: str = "log"  # 日志目录
LOG_RETENTION_DAYS: int = 30  # 日志保留天数
ENABLE_CONSOLE_LOG: bool = True  # 是否启用控制台日志
ENABLE_FILE_LOG: bool = True  # 是否启用文件日志
ENABLE_ERROR_LOG: bool = True  # 是否启用单独的错误日志文件
```

## `app/database.py`

```python
# app/database.py

import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship
from datetime import datetime
from typing import Generator
import sqlite3

from .logger_config import get_logger

logger = get_logger(__name__)

class Base(DeclarativeBase):
    pass
'''
# 数据库配置
DATABASE_CONFIG = {
    "user": "postgres",
    "password": "052756", 
    "host": "127.0.0.1",
    "port": "5432",
    "database": "mydb"
}

# 构建数据库URL（使用同步驱动）
DATABASE_URL = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
'''
DATABASE_URL = "sqlite:///stream_chat_bot.db"
# 创建同步引擎

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  
    echo=False  # 设为True可以看到SQL语句
)

# 创建同步会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator:
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"数据库会话错误: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def init_database():
    """初始化数据库表"""
    try:
        # 检查数据库文件是否存在
        db_file = 'stream_chat_bot.db'
        is_new_db = not os.path.exists(db_file)
        
        Base.metadata.create_all(bind=engine)
        if is_new_db:
            logger.info(f"新的SQLite数据库文件 '{db_file}' 已创建并初始化。")
        else:
            logger.info("数据库表初始化检查完成。")

        # 检查新表是否创建成功
        from sqlalchemy import text
        db = SessionLocal()
        try:
            # 检查chat_sessions表
            result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_sessions'"))
            if result.fetchone():
                logger.info("✅ chat_sessions表已存在")
            
            # 检查messages表
            result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"))
            if result.fetchone():
                logger.info("✅ messages表已存在")
                
        except Exception as e:
            logger.warning(f"检查新表时出错: {e}")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise
```

## `app/hot_reload_manager.py`

```python
# rag/hot_reload_manager.py

import os
import time
import threading
from pathlib import Path
from typing import Dict, Set, Optional, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

from .prompt_manager import prompt_manager
from . import config
from .logger_config import get_logger

# 配置日志
logger = get_logger(__name__)


class PromptFileHandler(FileSystemEventHandler):
    """提示词文件变化处理器"""
    
    def __init__(self, callback: Optional[Callable[[str, str], None]] = None):
        """
        初始化文件处理器
        
        Args:
            callback: 文件变化时的回调函数，参数为(event_type, prompt_name)
        """
        super().__init__()
        self.callback = callback
        self.last_modified: Dict[str, float] = {}
        self.debounce_time = config.HOT_RELOAD_DEBOUNCE_TIME  # 防抖时间（秒）
        
    def _should_process_event(self, file_path: str) -> bool:
        """
        判断是否应该处理该事件（防抖处理）
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否应该处理
        """
        current_time = time.time()
        last_time = self.last_modified.get(file_path, 0)
        
        if current_time - last_time < self.debounce_time:
            return False
        
        self.last_modified[file_path] = current_time
        return True
    
    def _get_prompt_name(self, file_path: str) -> Optional[str]:
        """
        从文件路径获取提示词名称
        
        Args:
            file_path: 文件路径
            
        Returns:
            提示词名称，如果不是提示词文件则返回None
        """
        path = Path(file_path)
        
        # 检查是否是提示词文件
        if (path.suffix == '.txt' and 
            'prompts' in str(path) and 
            path.parent.name == 'prompts'):
            return path.stem
        
        return None
    
    def on_modified(self, event):
        """文件修改事件处理"""
        if event.is_directory:
            return
        
        prompt_name = self._get_prompt_name(event.src_path)
        if not prompt_name:
            return
        
        if not self._should_process_event(event.src_path):
            return
        
        try:
            logger.info(f"检测到提示词文件修改: {prompt_name}")
            
            # 清除所有相关缓存
            prompt_manager._prompt_cache.pop(prompt_name, None)
            prompt_manager._template_cache.pop(prompt_name, None)
            
            # 重新加载提示词（这会重新填充缓存）
            prompt_manager.load_prompt(prompt_name)
            logger.info(f"自动重载完成: {prompt_name}")
            
            # 调用回调函数
            if self.callback:
                self.callback("modified", prompt_name)
                
        except Exception as e:
            logger.error(f"自动重载失败 {prompt_name}: {e}")
    
    def on_created(self, event):
        """文件创建事件处理"""
        if event.is_directory:
            return
        
        prompt_name = self._get_prompt_name(event.src_path)
        if not prompt_name:
            return
        
        try:
            logger.info(f"检测到新提示词文件: {prompt_name}")
            
            # 加载新提示词
            prompt_manager.load_prompt(prompt_name)
            logger.info(f"自动加载完成: {prompt_name}")
            
            # 调用回调函数
            if self.callback:
                self.callback("created", prompt_name)
                
        except Exception as e:
            logger.error(f"自动加载失败 {prompt_name}: {e}")
    
    def on_deleted(self, event):
        """文件删除事件处理"""
        if event.is_directory:
            return
        
        prompt_name = self._get_prompt_name(event.src_path)
        if not prompt_name:
            return
        
        try:
            logger.info(f"检测到提示词文件删除: {prompt_name}")
            
            # 从缓存中移除
            prompt_manager._prompt_cache.pop(prompt_name, None)
            prompt_manager._template_cache.pop(prompt_name, None)
            logger.info(f"缓存清理完成: {prompt_name}")
            
            # 调用回调函数
            if self.callback:
                self.callback("deleted", prompt_name)
                
        except Exception as e:
            logger.error(f"缓存清理失败 {prompt_name}: {e}")


class HotReloadManager:
    """热重载管理器"""
    
    def __init__(self, enable_hot_reload: bool = True):
        """
        初始化热重载管理器
        
        Args:
            enable_hot_reload: 是否启用热重载功能
        """
        self.enable_hot_reload = enable_hot_reload
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[PromptFileHandler] = None
        self.is_running = False
        self.callbacks: Set[Callable[[str, str], None]] = set()
        
        # 监控的目录
        self.watch_directory = prompt_manager.prompts_dir
        
        if self.enable_hot_reload:
            self._setup_file_watcher()
    
    def _setup_file_watcher(self):
        """设置文件监控器"""
        try:
            # 确保监控目录存在
            self.watch_directory.mkdir(exist_ok=True)
            
            # 创建事件处理器
            self.event_handler = PromptFileHandler(callback=self._on_file_change)
            
            # 创建观察者
            self.observer = Observer()
            self.observer.schedule(
                self.event_handler,
                str(self.watch_directory),
                recursive=False
            )
            
            logger.info(f"热重载监控已设置，监控目录: {self.watch_directory}")
            
        except Exception as e:
            logger.error(f"设置文件监控器失败: {e}")
            self.enable_hot_reload = False
    
    def _on_file_change(self, event_type: str, prompt_name: str):
        """文件变化回调处理"""
        # 通知所有注册的回调函数
        for callback in self.callbacks:
            try:
                callback(event_type, prompt_name)
            except Exception as e:
                logger.error(f"回调函数执行失败: {e}")
    
    def start(self):
        """启动热重载监控"""
        if not self.enable_hot_reload:
            logger.warning("热重载功能未启用")
            return False
        
        if self.is_running:
            logger.warning("热重载监控已在运行中")
            return True
        
        # 如果observer已经停止，需要重新创建
        if self.observer and not self.observer.is_alive():
            self._setup_file_watcher()
        
        if not self.observer:
            logger.error("文件监控器初始化失败")
            return False
        
        try:
            self.observer.start()
            self.is_running = True
            logger.info(f"热重载监控已启动，正在监控: {self.watch_directory}")
            return True
            
        except Exception as e:
            logger.error(f"启动热重载监控失败: {e}")
            # 尝试重新创建observer
            self._setup_file_watcher()
            if self.observer:
                try:
                    self.observer.start()
                    self.is_running = True
                    logger.info(f"热重载监控已重新启动，正在监控: {self.watch_directory}")
                    return True
                except Exception as e2:
                    logger.error(f"重新启动也失败: {e2}")
            return False
    
    def stop(self):
        """停止热重载监控"""
        if not self.observer or not self.is_running:
            return
        
        try:
            self.observer.stop()
            self.observer.join(timeout=5)  # 等待最多5秒
            self.is_running = False
            logger.info("热重载监控已停止")
            
        except Exception as e:
            logger.error(f"停止热重载监控失败: {e}")
    
    def add_callback(self, callback: Callable[[str, str], None]):
        """
        添加文件变化回调函数
        
        Args:
            callback: 回调函数，参数为(event_type, prompt_name)
        """
        self.callbacks.add(callback)
        logger.info("已添加热重载回调函数")
    
    def remove_callback(self, callback: Callable[[str, str], None]):
        """
        移除文件变化回调函数
        
        Args:
            callback: 要移除的回调函数
        """
        self.callbacks.discard(callback)
        logger.info("已移除热重载回调函数")
    
    def get_status(self) -> Dict[str, any]:
        """
        获取热重载状态信息
        
        Returns:
            状态信息字典
        """
        return {
            "enabled": self.enable_hot_reload,
            "running": self.is_running,
            "watch_directory": str(self.watch_directory),
            "callbacks_count": len(self.callbacks),
            "observer_alive": self.observer.is_alive() if self.observer else False
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


# 检查是否安装了watchdog库
try:
    import watchdog
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("未安装watchdog库，热重载功能不可用")
    logger.info("安装命令: uv add watchdog")


# 创建全局热重载管理器实例
hot_reload_manager = HotReloadManager(
    enable_hot_reload=WATCHDOG_AVAILABLE and getattr(config, 'ENABLE_HOT_RELOAD', True)
) if WATCHDOG_AVAILABLE else None


def enable_hot_reload():
    """启用热重载功能"""
    if not WATCHDOG_AVAILABLE:
        logger.error("watchdog库未安装，无法启用热重载功能")
        logger.info("安装命令: uv add watchdog")
        return False
    
    if hot_reload_manager:
        return hot_reload_manager.start()
    return False


def disable_hot_reload():
    """禁用热重载功能"""
    if hot_reload_manager:
        hot_reload_manager.stop()


def is_hot_reload_enabled() -> bool:
    """检查热重载是否启用"""
    return (hot_reload_manager is not None and 
            hot_reload_manager.is_running if hot_reload_manager else False)


def get_hot_reload_status() -> Dict[str, any]:
    """获取热重载状态"""
    if hot_reload_manager:
        return hot_reload_manager.get_status()
    else:
        return {
            "enabled": False,
            "running": False,
            "error": "watchdog库未安装" if not WATCHDOG_AVAILABLE else "热重载管理器未初始化"
        }
```

## `app/logger_config.py`

```python
# app/logger_config.py

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime, timedelta
import glob


class LoggerConfig:
    """日志配置管理器"""
    
    def __init__(self, log_dir: str = None, max_days: int = None):
        """
        初始化日志配置
        
        Args:
            log_dir: 日志目录，None时从config读取
            max_days: 保留日志的最大天数，None时从config读取
        """
        # 延迟导入避免循环依赖
        try:
            from . import config
            self.log_dir = Path(log_dir or config.LOG_DIR)
            self.max_days = max_days or config.LOG_RETENTION_DAYS
            self.enable_console = config.ENABLE_CONSOLE_LOG
            self.enable_file = config.ENABLE_FILE_LOG
            self.enable_error = config.ENABLE_ERROR_LOG
            self.log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
        except ImportError:
            # 如果无法导入config，使用默认值
            self.log_dir = Path(log_dir or "log")
            self.max_days = max_days or 30
            self.enable_console = True
            self.enable_file = True
            self.enable_error = True
            self.log_level = logging.INFO
        
        self.log_dir.mkdir(exist_ok=True)
        
        # 清理旧日志
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """清理超过保留期的日志文件"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_days)
            
            # 查找所有日志文件
            log_files = glob.glob(str(self.log_dir / "*.log"))
            
            for log_file in log_files:
                file_path = Path(log_file)
                # 获取文件修改时间
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_mtime < cutoff_date:
                    try:
                        file_path.unlink()
                        print(f"🗑️ 已删除过期日志文件: {file_path.name}")
                    except Exception as e:
                        print(f"❌ 删除日志文件失败 {file_path.name}: {e}")
                        
        except Exception as e:
            print(f"❌ 清理旧日志失败: {e}")
    
    def setup_logger(self, name: str = None, level: int = None) -> logging.Logger:
        """
        设置并返回配置好的日志器
        
        Args:
            name: 日志器名称，默认为调用模块名
            level: 日志级别，None时使用配置文件中的级别
            
        Returns:
            配置好的日志器
        """
        if name is None:
            name = __name__
        
        if level is None:
            level = self.log_level
        
        logger = logging.getLogger(name)
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
        
        logger.setLevel(level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 1. 控制台处理器（根据配置决定是否启用）
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 2. 文件处理器 - 按日期轮转（根据配置决定是否启用）
        if self.enable_file:
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"chatbot_{today}.log"
            
            file_handler = logging.FileHandler(
                log_file, 
                mode='a', 
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # 3. 错误日志单独文件（根据配置决定是否启用）
        if self.enable_error:
            today = datetime.now().strftime("%Y-%m-%d")
            error_log_file = self.log_dir / f"error_{today}.log"
            error_handler = logging.FileHandler(
                error_log_file,
                mode='a',
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)
        
        # 防止日志向上传播到根日志器
        logger.propagate = False
        
        return logger
    
    def get_log_stats(self) -> dict:
        """
        获取日志统计信息
        
        Returns:
            日志统计信息字典
        """
        try:
            log_files = list(self.log_dir.glob("*.log"))
            total_size = sum(f.stat().st_size for f in log_files)
            
            return {
                "log_directory": str(self.log_dir),
                "total_files": len(log_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "retention_days": self.max_days,
                "files": [
                    {
                        "name": f.name,
                        "size_kb": round(f.stat().st_size / 1024, 2),
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    }
                    for f in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)
                ]
            }
        except Exception as e:
            return {"error": f"获取日志统计失败: {e}"}


# 创建全局日志配置实例
logger_config = LoggerConfig()

# 便捷函数
def get_logger(name: str = None) -> logging.Logger:
    """获取配置好的日志器"""
    return logger_config.setup_logger(name)
```

## `app/models.py`

```python
# app/models.py

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("Message", back_populates="chat_session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' 或 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    chat_session = relationship("ChatSession", back_populates="messages")

# 保留旧的Conversation表以兼容现有数据
class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    # 注意：旧表没有updated_at字段，所以不包含在模型中
    
    # 关系
    user = relationship("User", back_populates="conversations")
```

## `app/prompt_manager.py`

```python
# rag/prompt_manager.py

import os
from pathlib import Path
from typing import Dict, Optional, Any
from langchain_core.prompts import PromptTemplate
from .logger_config import get_logger

# 配置日志
logger = get_logger(__name__)


class PromptManager:
    """
    提示词管理器，负责加载和管理所有提示词模板。
    实现提示词与代码的解耦。
    """
    
    def __init__(self):
        """初始化提示词管理器。"""
        self.prompts_dir = Path(__file__).parent / "prompts"
        self._prompt_cache: Dict[str, str] = {}
        self._template_cache: Dict[str, PromptTemplate] = {}
        
        # 确保提示词目录存在
        self.prompts_dir.mkdir(exist_ok=True)
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        加载指定的提示词内容。
        
        Args:
            prompt_name: 提示词文件名（不含扩展名）
            
        Returns:
            提示词内容字符串
            
        Raises:
            FileNotFoundError: 如果提示词文件不存在
        """
        # 检查缓存
        if prompt_name in self._prompt_cache:
            return self._prompt_cache[prompt_name]
        
        # 构建文件路径
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"提示词文件不存在: {prompt_file}")
        
        # 读取文件内容
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 缓存内容
            self._prompt_cache[prompt_name] = content
            return content
            
        except Exception as e:
            raise RuntimeError(f"读取提示词文件失败 {prompt_file}: {e}")
    
    def get_template(self, prompt_name: str) -> PromptTemplate:
        """
        获取指定的提示词模板对象。
        
        Args:
            prompt_name: 提示词文件名（不含扩展名）
            
        Returns:
            LangChain PromptTemplate 对象
        """
        # 检查缓存
        if prompt_name in self._template_cache:
            return self._template_cache[prompt_name]
        
        # 加载提示词内容
        prompt_content = self.load_prompt(prompt_name)
        
        # 创建模板对象
        template = PromptTemplate.from_template(prompt_content)
        
        # 缓存模板
        self._template_cache[prompt_name] = template
        return template
    
    def reload_prompt(self, prompt_name: str) -> str:
        """
        重新加载指定的提示词（清除缓存后重新读取）。
        
        Args:
            prompt_name: 提示词文件名（不含扩展名）
            
        Returns:
            提示词内容字符串
        """
        # 清除缓存
        self._prompt_cache.pop(prompt_name, None)
        self._template_cache.pop(prompt_name, None)
        
        # 重新加载
        return self.load_prompt(prompt_name)
    
    def list_available_prompts(self) -> list:
        """
        列出所有可用的提示词文件。
        
        Returns:
            提示词文件名列表（不含扩展名）
        """
        prompt_files = []
        # 使用 pathlib.Path.glob() 方法 (推荐)
        for file_path in self.prompts_dir.glob("*.txt"):
            prompt_files.append(file_path.stem)  # .stem 获取不含扩展名的文件名
        return sorted(prompt_files)
        
        # 如果使用标准库 glob 的等价写法：
        # import glob
        # pattern = str(self.prompts_dir / "*.txt")
        # for file_path in glob.glob(pattern):
        #     filename = os.path.splitext(os.path.basename(file_path))[0]
        #     prompt_files.append(filename)
    
    def save_prompt(self, prompt_name: str, content: str) -> None:
        """
        保存提示词到文件。
        
        Args:
            prompt_name: 提示词文件名（不含扩展名）
            content: 提示词内容
        """
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        try:
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            
            # 清除缓存，确保下次加载时使用新内容
            self._prompt_cache.pop(prompt_name, None)
            self._template_cache.pop(prompt_name, None)
            
            logger.info(f"提示词已保存到: {prompt_file}")
            
        except Exception as e:
            raise RuntimeError(f"保存提示词文件失败 {prompt_file}: {e}")
    
    def clear_cache(self) -> None:
        """清除所有缓存。"""
        self._prompt_cache.clear()
        self._template_cache.clear()
        logger.info("提示词缓存已清除")
    
    def reload_all_prompts(self) -> Dict[str, str]:
        """
        重新加载所有提示词。
        
        Returns:
            重新加载的提示词字典
        """
        # 清除所有缓存
        self.clear_cache()
        
        # 重新加载所有提示词
        reloaded_prompts = {}
        for prompt_name in self.list_available_prompts():
            try:
                content = self.load_prompt(prompt_name)
                reloaded_prompts[prompt_name] = content
                logger.info(f"重新加载: {prompt_name}")
            except Exception as e:
                logger.error(f"重新加载失败 {prompt_name}: {e}")
        
        return reloaded_prompts
    
    def get_prompt_info(self, prompt_name: str) -> Dict[str, Any]:
        """
        获取提示词的详细信息。
        
        Args:
            prompt_name: 提示词文件名（不含扩展名）
            
        Returns:
            提示词信息字典
        """
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            return {"exists": False, "error": f"提示词文件不存在: {prompt_file}"}
        
        try:
            stat = prompt_file.stat()
            content = self.load_prompt(prompt_name)
            template = self.get_template(prompt_name)
            
            return {
                "exists": True,
                "file_path": str(prompt_file),
                "file_size": stat.st_size,
                "modified_time": stat.st_mtime,
                "content_length": len(content),
                "content_preview": content[:100] + "..." if len(content) > 100 else content,
                "template_variables": template.input_variables,
                "is_cached": prompt_name in self._prompt_cache
            }
        except Exception as e:
            return {"exists": True, "error": f"获取提示词信息失败: {e}"}
    
    def validate_prompt(self, prompt_name: str) -> Dict[str, Any]:
        """
        验证提示词模板的有效性。
        
        Args:
            prompt_name: 提示词文件名（不含扩展名）
            
        Returns:
            验证结果字典
        """
        try:
            template = self.get_template(prompt_name)
            
            # 检查必需的变量
            required_vars = {"context", "question"}  # 问答提示词的必需变量
            missing_vars = required_vars - set(template.input_variables)
            extra_vars = set(template.input_variables) - required_vars
            
            # 尝试格式化测试
            test_values = {var: f"test_{var}" for var in template.input_variables}
            try:
                formatted = template.format(**test_values)
                format_test = {"success": True, "formatted_length": len(formatted)}
            except Exception as e:
                format_test = {"success": False, "error": str(e)}
            
            return {
                "valid": len(missing_vars) == 0 and format_test["success"],
                "template_variables": template.input_variables,
                "missing_variables": list(missing_vars),
                "extra_variables": list(extra_vars),
                "format_test": format_test
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"验证提示词失败: {e}"
            }


# 创建全局提示词管理器实例
prompt_manager = PromptManager()

'''
def get_qa_prompt_template() -> PromptTemplate:
    """获取问答提示词模板。"""
    return prompt_manager.get_template("qa_prompt")


def get_query_rewrite_prompt_template() -> PromptTemplate:
    """获取问题改写提示词模板。"""
    return prompt_manager.get_template("query_rewrite_prompt")


def load_qa_prompt() -> str:
    """加载问答提示词内容。"""
    return prompt_manager.load_prompt("qa_prompt")


def load_query_rewrite_prompt() -> str:
    """加载问题改写提示词内容。"""
    return prompt_manager.load_prompt("query_rewrite_prompt")
'''
```

## `app/prompts/assistant_prompt.txt`

````
你是一个名为 "AI-Jay" 的企业级AI助手。

你的职责是：
1. 以友好、专业、乐于助人的语气与用户交流，可以使用Emoji 表情。
2. 能够基于对话历史（如果提供）进行多轮对话，理解上下文和代词指代。
3. 如果遇到不知道如何回答的问题，要诚实地说明，而不是编造答案。
4. 你的回答应力求简洁、清晰、有条理。
5. 在对话开始时，可以简单地问候用户。
````

## `app/session_manager.py`

```python
# app/session_manager.py

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
import uuid

from .models import ChatSession, Message, User, Conversation
from .logger_config import get_logger

logger = get_logger(__name__)

class SessionManager:
    """完整的会话管理器 - 支持真正的多轮对话会话"""
    
    def __init__(self):
        self.active_sessions: Dict[int, List[Dict]] = {}  # 以session_id为key的活跃会话
    
    def create_new_session(self, db: Session, user_id: int, first_question: str) -> int:
        """创建新的会话"""
        try:
            # 生成会话标题（使用问题的前50个字符）
            title = first_question[:50] + "..." if len(first_question) > 50 else first_question
            
            # 创建数据库会话记录
            chat_session = ChatSession(
                user_id=user_id,
                title=title
            )
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)
            
            # 初始化内存会话
            self.active_sessions[chat_session.id] = []
            
            logger.info(f"创建新会话: {chat_session.id} for 用户 {user_id}, 标题: {title}")
            return chat_session.id
            
        except Exception as e:
            logger.error(f"创建新会话失败: {e}")
            db.rollback()
            return None
    
    def get_session_messages(self, db: Session, session_id: int, user_id: int) -> List[Dict]:
        """获取会话的所有消息"""
        try:
            # 验证会话属于该用户
            session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id
            ).first()
            
            if not session:
                logger.warning(f"会话 {session_id} 不存在或不属于用户 {user_id}")
                return []
            
            # 获取会话的所有消息
            messages = db.query(Message).filter(
                Message.chat_session_id == session_id
            ).order_by(Message.created_at).all()
            
            # 转换为字典格式
            message_list = []
            for msg in messages:
                message_list.append({
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat() + "Z"
                })
            
            return message_list
            
        except Exception as e:
            logger.error(f"获取会话消息失败: {e}")
            return []
    
    def add_message_to_session(self, db: Session, session_id: int, role: str, content: str) -> Optional[int]:
        """添加消息到会话"""
        try:
            message = Message(
                chat_session_id=session_id,
                role=role,
                content=content
            )
            db.add(message)
            
            # 更新会话的最后更新时间
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session:
                session.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(message)
            
            # 同时更新内存中的会话
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = []
            
            self.active_sessions[session_id].append({
                "id": message.id,
                "role": role,
                "content": content,
                "timestamp": message.created_at
            })
            
            # 限制内存中的消息数量
            if len(self.active_sessions[session_id]) > 50:
                self.active_sessions[session_id] = self.active_sessions[session_id][-50:]
            
            logger.debug(f"添加消息到会话 {session_id}: {role}")
            return message.id
            
        except Exception as e:
            logger.error(f"添加消息失败: {e}")
            db.rollback()
            return None
    
    def get_user_sessions(self, db: Session, user_id: int) -> List[Dict]:
        """获取用户的所有会话列表"""
        try:
            sessions = db.query(ChatSession).filter(
                ChatSession.user_id == user_id
            ).order_by(ChatSession.updated_at.desc()).limit(50).all()
            
            session_list = []
            for session in sessions:
                # 获取会话的最后一条消息作为预览
                last_message = db.query(Message).filter(
                    Message.chat_session_id == session.id
                ).order_by(Message.created_at.desc()).first()
                
                preview = ""
                if last_message:
                    preview = last_message.content[:100] + "..." if len(last_message.content) > 100 else last_message.content
                
                session_list.append({
                    "id": session.id,
                    "title": session.title,
                    "preview": preview,
                    "created_at": session.created_at.isoformat() + "Z",
                    "updated_at": session.updated_at.isoformat() + "Z",
                    "message_count": len(session.messages)
                })
            
            return session_list
            
        except Exception as e:
            logger.error(f"获取用户会话列表失败: {e}")
            return []
    
    def delete_session(self, db: Session, user_id: int, session_id: int) -> bool:
        """删除会话（级联删除所有消息）"""
        try:
            session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id
            ).first()
            
            if session:
                db.delete(session)  # 级联删除所有相关消息
                db.commit()
                
                # 清除内存中的会话
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                
                logger.info(f"删除会话成功: 用户{user_id}, 会话{session_id}")
                return True
            else:
                logger.warning(f"未找到要删除的会话: 用户{user_id}, 会话{session_id}")
                return False
                
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            db.rollback()
            return False
    
    def get_session_context_for_ai(self, db: Session, session_id: int, user_id: int, max_messages: int = 10) -> List[Dict]:
        """获取会话上下文供AI使用"""
        try:
            messages = self.get_session_messages(db, session_id, user_id)
            
            # 返回最近的消息作为上下文
            return messages[-max_messages:] if len(messages) > max_messages else messages
            
        except Exception as e:
            logger.error(f"获取会话上下文失败: {e}")
            return []
    
    # 兼容旧系统的方法
    def get_legacy_conversations(self, db: Session, user_id: int) -> List[Dict]:
        """获取旧的对话记录（兼容现有前端）"""
        try:
            conversations = db.query(Conversation).filter(
                Conversation.user_id == user_id
            ).order_by(Conversation.created_at.desc()).limit(50).all()
            
            conv_list = []
            for conv in conversations:
                conv_list.append({
                    "id": conv.id,
                    "title": conv.question[:50] + "..." if len(conv.question) > 50 else conv.question,
                    "preview": conv.answer[:100] + "..." if len(conv.answer) > 100 else conv.answer,
                    "created_at": conv.created_at.isoformat(),
                    "question": conv.question,
                    "answer": conv.answer
                })
            
            return conv_list
            
        except Exception as e:
            logger.error(f"获取旧对话记录失败: {e}")
            return []

# 创建全局会话管理器实例
session_manager = SessionManager()
```

## `chatbot_web_demo.py`

```python
# chatbot_web_demo.py

import asyncio
import json
import time
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session

# 导入我们的对话机器人核心
from app.chatbot_pipeline import ChatbotPipeline, StreamEventType, StreamEvent
from app import config
from app.hot_reload_manager import hot_reload_manager

# 导入数据库和认证相关
from app.database import engine, get_db
from app.models import Base, User, Conversation, ChatSession
from app.api_routes import router as api_router

# 导入新的日志配置
from app.logger_config import get_logger

# 配置日志
logger = get_logger(__name__)
# 全局单例
pipeline: ChatbotPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI应用的生命周期管理器。
    在应用启动时执行yield之前的部分，在应用关闭时执行yield之后的部分。
    """
    # --- 应用启动时执行 ---
    global pipeline
    logger.info("应用启动，正在初始化...")
    try:
        # 初始化数据库
        from app.database import init_database
        init_database()
        logger.info("数据库初始化完成。")
        
        # 初始化对话机器人
        pipeline = ChatbotPipeline()
        logger.info("对话机器人初始化完成。")
        
        # 启动热重载
        if hot_reload_manager and config.ENABLE_HOT_RELOAD:
            hot_reload_manager.start()
            
    except Exception as e:
        logger.error(f"应用初始化失败: {e}", exc_info=True)
        # 即使失败，也需要yield一次，让FastAPI知道启动流程已（不成功地）走完
    
    yield  # <--- 这是关键的分割点

    # --- 应用关闭时执行 ---
    logger.info("应用关闭...")
    if hot_reload_manager:
        hot_reload_manager.stop()
        
    if pipeline and hasattr(pipeline, 'executor'):
        logger.info("正在清理线程池...")
        pipeline.executor.shutdown(wait=True)
        logger.info("线程池已关闭。")

app = FastAPI(
    title="企业级AI对话机器人", 
    description="一个支持实时流式响应、具备记忆和可热重载角色的高级对话平台",
    lifespan=lifespan # <--- 在这里注册
)

# 注册API路由
app.include_router(api_router, prefix="/api", tags=["auth"])

# --- 静态文件服务 ---
# 挂载static目录，让FastAPI能直接提供HTML, CSS, JS文件
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_homepage():
    """
    当用户访问根路径时，返回我们的主HTML文件。
    """
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    """
    健康检查端点，用于Docker和负载均衡器检查服务状态。
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "chatbot-api",
        "version": "1.0.0"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket连接已建立")
    
    user = None
    db = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理认证消息
            if message.get("type") == "auth":
                token = message.get("token", "")
                if not token:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "未提供认证令牌"}
                    }))
                    continue
                
                # 验证token
                try:
                    from jose import jwt, JWTError
                    import os
                    
                    SECRET_KEY = os.getenv("SECRET_KEY")
                    ALGORITHM = os.getenv("ALGORITHM", "HS256")
                    
                    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                    email = payload.get("sub")
                    
                    if not email:
                        await websocket.send_text(json.dumps({
                            "type": "auth_error",
                            "data": {"error": "无效的认证令牌"}
                        }))
                        continue
                        
                    # 获取数据库会话和用户信息
                    from app.database import SessionLocal
                    db = SessionLocal()
                    try:
                        # 使用同步查询
                        user = db.query(User).filter(User.email == email).first()
                    except Exception as e:
                        logger.error(f"数据库查询错误: {e}")
                        db.close()
                        continue
                    
                except JWTError:
                    await websocket.send_text(json.dumps({
                        "type": "auth_error",
                        "data": {"error": "无效的认证令牌"}
                    }))
                    continue
                except Exception as e:
                    logger.error(f"WebSocket认证错误: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "auth_error",
                        "data": {"error": "认证失败"}
                    }))
                    continue
                
                if not user:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "用户不存在"}
                    }))
                    continue
                
                # 认证成功
                await websocket.send_text(json.dumps({
                    "type": "auth_success",
                    "data": {"message": f"欢迎回来，{user.email}！"}
                }))
                logger.info(f"用户 {user.email} WebSocket认证成功")
                continue
            
            # 处理对话消息
            if message.get("type") == "question":
                # 检查用户是否已认证
                if not user or not db:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "请先进行身份认证"}
                    }))
                    continue
                
                question = message.get("content", "")
                session_id = message.get("session_id")  # 可选的会话ID
                
                if not pipeline:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "机器人核心引擎未准备就绪，请检查服务器日志。"}
                    }))
                    continue

                logger.info(f"用户 {user.email} 在会话 {session_id} 中提问: {question}")
                
                # 使用用户ID和会话ID进行对话
                async for event in pipeline.ask_stream(question, db, user.id, session_id):
                    response = {"type": event.type.value, "data": event.data}
                    await websocket.send_text(json.dumps(response))
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接已断开 - 用户: {user.email if user else '未认证'}")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}", exc_info=True)
        if websocket.client_state == 1:  # OPEN
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": f"服务器内部错误: {str(e)}"}
            }))
    finally:
        # 确保数据库会话被正确关闭
        if db:
            try:
                db.close()
            except Exception as e:
                logger.error(f"关闭数据库会话时出错: {e}")

if __name__ == "__main__":
    import uvicorn
    logger.info("🤖 启动企业级AI对话机器人Web演示...")
    logger.info("🌐 访问地址: http://localhost:8003")
    logger.info("🔥 提示词热重载已激活，尝试修改 app/prompts/assistant_prompt.txt 并刷新对话！")
    
    # 显示日志统计信息
    from app.logger_config import logger_config
    log_stats = logger_config.get_log_stats()
    logger.info(f"📊 日志系统已启动，日志目录: {log_stats['log_directory']}")
    
    uvicorn.run("chatbot_web_demo:app", host="0.0.0.0", port=8003, reload=True)
```

## `log_manager.py`

```python
#!/usr/bin/env python3
# log_manager.py

"""
日志管理工具
提供日志查看、清理、统计等功能
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# 添加app目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from app.logger_config import logger_config


def show_log_stats():
    """显示日志统计信息"""
    stats = logger_config.get_log_stats()
    
    if "error" in stats:
        print(f"❌ {stats['error']}")
        return
    
    print("📊 日志统计信息")
    print("=" * 50)
    print(f"日志目录: {stats['log_directory']}")
    print(f"文件总数: {stats['total_files']}")
    print(f"总大小: {stats['total_size_mb']} MB")
    print(f"保留天数: {stats['retention_days']} 天")
    print()
    
    if stats['files']:
        print("📁 日志文件列表:")
        for file_info in stats['files']:
            print(f"  {file_info['name']} - {file_info['size_kb']} KB - {file_info['modified']}")
    else:
        print("📁 暂无日志文件")


def view_log(log_type="chatbot", date=None, lines=50):
    """查看日志内容"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    log_file = logger_config.log_dir / f"{log_type}_{date}.log"
    
    if not log_file.exists():
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    print(f"📖 查看日志: {log_file}")
    print("=" * 50)
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            
        # 显示最后N行
        display_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        for line in display_lines:
            print(line.rstrip())
            
        if len(all_lines) > lines:
            print(f"\n... (显示最后 {lines} 行，共 {len(all_lines)} 行)")
            
    except Exception as e:
        print(f"❌ 读取日志文件失败: {e}")


def cleanup_old_logs(days=None):
    """清理旧日志"""
    if days is None:
        days = logger_config.max_days
    
    print(f"🧹 清理 {days} 天前的日志文件...")
    
    cutoff_date = datetime.now() - timedelta(days=days)
    cleaned_count = 0
    
    for log_file in logger_config.log_dir.glob("*.log"):
        try:
            file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if file_mtime < cutoff_date:
                log_file.unlink()
                print(f"  ✅ 已删除: {log_file.name}")
                cleaned_count += 1
        except Exception as e:
            print(f"  ❌ 删除失败 {log_file.name}: {e}")
    
    print(f"🎉 清理完成，共删除 {cleaned_count} 个文件")


def main():
    parser = argparse.ArgumentParser(description="日志管理工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # stats命令
    subparsers.add_parser("stats", help="显示日志统计信息")
    
    # view命令
    view_parser = subparsers.add_parser("view", help="查看日志内容")
    view_parser.add_argument("--type", default="chatbot", help="日志类型 (chatbot/error)")
    view_parser.add_argument("--date", help="日期 (YYYY-MM-DD)")
    view_parser.add_argument("--lines", type=int, default=50, help="显示行数")
    
    # cleanup命令
    cleanup_parser = subparsers.add_parser("cleanup", help="清理旧日志")
    cleanup_parser.add_argument("--days", type=int, help="保留天数")
    
    args = parser.parse_args()
    
    if args.command == "stats":
        show_log_stats()
    elif args.command == "view":
        view_log(args.type, args.date, args.lines)
    elif args.command == "cleanup":
        cleanup_old_logs(args.days)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

## `pyproject.toml`

```
[project]
name = "rag-example"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "asyncpg>=0.30.0",
    "bcrypt>=4.3.0",
    "chromadb>=1.0.15",
    "dotenv>=0.9.9",
    "email-validator>=2.2.0",
    "fastapi>=0.116.1",
    "jieba>=0.42.1",
    "langchain>=0.3.26",
    "langchain-chroma>=0.2.5",
    "langchain-community>=0.3.27",
    "langchain-huggingface>=0.3.1",
    "langchain-openai>=0.3.28",
    "openai>=1.97.0",
    "pandas>=2.3.1",
    "passlib[bcrypt]>=1.7.4",
    "psycopg2>=2.9.10",
    "pytest>=8.4.1",
    "pytest-asyncio>=1.1.0",
    "python-jose[cryptography]>=3.5.0",
    "python-multipart>=0.0.20",
    "rank-bm25>=0.2.2",
    "sentence-transformers>=5.0.0",
    "sqlalchemy[asyncio]>=2.0.41",
    "sse-starlette>=3.0.2",
    "uvicorn>=0.35.0",
    "watchdog>=6.0.0",
    "websockets>=14.0",
]

```

## `README.md`

````text
\# 🤖 企业级AI对话机器人平台 (V1.0)

本项目是一个功能完备、架构先进、支持实时流式响应、具备短期记忆和可热重载角色功能的企业级对话式AI平台。它从一个复杂的RAG系统中精炼而来，剥离了检索增强的特定逻辑，专注于提供一个通用的、高性能的对话机器人核心框架，可被轻松定制为任何角色。

#\# ✨ 核心特性

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

#\# 🏗️ 项目结构

\`\`\`
/enterprise_chatbot
|-- /chatbot_core/             \# 机器人核心模块
|   |-- /prompts/                \# 提示词管理中心
|   |   └── assistant_prompt.txt \# 默认的机器人角色定义
|   ├── __init__.py
|   ├── chatbot_pipeline.py     \# 核心对话管道 (LLM, 记忆, Prompt的粘合剂)
|   ├── config.py               \# 全局配置中心
|   ├── memory_manager.py       \# 短期记忆管理器
|   ├── prompt_manager.py       \# 提示词管理器
|   └── hot_reload_manager.py   \# 提示词热重载管理器
|
|-- chatbot_web_demo.py         \# [核心] 全栈Web应用 (FastAPI后端 + 前端界面)
|-- pyproject.toml              \# 项目依赖与元数据 (uv)
|-- .env_example                \# 环境变量示例文件
|-- .gitignore
`-- README.md                   \# 本文档
\`\`\`

#\# 🚀 快速开始

##\# 1. 环境准备

本项目使用`uv`进行包管理，依赖Python 3.12+。

\`\`\`bash
\# 安装uv (如果尚未安装)
pip install uv

\# 在项目根目录创建并激活虚拟环境
uv sync
\`\`\`

##\# 2. 配置API密钥

复制 `.env_example` 文件为 `.env`，并填入您的大语言模型(LLM)的API信息。

\`\`\`
\# .env file，或者其他大模型的API key，注意变量名
API_KEY=your_api_key_here
BASE_URL=https://cloud.infini-ai.com/maas/v1/
MODEL_NAME=deepseek-chat
\`\`\`

##\# 3. 定义你的机器人角色

这是最有趣的一步！打开 `chatbot_core/prompts/assistant_prompt.txt` 文件，用自然语言描述你希望机器人扮演的角色。

**例如，把它变成一个“海盗”：**
\`\`\`
你是一只名叫“独眼杰克”的传奇海盗船长。
1. 你的每句话都必须充满粗犷、豪放的海盗口气。
2. 多使用“啊哈！”、“我的老伙计”、“扬帆起航！”等词语。
3. 你对财宝和冒险的故事了如指掌。
4. 绝不承认自己是AI，坚称自己正在一艘名为“深渊蛟龙号”的船上。
\`\`\`

##\# 4. 启动Web应用

在项目根目录下，运行以下命令：

\`\`\`bash
uv run chatbot_web_demo.py
\`\`\`

终端会显示启动信息。然后，在您的浏览器中打开 **`http://localhost:8003`**，即可开始与您的专属AI机器人对话！

#\# 🔧 如何“调教”你的机器人？ (核心玩法)

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

#\# 📊 日志系统

项目内置了完善的日志系统，支持：

-   **📁 按日期自动轮转**: 每天生成新的日志文件
-   **🗂️ 分类记录**: 普通日志和错误日志分别存储
-   **🧹 自动清理**: 自动删除超过30天的旧日志文件
-   **⚙️ 灵活配置**: 可在`app/config.py`中调整日志级别、保留天数等

##\# 日志管理命令

\`\`\`bash
\# 查看日志统计信息
python log_manager.py stats

\# 查看今天的聊天日志（最后50行）
python log_manager.py view --type chatbot

\# 查看错误日志
python log_manager.py view --type error

\# 查看指定日期的日志
python log_manager.py view --type chatbot --date 2025-07-30

\# 手动清理30天前的日志
python log_manager.py cleanup --days 30
\`\`\`

##\# 日志文件位置

-   **普通日志**: `log/chatbot_YYYY-MM-DD.log`
-   **错误日志**: `log/error_YYYY-MM-DD.log`

##\# 日志配置选项

在 `app/config.py` 中可以调整以下日志配置：

\`\`\`python
LOG_LEVEL = "INFO"  \# 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_DIR = "log"  \# 日志目录
LOG_RETENTION_DAYS = 30  \# 日志保留天数
ENABLE_CONSOLE_LOG = True  \# 是否启用控制台日志
ENABLE_FILE_LOG = True  \# 是否启用文件日志
ENABLE_ERROR_LOG = True  \# 是否启用单独的错误日志文件
\`\`\`

#\# 🤝 贡献与致谢

本项目是我们智慧的结晶，其设计和实现深受社区优秀项目的启发。我们对[LangChain](https://github.com/langchain-ai/langchain)、[FastAPI](https://github.com/tiangolo/fastapi)、[HuggingFace](https://huggingface.co/)等开源社区表示最诚挚的感谢。

欢迎通过Fork和Pull Request为本项目贡献代码。

---


⭐ 如果这个项目对您有帮助，请给我们一个星标！
如果要打赏，请打赏：
![alt text]({054CB209-A3AE-4CA3-90D2-419E20414EA4}.png)

````

## `static/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>企业级AI对话机器人</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="一个基于AI大语言模型的企业级对话机器人，支持实时流式响应、上下文记忆和动态角色配置。">
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <!-- 登录/注册界面 -->
    <div id="authContainer" class="auth-container">
        <div class="auth-form">
            <h1>🤖 企业级AI对话机器人</h1>
            <div class="auth-tabs">
                <button id="loginTab" class="auth-tab active">登录</button>
                <button id="registerTab" class="auth-tab">注册</button>
            </div>
            
            <!-- 登录表单 -->
            <form id="loginForm" class="auth-form-content">
                <div class="form-group">
                    <label for="loginEmail">邮箱</label>
                    <input type="email" id="loginEmail" required placeholder="请输入邮箱">
                </div>
                <div class="form-group">
                    <label for="loginPassword">密码</label>
                    <input type="password" id="loginPassword" required placeholder="请输入密码">
                </div>
                <button type="submit" class="auth-button">登录</button>
            </form>
            
            <!-- 注册表单 -->
            <form id="registerForm" class="auth-form-content hidden">
                <div class="form-group">
                    <label for="registerEmail">邮箱</label>
                    <input type="email" id="registerEmail" required placeholder="请输入邮箱">
                </div>
                <div class="form-group">
                    <label for="registerPassword">密码</label>
                    <input type="password" id="registerPassword" required placeholder="请输入密码（至少6位）" minlength="6">
                </div>
                <div class="form-group">
                    <label for="confirmPassword">确认密码</label>
                    <input type="password" id="confirmPassword" required placeholder="请再次输入密码" minlength="6">
                </div>
                <button type="submit" class="auth-button">注册</button>
            </form>
            
            <div id="authMessage" class="auth-message"></div>
        </div>
    </div>

    <!-- 聊天界面 -->
    <div id="chatApp" class="chat-app hidden">
        <!-- 左侧边栏 -->
        <div class="sidebar">
            <div class="sidebar-header">
                <button id="newChatButton" class="new-chat-button">
                    <span class="icon">+</span>
                    新建对话
                </button>
            </div>
            
            <div class="chat-history">
                <div class="chat-history-header">
                    <h3>聊天记录</h3>
                </div>
                <div id="chatHistoryList" class="chat-history-list">
                    <!-- 聊天记录将在这里动态加载 -->
                </div>
            </div>
            
            <div class="sidebar-footer">
                <div class="user-info">
                    <div class="user-avatar">👤</div>
                    <div class="user-details">
                        <div id="userEmail" class="user-email"></div>
                        <button id="logoutButton" class="logout-button">登出</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- 主聊天区域 -->
        <div class="main-content">
            <div class="chat-header">
                <h1>🤖 企业级AI对话机器人</h1>
                <div id="connectionStatus" class="connection-status disconnected">正在连接...</div>
            </div>
            
            <div id="chatContainer" class="chat-container">
                <div class="welcome-message">
                    <div class="welcome-icon">🤖</div>
                    <h2>欢迎使用AI助手</h2>
                    <p>我是AI-Jay，随时准备为您服务。您可以问我任何问题！</p>
                </div>
            </div>
            
            <div class="input-container">
                <div class="input-wrapper">
                    <input type="text" id="questionInput" placeholder="请输入您的问题..." />
                    <button id="sendButton" disabled>
                        <span class="send-icon">➤</span>
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="/static/main.js"></script>
</body>
</html>
```

## `static/main.js`

```javascript
// static/main.js

(() => {
    // 全局状态
    let ws = null;
    let currentUser = null;
    let accessToken = null;
    let currentSessionId = null;
    let chatSessions = [];
    let currentUserMessageDiv = null; // 用于追踪新会话的用户消息DOM
    let currentBotMessageDiv = null;  // 用于追踪新会话的机器人消息DOM
    
    // DOM元素
    const authContainer = document.getElementById('authContainer');
    const chatApp = document.getElementById('chatApp');
    const loginTab = document.getElementById('loginTab');
    const registerTab = document.getElementById('registerTab');
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const authMessage = document.getElementById('authMessage');
    const userEmail = document.getElementById('userEmail');
    const logoutButton = document.getElementById('logoutButton');
    const newChatButton = document.getElementById('newChatButton');
    const chatHistoryList = document.getElementById('chatHistoryList');
    const chatContainer = document.getElementById('chatContainer');
    const questionInput = document.getElementById('questionInput');
    const sendButton = document.getElementById('sendButton');
    const connectionStatus = document.getElementById('connectionStatus');
    
    function init() {
        const savedToken = localStorage.getItem('access_token');
        if (savedToken) {
            accessToken = savedToken;
            verifyTokenAndShowChat();
        } else {
            showAuthInterface();
        }
        bindEvents();
    }
    
    function bindEvents() {
        loginTab.addEventListener('click', () => switchAuthTab('login'));
        registerTab.addEventListener('click', () => switchAuthTab('register'));
        loginForm.addEventListener('submit', handleLogin);
        registerForm.addEventListener('submit', handleRegister);
        logoutButton.addEventListener('click', handleLogout);
        newChatButton.addEventListener('click', startNewChat);
        sendButton.addEventListener('click', sendQuestion);
        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuestion();
            }
        });
    }
    
    function switchAuthTab(tab) {
        if (tab === 'login') {
            loginTab.classList.add('active');
            registerTab.classList.remove('active');
            loginForm.classList.remove('hidden');
            registerForm.classList.add('hidden');
        } else {
            registerTab.classList.add('active');
            loginTab.classList.remove('active');
            registerForm.classList.remove('hidden');
            loginForm.classList.add('hidden');
        }
        clearAuthMessage();
    }
    
    async function handleLogin(e) {
        e.preventDefault();
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;
        
        try {
            showAuthMessage('正在登录...', 'info');
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });
            const data = await response.json();
            if (response.ok) {
                accessToken = data.access_token;
                currentUser = { email: data.user_email };
                localStorage.setItem('access_token', accessToken);
                showAuthMessage('登录成功！', 'success');
                setTimeout(() => showChatInterface(), 1000);
            } else {
                showAuthMessage(data.detail || '登录失败', 'error');
            }
        } catch (error) {
            showAuthMessage('网络错误，请稍后重试', 'error');
        }
    }
    
    async function handleRegister(e) {
        e.preventDefault();
        const email = document.getElementById('registerEmail').value;
        const password = document.getElementById('registerPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        
        if (password !== confirmPassword) {
            showAuthMessage('两次输入的密码不一致', 'error');
            return;
        }
        
        try {
            showAuthMessage('正在注册...', 'info');
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });
            const data = await response.json();
            if (response.ok) {
                accessToken = data.access_token;
                currentUser = { email: data.user_email };
                localStorage.setItem('access_token', accessToken);
                showAuthMessage('注册成功！', 'success');
                setTimeout(() => showChatInterface(), 1000);
            } else {
                showAuthMessage(data.detail || '注册失败', 'error');
            }
        } catch (error) {
            showAuthMessage('网络错误，请稍后重试', 'error');
        }
    }
    
    async function verifyTokenAndShowChat() {
        try {
            const response = await fetch('/api/me', { headers: { 'Authorization': `Bearer ${accessToken}` } });
            if (response.ok) {
                currentUser = await response.json();
                showChatInterface();
            } else {
                handleLogout();
            }
        } catch (error) {
            handleLogout();
        }
    }
    
    function showAuthInterface() {
        authContainer.classList.remove('hidden');
        chatApp.classList.add('hidden');
    }
    
    function showChatInterface() {
        authContainer.classList.add('hidden');
        chatApp.classList.remove('hidden');
        userEmail.textContent = currentUser.email;
        clearChatHistory();
        loadChatSessions();
        connectWebSocket();
    }
    
    function handleLogout() {
        localStorage.removeItem('access_token');
        accessToken = null;
        currentUser = null;
        currentSessionId = null;
        chatSessions = [];
        if (ws) {
            ws.close();
            ws = null;
        }
        clearChatHistory();
        clearChatHistoryList();
        showAuthInterface();
    }
    
    function startNewChat() {
        currentSessionId = null;
        clearChatHistory();
        document.querySelectorAll('.chat-history-item').forEach(item => item.classList.remove('active'));
        questionInput.focus();
    }
    
    async function loadChatSessions() {
        try {
            const response = await fetch('/api/conversations', { headers: { 'Authorization': `Bearer ${accessToken}` } });
            if (response.ok) {
                const data = await response.json();
                chatSessions = data.conversations || [];
                renderChatSessionsList();
            }
        } catch (error) {
            console.error('加载聊天会话错误:', error);
        }
    }
    
    function renderChatSessionsList() {
        chatHistoryList.innerHTML = '';
        if (chatSessions.length === 0) {
            chatHistoryList.innerHTML = '<div style="padding: 16px; text-align: center; color: #8e8ea0; font-size: 14px;">暂无聊天记录</div>';
            return;
        }
        chatSessions.forEach(session => {
            const sessionItem = document.createElement('div');
            sessionItem.className = 'chat-history-item';
            sessionItem.dataset.sessionId = session.id;
            
            const formatTime = (isoString) => {
                const date = new Date(isoString);
                const now = new Date();
                const diffMs = now - date;
                const diffMins = Math.floor(diffMs / 60000);
                if (diffMins < 1) return '刚刚';
                if (diffMins < 60) return `${diffMins}分钟前`;
                const diffHours = Math.floor(diffMs / 3600000);
                if (diffHours < 24) return `${diffHours}小时前`;
                const diffDays = Math.floor(diffMs / 86400000);
                if (diffDays < 7) return `${diffDays}天前`;
                return date.toLocaleDateString();
            };
            
            sessionItem.innerHTML = `
                <div class="chat-content">
                    <div class="chat-title">${session.title}</div>
                    <div class="chat-preview">${session.preview}</div>
                    <div class="chat-time">${formatTime(session.updated_at || session.created_at)} • ${session.message_count || ''}条消息</div>
                </div>
                <div class="chat-actions">
                    <button class="delete-session-btn" title="删除对话">🗑️</button>
                </div>
            `;
            
            sessionItem.querySelector('.chat-content').addEventListener('click', () => loadChatSession(session));
            sessionItem.querySelector('.delete-session-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteChatSession(session);
            });
            chatHistoryList.appendChild(sessionItem);
        });
    }
    
    async function loadChatSession(session) {
        // [FIX] 如果点击的是当前已加载的会话，则不执行任何操作
        if (currentSessionId === session.id) {
            console.log("已经是当前会话，无需重新加载。");
            return;
        }
        currentSessionId = session.id;
        
        document.querySelectorAll('.chat-history-item').forEach(item => item.classList.remove('active'));
        document.querySelector(`[data-session-id="${session.id}"]`).classList.add('active');
        
        clearChatHistory();
        
        try {
            const apiEndpoint = session.session_type === 'chat_session' ? `/api/chat-sessions/${session.id}/messages` : `/api/chat-sessions/${session.id}/messages`; // 统一使用新API
            const response = await fetch(apiEndpoint, { headers: { 'Authorization': `Bearer ${accessToken}` } });
            
            if (response.ok) {
                const data = await response.json();
                data.messages.forEach(message => addMessage(message.content, message.role === 'user' ? 'user' : 'bot', message.id));
            } else {
                addMessage('加载历史消息失败', 'status');
            }
        } catch (error) {
            addMessage('加载历史消息失败', 'status');
        }
    }
    
    function clearChatHistoryList() {
        chatHistoryList.innerHTML = '';
    }
    
    function showAuthMessage(message, type) {
        authMessage.textContent = message;
        authMessage.className = `auth-message ${type}`;
        authMessage.style.display = 'block';
    }
    
    function clearAuthMessage() {
        authMessage.style.display = 'none';
    }
    
    function clearChatHistory() {
        chatContainer.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">🤖</div>
                <h2>欢迎使用AI助手</h2>
                <p>我是AI-Jay，随时准备为您服务。您可以问我任何问题！</p>
            </div>
        `;
        currentBotMessageDiv = null;
        currentUserMessageDiv = null;
    }
    
    function connectWebSocket() {
        if (!accessToken) return;
        if (ws && ws.readyState === WebSocket.OPEN) return;
        
        ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = () => {
            connectionStatus.textContent = '正在认证...';
            ws.send(JSON.stringify({ type: 'auth', token: accessToken }));
        };
        
        ws.onmessage = (event) => handleWebSocketMessage(JSON.parse(event.data));
        
        ws.onclose = () => {
            connectionStatus.textContent = '❌ 连接断开，3秒后尝试重连...';
            connectionStatus.className = 'connection-status disconnected';
            sendButton.disabled = true;
            setTimeout(() => { if (accessToken && currentUser) connectWebSocket(); }, 3000);
        };
        
        ws.onerror = (error) => console.error('WebSocket错误:', error);
    }
    
    function handleWebSocketMessage(event) {
        switch (event.type) {
            case 'auth_success':
                connectionStatus.textContent = '✅ 已连接';
                connectionStatus.className = 'connection-status connected';
                sendButton.disabled = false;
                break;
            case 'processing':
                if (event.data.session_id && !currentSessionId) {
                    currentSessionId = event.data.session_id;
                }
                addStatusMessage(`[${event.data.message}]`);
                break;
            case 'generation_start':
                clearStatusMessages();
                currentBotMessageDiv = addMessage('', 'bot');
                break;
            case 'generation_chunk':
                if (currentBotMessageDiv) {
                    currentBotMessageDiv.querySelector('.message-content').textContent += event.data.chunk;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                break;
            case 'complete':
                const { user_message_id, ai_message_id } = event.data;
                if (user_message_id && currentUserMessageDiv) {
                    addDeleteButtonToMessage(currentUserMessageDiv, user_message_id);
                }
                if (ai_message_id && currentBotMessageDiv) {
                    addDeleteButtonToMessage(currentBotMessageDiv, ai_message_id);
                }
                currentUserMessageDiv = null;
                currentBotMessageDiv = null;
                sendButton.disabled = false;
                sendButton.innerHTML = '<span class="send-icon">➤</span>';
                if (event.data.session_id) {
                    currentSessionId = event.data.session_id;
                }
                loadChatSessions();
                break;
            case 'error':
                clearStatusMessages();
                addMessage(`[错误]: ${event.data.error}`, 'status');
                sendButton.disabled = false;
                sendButton.innerHTML = '<span class="send-icon">➤</span>';
                break;
        }
    }

    function addMessage(content, type, messageId = null) {
        const welcomeMessage = chatContainer.querySelector('.welcome-message');
        if (welcomeMessage) welcomeMessage.remove();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        if (messageId) messageDiv.dataset.messageId = messageId;
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${type === 'user' ? '👤' : (type === 'bot' ? '🤖' : 'ℹ️')}</div>
            <div class="message-content">${content}</div>
        `;
        
        if ((type === 'user' || type === 'bot') && messageId) {
            addDeleteButtonToMessage(messageDiv, messageId);
        }
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return messageDiv;
    }

    function addDeleteButtonToMessage(messageDiv, messageId) {
        if (!messageDiv || !messageId) return;
        messageDiv.dataset.messageId = messageId;
        
        if (messageDiv.querySelector('.message-actions')) return; // 避免重复添加

        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-message-btn';
        deleteBtn.innerHTML = '🗑️';
        deleteBtn.title = '删除消息';
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteMessage(messageId, messageDiv);
        });
        actionsDiv.appendChild(deleteBtn);
        messageDiv.appendChild(actionsDiv);
    }
    
    function addStatusMessage(content) {
        const welcomeMessage = chatContainer.querySelector('.welcome-message');
        if (welcomeMessage) welcomeMessage.remove();
        const messageDiv = addMessage(content, 'status');
        messageDiv.classList.add('temp-status');
        return messageDiv;
    }
    
    function clearStatusMessages() {
        chatContainer.querySelectorAll('.temp-status').forEach(m => m.remove());
    }
    
    function sendQuestion() {
        const question = questionInput.value.trim();
        if (!question || !ws || ws.readyState !== WebSocket.OPEN) return;
        
        currentUserMessageDiv = addMessage(question, 'user');
        
        const messageData = { type: 'question', content: question };
        if (currentSessionId) messageData.session_id = currentSessionId;
        
        ws.send(JSON.stringify(messageData));
        
        questionInput.value = '';
        sendButton.disabled = true;
        sendButton.innerHTML = '⏳';
        
        if (!currentSessionId) {
            document.querySelectorAll('.chat-history-item').forEach(item => item.classList.remove('active'));
        }
    }
    
    async function deleteChatSession(session) {
        if (!confirm(`确定要删除对话"${session.title}"吗？`)) return;
        
        try {
            const response = await fetch(`/api/chat-sessions/${session.id}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${accessToken}` },
            });
            if (response.ok) {
                if (currentSessionId === session.id) startNewChat();
                await loadChatSessions();
            } else {
                const data = await response.json();
                alert(`删除失败: ${data.detail || '未知错误'}`);
            }
        } catch (error) {
            alert('删除失败，请稍后重试');
        }
    }
    
    async function deleteMessage(messageId, messageElement) {
        if (!confirm('确定要删除这条消息吗？')) return;
        
        try {
            const response = await fetch(`/api/messages/${messageId}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${accessToken}` },
            });
            if (response.ok) {
                messageElement.remove();
                await loadChatSessions();
            } else {
                const data = await response.json();
                alert(`删除失败: ${data.detail || '未知错误'}`);
            }
        } catch (error) {
            alert('删除失败，请稍后重试');
        }
    }
    
    init();
})();
```

## `static/style.css`

```css
/* static/style.css */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f7f7f8;
    height: 100vh;
    overflow: hidden;
}

.hidden {
    display: none !important;
}

/* 认证界面样式 */
.auth-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.auth-form {
    background: white;
    border-radius: 12px;
    padding: 40px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    width: 100%;
    max-width: 400px;
}

.auth-form h1 {
    color: #333;
    text-align: center;
    margin-bottom: 30px;
    font-size: 24px;
}

.auth-tabs {
    display: flex;
    margin-bottom: 30px;
    border-bottom: 1px solid #e5e5e5;
}

.auth-tab {
    flex: 1;
    padding: 12px;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 16px;
    color: #666;
    border-bottom: 2px solid transparent;
    transition: all 0.3s;
}

.auth-tab.active {
    color: #667eea;
    border-bottom-color: #667eea;
}

.auth-tab:hover {
    color: #667eea;
}

.form-group {
    margin-bottom: 20px;
}

.form-group label {
    display: block;
    margin-bottom: 5px;
    color: #333;
    font-weight: 500;
}

.form-group input {
    width: 100%;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: 16px;
    transition: border-color 0.3s;
}

.form-group input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.auth-button {
    width: 100%;
    padding: 12px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
    font-weight: 500;
    transition: transform 0.2s;
}

.auth-button:hover {
    transform: translateY(-1px);
}

.auth-button:disabled {
    background: #ccc;
    cursor: not-allowed;
    transform: none;
}

.auth-message {
    margin-top: 15px;
    padding: 10px;
    border-radius: 6px;
    text-align: center;
    font-size: 14px;
}

.auth-message.success {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.auth-message.error {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

/* 聊天应用布局 */
.chat-app {
    display: flex;
    height: 100vh;
}

/* 左侧边栏 */
.sidebar {
    width: 260px;
    background: #202123;
    color: white;
    display: flex;
    flex-direction: column;
    border-right: 1px solid #4d4d4f;
}

.sidebar-header {
    padding: 16px;
    border-bottom: 1px solid #4d4d4f;
}

.new-chat-button {
    width: 100%;
    padding: 12px;
    background: transparent;
    color: white;
    border: 1px solid #4d4d4f;
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    transition: background-color 0.2s;
}

.new-chat-button:hover {
    background: #40414f;
}

.new-chat-button .icon {
    font-size: 16px;
    font-weight: bold;
}

.chat-history {
    flex: 1;
    overflow-y: auto;
}

.chat-history-header {
    padding: 16px;
    border-bottom: 1px solid #4d4d4f;
}

.chat-history-header h3 {
    font-size: 14px;
    color: #8e8ea0;
    font-weight: 500;
}

.chat-history-list {
    padding: 8px;
}

.chat-history-item {
    display: flex;
    align-items: center;
    padding: 12px;
    margin-bottom: 4px;
    border-radius: 6px;
    transition: background-color 0.2s;
    border: 1px solid transparent;
    position: relative;
}

.chat-history-item:hover {
    background: #40414f;
}

.chat-history-item.active {
    background: #40414f;
    border-color: #10a37f;
}

.chat-content {
    flex: 1;
    cursor: pointer;
    min-width: 0; /* 允许文本截断 */
}

.chat-history-item .chat-title {
    font-size: 14px;
    color: white;
    margin-bottom: 4px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.chat-history-item .chat-preview {
    font-size: 12px;
    color: #8e8ea0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.chat-history-item .chat-time {
    font-size: 11px;
    color: #6f6f6f;
    margin-top: 4px;
}

.chat-actions {
    display: flex;
    align-items: center;
    opacity: 0;
    transition: opacity 0.2s;
    margin-left: 8px;
}

.chat-history-item:hover .chat-actions {
    opacity: 1;
}

.delete-session-btn {
    background: none;
    border: none;
    color: #8e8ea0;
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    font-size: 14px;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
}

.delete-session-btn:hover {
    background: #ff4444;
    color: white;
    transform: scale(1.1);
}

.sidebar-footer {
    padding: 16px;
    border-top: 1px solid #4d4d4f;
}

.user-info {
    display: flex;
    align-items: center;
    gap: 12px;
}

.user-avatar {
    width: 32px;
    height: 32px;
    background: #40414f;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

.user-details {
    flex: 1;
}

.user-email {
    font-size: 14px;
    color: white;
    margin-bottom: 4px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.logout-button {
    background: none;
    border: none;
    color: #8e8ea0;
    cursor: pointer;
    font-size: 12px;
    padding: 0;
    transition: color 0.2s;
}

.logout-button:hover {
    color: white;
}

/* 主聊天区域 */
.main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: white;
}

.chat-header {
    padding: 16px 24px;
    border-bottom: 1px solid #e5e5e5;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: white;
}

.chat-header h1 {
    font-size: 20px;
    color: #333;
    margin: 0;
}

.connection-status {
    padding: 6px 12px;
    border-radius: 16px;
    font-size: 12px;
    font-weight: 500;
}

.connected {
    background-color: #d4edda;
    color: #155724;
}

.disconnected {
    background-color: #f8d7da;
    color: #721c24;
}

.chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    background: #f7f7f8;
}

.welcome-message {
    text-align: center;
    padding: 60px 20px;
    color: #666;
}

.welcome-message .welcome-icon {
    font-size: 48px;
    margin-bottom: 16px;
}

.welcome-message h2 {
    font-size: 24px;
    margin-bottom: 8px;
    color: #333;
}

.welcome-message p {
    font-size: 16px;
    color: #666;
}

.message {
    margin-bottom: 24px;
    display: flex;
    gap: 12px;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
    position: relative;
}

.message:hover .message-actions {
    opacity: 1;
}

.message-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}

.user-message .message-avatar {
    background: #667eea;
    color: white;
}

.bot-message .message-avatar {
    background: #10a37f;
    color: white;
}

.status-message .message-avatar {
    background: #f59e0b;
    color: white;
}

.message-content {
    flex: 1;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.5;
    word-wrap: break-word;
}

.user-message .message-content {
    background: #667eea;
    color: white;
    margin-left: auto;
    max-width: 70%;
}

.bot-message .message-content {
    background: white;
    color: #333;
    border: 1px solid #e5e5e5;
    max-width: 85%;
}

.status-message .message-content {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fcd34d;
    font-style: italic;
    text-align: center;
    max-width: 100%;
}

.message-actions {
    position: absolute;
    top: 8px;
    right: 8px;
    opacity: 0;
    transition: opacity 0.2s;
    display: flex;
    gap: 4px;
}

.delete-message-btn {
    background: rgba(0, 0, 0, 0.1);
    border: none;
    color: #666;
    cursor: pointer;
    padding: 4px 6px;
    border-radius: 4px;
    font-size: 12px;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
}

.delete-message-btn:hover {
    background: #ff4444;
    color: white;
    transform: scale(1.1);
}

.user-message .delete-message-btn {
    background: rgba(255, 255, 255, 0.2);
    color: rgba(255, 255, 255, 0.8);
}

.user-message .delete-message-btn:hover {
    background: #ff4444;
    color: white;
}

.input-container {
    padding: 24px;
    background: white;
    border-top: 1px solid #e5e5e5;
}

.input-wrapper {
    max-width: 800px;
    margin: 0 auto;
    display: flex;
    gap: 12px;
    align-items: flex-end;
}

#questionInput {
    flex: 1;
    padding: 12px 16px;
    border: 1px solid #d1d5db;
    border-radius: 24px;
    font-size: 16px;
    resize: none;
    min-height: 48px;
    max-height: 120px;
    transition: border-color 0.3s;
}

#questionInput:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

#sendButton {
    width: 48px;
    height: 48px;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color 0.3s, transform 0.2s;
    flex-shrink: 0;
}

#sendButton:hover:not(:disabled) {
    background: #5a67d8;
    transform: scale(1.05);
}

#sendButton:disabled {
    background: #d1d5db;
    cursor: not-allowed;
    transform: none;
}

.send-icon {
    font-size: 16px;
    margin-left: 2px;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .sidebar {
        width: 100%;
        position: fixed;
        top: 0;
        left: -100%;
        z-index: 1000;
        transition: left 0.3s;
    }
    
    .sidebar.open {
        left: 0;
    }
    
    .main-content {
        width: 100%;
    }
    
    .message {
        margin-bottom: 16px;
    }
    
    .user-message .message-content,
    .bot-message .message-content {
        max-width: 85%;
    }
}
```

## `stream_chat_bot.db`

```
SQLite format 3   @     D   
           	                                                 D .v
   	
b &

'#

b                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    N	)eindexix_messages_idmessages
CREATE INDEX ix_messages_id ON messages (id)
gtablemessagesmessages	CREATE TABLE messages (
	id INTEGER NOT NULL, 
	chat_session_id INTEGER NOT NULL, 
	role VARCHAR NOT NULL, 
	content TEXT NOT NULL, 
	created_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(chat_session_id) REFERENCES chat_sessions (id)
)b3'yindexix_conversations_idconversationsCREATE INDEX ix_conversations_id ON conversations (id)''AtableconversationsconversationsCREATE TABLE conversations (
	id INTEGER NOT NULL, 
	user_id INTEGER NOT NULL, 
	question TEXT NOT NULL, 
	answer TEXT NOT NULL, 
	created_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id)
)b3'yindexix_chat_sessions_idchat_sessionsCREATE INDEX ix_chat_sessions_id ON chat_sessions (id) ''?tablechat_sessionschat_sessionsCREATE TABLE chat_sessions (
	id INTEGER NOT NULL, 
	user_id INTEGER NOT NULL, 
	title VARCHAR NOT NULL, 
	created_at DATETIME, 
	updated_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id)
)B#Yindexix_users_idusersCREATE INDEX ix_users_id ON users (id)R)sindexix_users_emailusersCREATE UNIQUE INDEX ix_users_email ON users (email)W
tableusersusersCREATE TABLE users (
	id INTEGER NOT NULL, 
	email VARCHAR NOT NULL, 
	hashed_password VARCHAR NOT NULL, 
	is_active BOOLEAN, 
	created_at DATETIME, 
	updated_at DATETIME, 
	PRIMARY KEY (id)
)
    v                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 +	AAxiaohei@123.com$2b$12$3ArZEZFYD0Re4MxUxdPpv..F9hrvfocwivk01o/4hdAfoADRPOSB22025-09-05 01:37:02.4120242025-09-05 01:37:02.412024 +	AAxiaobai@123.com$2b$12$PisXgqq89W5JDEKnVuCr6uJNEmbtMAJASclqEnjwheQo6Z7nEkG.K2025-09-05 01:31:35.1661722025-09-05 01:31:35.166172
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 +xiaohei@123.com+	xiaobai@123.com
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              		
( 
 Z	


p                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                OL	 	1AA新英格兰地区2025-09-05 03:05:10.4131992025-09-05 03:05:20.267320M 1AA孟德尔的成就2025-09-05 02:46:14.3201762025-09-05 02:46:27.875412   	AA你好2025-09-05 02:45:26.8268712025-[ 	OAA北美十三州今天还在吗？2025-09-05 03:04:30.6181472025-09-05 03:04:40.962442U 	CAA欧洲的宗教改革运行2025-09-05 02:29:28.8793792025-09-05 02:29:43.365722O 	7AA文艺复兴的开启2025-09-05 01:40:02.0537252025-09-05 01:40:11.203324O 	7AA讲个小白的笑话2025-09-05 01:39:40.5100332025-09-05 01:39:45.061042S =AA大航海时代的起源2025-09-05 01:37:21.3670282025-09-05 01:38:37.983700
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            				   		
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
n                                                                                                                                                                                                                                                                                                                                                            : 7Auser讲个小白的笑话2025-09-05 01:39:40.519595 	7Aassistant🌍 你的观察很敏锐！荷兰确实在大航海时代中后期扮演了关键角色，但严格来说，他们并非“开启”大航海时代的先驱者。以下是更准确的解释：

---

### 🇳🇱 **荷兰的参与时间与角色**  
1. **相对较晚的入场**：  
   - 大航海时代始于15世纪的葡萄牙和西班牙（如迪亚斯、哥伦布、达·伽马的航行），而荷兰的大规模海上扩张主要集中在 **16世纪末至17世纪**。  
   - 原因：荷兰当时受西班牙哈布斯堡王朝统治，1568年爆发独立战争后，才逐渐摆脱束缚走向海洋。  

2. **“海上马车夫”的崛起**：  
   - 荷兰凭借先进的造船技术、金融体系（如世界上第一家股份制公司——荷兰东印度公司，1602年成立）和高效贸易网络，后来居上。  
   - 17世纪时，荷兰取代葡萄牙和西班牙，成为全球海洋贸易的霸主，垄断了东南亚香料贸易，并建立了殖民地（如台湾、印尼、南非等）。  

---

### ⚓ **为何常被误认为“开启者”？**  
- **历史视角差异**：荷兰在航海技术、商业模式和殖民效率上的创新如此突出，以至于其成就常被单独强调，但时间线上并非最早。  
- **文化传播**：荷兰的航海故事（如探险家威廉·巴伦支、阿贝尔·塔斯曼）和东印度公司的全球影响深入人心。  

---

### 🌟 总结：  
- **先驱者**：葡萄牙、西班牙（15-16世纪）。  
- **巅峰代表**：荷兰（17世纪）、英国（后期）。  
- 荷兰更像是大航海时代的“优化者”和“商业王者”，而非开启者。  

如果需要具体案例或荷兰航海家的故事，可以告诉我哦！🚢✨2025-09-05 01:38:38.006328B 	IAuser荷兰人开启了大航海？2025-09-05 01:38:25.606664O 	WAassistant🌊 你好！关于“大航海时代”（Age of Discovery）的起源，通常指15世纪至17世纪欧洲国家大规模探索和开拓全球航线的历史时期。其起源主要与以下因素相关：

---

### 1️⃣ **核心动因**  
- **经济需求**：欧洲对亚洲香料、丝绸和黄金的渴望，希望摆脱阿拉伯和威尼斯商人的垄断，寻找直接贸易路线。  
- **政治竞争**：葡萄牙、西班牙等国家希望通过海外扩张增强国力，争夺资源和领土。  
- **宗教传播**：基督教（尤其是天主教）势力试图向外传播信仰，对抗伊斯兰世界的影响。  

---

### 2️⃣ **关键技术与条件**  
- **航海技术进步**：如罗盘、星盘、卡拉维尔帆船（Caravel）的应用，使远洋航行成为可能。  
- **地理知识增长**：受文艺复兴影响，地圆说逐渐被接受，地图绘制更加精确。  
- **王室支持**：葡萄牙恩里克王子、西班牙伊莎贝拉女王等统治者资助探险活动。  

---

### 3️⃣ **标志性事件**  
- **1415年**：葡萄牙占领北非休达，开启海外扩张。  
- **1488年**：迪亚斯抵达非洲好望角。  
- **1492年**：哥伦布横渡大西洋，误抵美洲（为西班牙开辟新大陆）。  
- **1498年**：达·伽马绕过好望角抵达印度，建立欧洲至亚洲的海上航线。  

---

### 4️⃣ **影响**  
- 全球贸易网络初步形成（如“哥伦布大交换”）。  
- 欧洲殖民主义兴起，同时也导致原住民文化的破坏。  
- 世界从孤立走向互联，奠定现代全球化基础。  

如果需要更具体的细节或延伸内容，可以告诉我哦！🚢✨2025-09-05 01:37:34.026740< 	=Auser大航海时代的起源2025-09-05 01:37   
   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

				
   n n                                                                                                                                                                                                                                                                                                                                                            : 7Auser讲个小白的笑话2025-09-05 01:39:40.519595 	7Aassistant🌍 你的观察很敏锐！荷兰确实在大航海时代中后期扮演了关键角色，但严格来说，他们并非“开启”大航海时代的先驱者。以下是更准确的解释：

---

### 🇳🇱 **荷兰的参与时间与角色**  
1. **相对较晚的入场**：  
   - 大航海时代始于15世纪的葡萄牙和西班牙（如迪亚斯、哥伦布、达·伽马的航行），而荷兰的大规模海上扩张主要集中在 **16世纪末至17世纪**。  
   - 原因：荷兰当时受西班牙哈布斯堡王朝统治，1568年爆发独立战争后，才逐渐摆脱束缚走向海洋。  

2. **“海上马车夫”的崛起**：  
   - 荷兰凭借先进的造船技术、金融体系（如世界上第一家股份制公司——荷兰东印度公司，1602年成立）和高效贸易网络，后来居上。  
   - 17世纪时，荷兰取代葡萄牙和西班牙，成为全球海洋贸易的霸主，垄断了东南亚香料贸易，并建立了殖民地（如台湾、印尼、南非等）。  

---

### ⚓ **为何常被误认为“开启者”？**  
- **历史视角差异**：荷兰在航海技术、商业模式和殖民效率上的创新如此突出，以至于其成就常被单独强调，但时间线上并非最早。  
- **文化传播**：荷兰的航海故事（如探险家威廉·巴伦支、阿贝尔·塔斯曼）和东印度公司的全球影响深入人心。  

---

### 🌟 总结：  
- **先驱者**：葡萄牙、西班牙（15-16世纪）。  
- **巅峰代表**：荷兰（17世纪）、英国（后期）。  
- 荷兰更像是大航海时代的“优化者”和“商业王者”，而非开启者。  

如果需要具体案例或荷兰航海家的故事，可以告诉我哦！🚢✨2025-09-05 01:38:38.006328B 	IAuser荷兰人开启了大航海？2025-09-05 01:38:25.606664O 	WAassistant🌊 你好！关于“大航海时代”（Age of Discovery）的起源，通常指15世纪至17世纪欧洲国家大规模探索和开拓全球航线的历史时期。其起源主要与以下因素相关：

---

### 1️⃣ **核心动因**  
- **经济需求**：欧洲对亚洲香料、丝绸和黄金的渴望，希望摆脱阿拉伯和威尼斯商人的垄断，寻找直接贸易路线。  
- **政治竞争**：葡萄牙、西班牙等国家希望通过海外扩张增强国力，争夺资源和领土。  
- **宗教传播**：基督教（尤其是天主教）势力试图向外传播信仰，对抗伊斯兰世界的影响。  

---

### 2️⃣ **关键技术与条件**  
- **航海技术进步**：如罗盘、星盘、卡拉维尔帆船（Caravel）的应用，使远洋航行成为可能。  
- **地理知识增长**：受文艺复兴影响，地圆说逐渐被接受，地图绘制更加精确。  
- **王室支持**：葡萄牙恩里克王子、西班牙伊莎贝拉女王等统治者资助探险活动。  

---

### 3️⃣ **标志性事件**  
- **1415年**：葡萄牙占领北非休达，开启海外扩张。  
- **1488年**：迪亚斯抵达非洲好望角。  
- **1492年**：哥伦布横渡大西洋，误抵美洲（为西班牙开辟新大陆）。  
- **1498年**：达·伽马绕过好望角抵达印度，建立欧洲至亚洲的海上航线。  

---

### 4️⃣ **影响**  
- 全球贸易网络初步形成（如“哥伦布大交换”）。  
- 欧洲殖民主义兴起，同时也导致原住民文化的破坏。  
- 世界从孤立走向互联，奠定现代全球化基础。  

如果需要更具体的细节或延伸内容，可以告诉我哦！🚢✨2025-09-05 01:37:34.026740< 	=Auser大航海时代的起源2025-09-05 01:37:21.388502
    c'		U	                                                                                                                                                                                                                                                                                                                                                                                                                1
 %Auser大麻丁岛2025-09-05 02:30:33.5187441 %Auser大麻丁岛2025-09-05 02:30:26.628558 ;Aassistant欧洲的宗教改革（16世纪）是一场深刻改变基督教和欧洲社会格局的运动，其核心是挑战罗马天主教会的权威，推动新教教派的兴起。以下是关键内容：

---

### **1. 背景与原因**
- **教会腐败**：出售赎罪券、神职人员世俗化、道德堕落引发不满。
- **思想解放**：文艺复兴倡导个人理性，印刷术加速思想传播。
- **政治因素**：君主和贵族希望削弱教廷权力，争夺教会财富和控制权。

---

### **2. 关键人物与事件**
- **马丁·路德（Martin Luther）**：  
  - 1517年发表《九十五条论纲》，质疑赎罪券，强调“因信称义”（信仰即可得救）。  
  - 翻译德语《圣经》，推动宗教本土化。  
- **约翰·加尔文（John Calvin）**：  
  - 在日内瓦建立神权政府，主张“预定论”和严谨的生活纪律。  
- **英国宗教改革**：  
  - 亨利八世因离婚问题与教廷决裂，自立英国国教会（圣公会）。

---

### **3. 影响与后果**
- **宗教分裂**：天主教vs.新教（路德宗、加尔文宗、圣公会等）。  
- **战争与冲突**：引发三十年战争（1618-1648），《威斯特伐利亚和约》确立教派平等原则。  
- **文化与社会**：  
  - 教育普及（新教鼓励阅读《圣经》）。  
  - 促进资本主义精神（如加尔文主义与商业伦理）。  
  - 削弱教皇权威，强化民族国家权力。

---

### **4. 天主教会的回应**
- **反宗教改革**：通过特伦特会议（1545-1563）整顿内部，成立耶稣会等传教组织，重新争取信徒。

---

如果需要更具体的细节（如某国家或人物的深入分析），可以告诉我哦！ 😊2025-09-05 02:29:43.368729@
 CAuser欧洲的宗教改革运行2025-09-05 02:29:43.355946@	 CAuser欧洲的宗教改革运行2025-09-05 02:29:28.896228
 QAassistant文艺复兴是欧洲历史上一个重要的文化运动，始于14世纪的意大利，逐渐扩展到整个欧洲，持续到17世纪初期。它标志着从中世纪到近代的过渡，核心思想是“人文主义”，强调对古典文化（古希腊和古罗马）的复兴，以及对人类理性、创造力和个体价值的重视。

### 关键特点：
1. **人文主义**：关注人的潜能和成就，而非宗教教条。
2. **艺术与科学结合**：艺术家（如达·芬奇、米开朗基罗）研究解剖学、透视法，追求现实主义。
3. **古典复兴**：重新发掘古希腊和罗马的文学、哲学和艺术。
4. **印刷术的推广**：古登堡印刷术（15世纪）加速了知识的传播。

### 重要人物：
- **但丁**（《神曲》）  
- **达·芬奇**（《蒙娜丽莎》《最后的晚餐》）  
- **米开朗基罗**（西斯廷教堂壁画）  
- **莎士比亚**（文学戏剧）  

### 影响：
- 推动了宗教改革、科学革命和地理大发现。  
- 为现代西方文化、艺术和思想奠定了基础。  

需要更具体的内容吗？2025-09-05 01:40:11.203324: 7Auser文艺复兴的开启2025-09-05 01:40:02.063885 kAassistant当然！这里有一个简单又可爱的笑话，希望能让你开心一下 😄：

---

🤖 小白：“妈妈，为什么冰箱会说话？”  
👩 妈妈：“傻孩子，冰箱怎么会说话呢？”  
🤖 小白：“可是它刚才对我说‘门没关好’！”  

---

希望这个笑话能让你笑一笑！如果需要更多，随时告诉我哦~2025-09-05 01:39:45.084332
  OO                                                                                                                                                                                                                                                                                                            gAassistant加拿大的枫糖（Maple Syrup）是加拿大最具代表性的特产之一，以其天然甜味和独特风味闻名全球。以下是关于u !Aassistant🇨🇦 你好！关于加拿大的枫叶卡（Mapl- Aassistant当然还在！😊 不过它们已经不再是“北美十三州”的原始形态了。这十三个英国殖民地通过**美国独立战争（1775-1783年| /A	assistant你好！😊 关于**新英格兰地区**（New England），我可以为你介绍一些基本信息：

**📍 地理位置**  
位于美国东北部，包括6个州：  
- 缅因州（Maine）  
- 新罕布什尔州（New Hampshire）  
- 佛蒙特州（Vermont）  
- 马萨诸塞州（Massachusetts）  
- 罗德岛州（Rhode Island）  
- 康涅狄格州（Connecticut）  

**🍁 特色与亮点**  
- **自然风光**：秋季红叶、阿卡迪亚国家公园、白山森林等。  
- **历史与文化**：美国独立运动发源地，拥有哈佛、MIT等名校，以及波士顿、普罗维登斯等城市。  
- **美食**：海鲜（尤其是龙虾和蛤蜊汤）、枫糖浆、新英格兰风味的 clam chowder（蛤蜊浓汤）。  
- **四季分明**：冬季滑雪胜地，夏季海滨度假，秋季赏枫叶。  

**🎯 常见用途**  
旅行规划、留学选择、历史文化研究或商业投资参考？需要我展开哪方面的信息呢？ 😊2025-09-05 03:05:20.278386[ mAassistant孟德尔的成就主要体现在**遗传学领域**，他被誉为“现代遗传学之父”🌱。以下是他的核心贡献：

1. **豌豆实验（1856-1863）**  
   通过长达8年的豌豆杂交实验，系统性地研究了7对相对性状（如种子形状、花色等）的遗传规律，奠定了遗传学基础。

2. **提出遗传三大定律**  
   - **分离定律**：成对的遗传因子（等位基因）在形成配子时分离，进入不同生殖细胞。  
   - **自由组合定律**：不同性状的遗传因子独立分配，互不干扰。  
   - （注：连锁遗传规律由摩尔根后续发现，但孟德尔的研究为后者提供了基础。）

3. **引入“显性”与“隐性”概念**  
   首次用数学方法（3:1的性状分离比）揭示了性状传递的量化规律，证明了遗传因子的离散性。

4. **发表划时代论文**  
   1866年发表《植物杂交试验》，但当时未被学界重视，直到1900年才被三位科学家重新发现并证实。

💡 **意义**：  
孟德尔的工作首次用科学实验证明了遗传的规律性，打破了当时“混合遗传”的错误观念，为现代遗传学、分子生物学和进化论提供了关键基础。他的发现也体现了跨学科研究（生物学+统计学）的重要性。

他的成就提醒我们：重大科学突破可能需要时间才能被认可，但坚持真理终会照亮人类认知的进程✨。2025-09-05 02:46:27.8879917 1Auser孟德尔的成就2025-09-05 02:46:14.332874   9Aassistant你好！😊 我是AI-Jay，很高兴为你提供帮助！有什么我可以帮你的2025-09-05 02:45:31.829394  Auser你好2025-09-05 02:45:26.851537  ^ mAassistant您好！😊 关于“大麻丁岛”，目前并没有一个广为人知或官方确认的地点使用这个名称。如果您指的是某个特定地区、旅游景点，或者可能是名称的误写（例如“大堡礁”、“大溪地”等），请提供更多信息，我会尽力帮助您！ 

如果是与大麻相关的内容，请注意：在中国，大麻属于违禁品，种植、销售或吁 9A	assistant你好！😊 我是AI-Jay，很高兴为你提供帮助！有什么我可以帮你F OAuser7 1A	user新英格兰地区2025-09-05 03:05:10.430514
```

## `view_database.py`

```python
#!/usr/bin/env python3
# 数据库查看工具 - 适合数据库小白使用

from app.database import SessionLocal
from app.models import User, ChatSession, Message, Conversation
from sqlalchemy import text
import pandas as pd
from datetime import datetime

def show_table_info():
    """显示所有表的基本信息"""
    db = SessionLocal()
    try:
        print("🗄️ 数据库表信息")
        print("=" * 60)
        
        # 获取所有表名
        result = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        
        tables = [row[0] for row in result.fetchall()]
        
        for table in tables:
            # 获取表的记录数
            count_result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = count_result.scalar()
            
            print(f"📋 表名: {table}")
            print(f"   记录数: {count}")
            
            # 获取表结构
            structure_result = db.execute(text(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
            """))
            
            print("   字段:")
            for col_name, data_type, nullable in structure_result.fetchall():
                null_info = "可空" if nullable == "YES" else "非空"
                print(f"     - {col_name}: {data_type} ({null_info})")
            print()
            
    except Exception as e:
        print(f"❌ 获取表信息失败: {e}")
    finally:
        db.close()

def show_users():
    """显示用户表内容"""
    db = SessionLocal()
    try:
        print("👥 用户表 (users)")
        print("=" * 60)
        
        users = db.query(User).all()
        
        if not users:
            print("📭 用户表为空")
            return
        
        print(f"📊 总用户数: {len(users)}")
        print()
        
        for user in users:
            print(f"🆔 ID: {user.id}")
            print(f"📧 邮箱: {user.email}")
            print(f"✅ 激活状态: {'是' if user.is_active else '否'}")
            print(f"📅 注册时间: {user.created_at}")
            print(f"🔄 更新时间: {user.updated_at}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 查看用户表失败: {e}")
    finally:
        db.close()

def show_chat_sessions():
    """显示会话表内容"""
    db = SessionLocal()
    try:
        print("💬 会话表 (chat_sessions)")
        print("=" * 60)
        
        sessions = db.query(ChatSession).all()
        
        if not sessions:
            print("📭 会话表为空")
            return
        
        print(f"📊 总会话数: {len(sessions)}")
        print()
        
        for session in sessions:
            # 获取会话的消息数量
            message_count = db.query(Message).filter(Message.chat_session_id == session.id).count()
            
            print(f"🆔 会话ID: {session.id}")
            print(f"👤 用户ID: {session.user_id}")
            print(f"📝 标题: {session.title}")
            print(f"💬 消息数: {message_count}")
            print(f"📅 创建时间: {session.created_at}")
            print(f"🔄 更新时间: {session.updated_at}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 查看会话表失败: {e}")
    finally:
        db.close()

def show_messages(session_id=None, limit=10):
    """显示消息表内容"""
    db = SessionLocal()
    try:
        print("💭 消息表 (messages)")
        print("=" * 60)
        
        query = db.query(Message)
        if session_id:
            query = query.filter(Message.chat_session_id == session_id)
            print(f"🔍 筛选条件: 会话ID = {session_id}")
        
        messages = query.order_by(Message.created_at.desc()).limit(limit).all()
        
        if not messages:
            print("📭 消息表为空")
            return
        
        print(f"📊 显示最新 {len(messages)} 条消息:")
        print()
        
        for msg in messages:
            role_icon = "👤" if msg.role == "user" else "🤖"
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            
            print(f"🆔 消息ID: {msg.id}")
            print(f"💬 会话ID: {msg.chat_session_id}")
            print(f"{role_icon} 角色: {msg.role}")
            print(f"📝 内容: {content_preview}")
            print(f"📅 时间: {msg.created_at}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 查看消息表失败: {e}")
    finally:
        db.close()

def show_conversations(limit=10):
    """显示旧对话表内容"""
    db = SessionLocal()
    try:
        print("📜 旧对话表 (conversations)")
        print("=" * 60)
        
        conversations = db.query(Conversation).order_by(Conversation.created_at.desc()).limit(limit).all()
        
        if not conversations:
            print("📭 对话表为空")
            return
        
        print(f"📊 显示最新 {len(conversations)} 条对话:")
        print()
        
        for conv in conversations:
            question_preview = conv.question[:50] + "..." if len(conv.question) > 50 else conv.question
            answer_preview = conv.answer[:50] + "..." if len(conv.answer) > 50 else conv.answer
            
            print(f"🆔 对话ID: {conv.id}")
            print(f"👤 用户ID: {conv.user_id}")
            print(f"❓ 问题: {question_preview}")
            print(f"💡 回答: {answer_preview}")
            print(f"📅 时间: {conv.created_at}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 查看对话表失败: {e}")
    finally:
        db.close()

def show_user_activity(user_id):
    """显示特定用户的活动"""
    db = SessionLocal()
    try:
        print(f"👤 用户 {user_id} 的活动详情")
        print("=" * 60)
        
        # 获取用户信息
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            print(f"❌ 未找到用户ID {user_id}")
            return
        
        print(f"📧 用户邮箱: {user.email}")
        print(f"📅 注册时间: {user.created_at}")
        print()
        
        # 获取用户的会话
        sessions = db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
        print(f"💬 总会话数: {len(sessions)}")
        
        for session in sessions:
            message_count = db.query(Message).filter(Message.chat_session_id == session.id).count()
            print(f"  📁 {session.title} ({message_count}条消息)")
        
        # 获取用户的旧对话
        old_conversations = db.query(Conversation).filter(Conversation.user_id == user_id).count()
        print(f"📜 旧对话数: {old_conversations}")
        
    except Exception as e:
        print(f"❌ 查看用户活动失败: {e}")
    finally:
        db.close()

def main_menu():
    """主菜单"""
    while True:
        print("\n🗄️ 数据库查看工具")
        print("=" * 40)
        print("1. 📋 查看所有表信息")
        print("2. 👥 查看用户表")
        print("3. 💬 查看会话表")
        print("4. 💭 查看消息表")
        print("5. 📜 查看旧对话表")
        print("6. 👤 查看特定用户活动")
        print("7. 🔍 自定义查询")
        print("0. 🚪 退出")
        print("-" * 40)
        
        choice = input("请选择操作 (0-7): ").strip()
        
        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            show_table_info()
        elif choice == "2":
            show_users()
        elif choice == "3":
            show_chat_sessions()
        elif choice == "4":
            limit = input("显示多少条消息？(默认10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            show_messages(limit=limit)
        elif choice == "5":
            limit = input("显示多少条对话？(默认10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            show_conversations(limit=limit)
        elif choice == "6":
            user_id = input("请输入用户ID: ").strip()
            if user_id.isdigit():
                show_user_activity(int(user_id))
            else:
                print("❌ 请输入有效的用户ID")
        elif choice == "7":
            custom_query()
        else:
            print("❌ 无效选择，请重试")
        
        input("\n按回车键继续...")

def custom_query():
    """自定义SQL查询"""
    db = SessionLocal()
    try:
        print("\n🔍 自定义查询")
        print("=" * 40)
        print("⚠️  注意：只支持SELECT查询，确保查询安全")
        print("示例查询:")
        print("  SELECT * FROM users LIMIT 5;")
        print("  SELECT COUNT(*) FROM messages;")
        print()
        
        query = input("请输入SQL查询: ").strip()
        
        if not query.upper().startswith("SELECT"):
            print("❌ 只支持SELECT查询")
            return
        
        result = db.execute(text(query))
        rows = result.fetchall()
        
        if not rows:
            print("📭 查询结果为空")
            return
        
        # 获取列名
        columns = result.keys()
        
        print(f"\n📊 查询结果 ({len(rows)} 行):")
        print("-" * 60)
        
        # 打印表头
        header = " | ".join(f"{col:15}" for col in columns)
        print(header)
        print("-" * len(header))
        
        # 打印数据
        for row in rows[:20]:  # 最多显示20行
            row_data = " | ".join(f"{str(val)[:15]:15}" for val in row)
            print(row_data)
        
        if len(rows) > 20:
            print(f"... 还有 {len(rows) - 20} 行数据")
            
    except Exception as e:
        print(f"❌ 查询失败: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print("🎉 欢迎使用数据库查看工具！")
    print("这个工具可以帮助你轻松查看数据库中的所有内容")
    main_menu()
```

