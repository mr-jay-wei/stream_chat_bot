# 项目概览: app

本文档由`generate_project_overview.py`自动生成，包含了项目的结构树和所有可读文件的内容。

## 项目结构

```
app/
├── core
├── prompts
│   └── assistant_prompt.txt
├── __init__.py
├── api_routes.py
├── auth.py
├── chatbot_pipeline.py
├── config.py
├── database.py
├── hot_reload_manager.py
├── limiter.py
├── logger_config.py
├── main.py
├── middleware.py
├── models.py
├── prompt_manager.py
└── session_manager.py
```

---

# 文件内容

## `__init__.py`

[文件为空]

## `api_routes.py`

```python
# app/api_routes.py

from datetime import timedelta
from typing import Optional, Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from starlette.responses import Response
import hashlib
import time

from fastapi_cache.decorator import cache
from fastapi.security import HTTPBearer
from fastapi_cache.decorator import cache
from fastapi_cache.decorator import cache  # harmless duplicate import if any

from .models import User, Conversation, Message, ChatSession, Prompt
from .database import get_db
from .auth import (
    authenticate_user, create_user, create_access_token,
    verify_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_user_by_email
)
from .logger_config import get_logger
from .limiter import limiter
from .session_manager import session_manager

logger = get_logger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)


def key_builder(
    func: Any,
    namespace: str = "",
    request: Request = None,
    response: Optional[Response] = None,
    **kwargs: Any,
) -> str:
    """
    安全的缓存 key 生成器：
      1) 优先使用 request.state.current_user.id（保证同用户缓存隔离）
      2) 其次，如果存在 Bearer token，使用 token 的 sha256 哈希（避免明文 token 出现在 key 中）
      3) 否则返回带时间戳的 nocache key（等同于禁用缓存，防止不同匿名用户共享同一缓存）
    """
    # 1. 优先 user_id
    current_user = getattr(request.state, "current_user", None)
    if current_user and hasattr(current_user, "id"):
        return f"{namespace}:{func.__module__}:{func.__name__}:user_{current_user.id}"

    # 2. 退回到 token 哈希（如果有）
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1]
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
        return f"{namespace}:{func.__module__}:{func.__name__}:token_{token_hash}"

    # 3. 没有用户也没有 token -> 禁用缓存（生成短期唯一 key）
    return f"{namespace}:{func.__module__}:{func.__name__}:nocache_{int(time.time()*1000)}"


# --- 请求模型 ---
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

class PromptBase(BaseModel):
    name: str
    content: str

class PromptCreate(PromptBase):
    pass

class PromptUpdate(PromptBase):
    pass

class PromptResponse(PromptBase):
    id: int
    user_id: int

    class Config:
        from_attributes = True # SQLAlchemy 2.0 orm_mode is deprecated

# [FIX] 重构的 get_current_user 依赖函数
def get_current_user(request: Request) -> User:
    """
    依赖函数：从 request.state 中获取由中间件设置的当前用户。
    如果用户未认证，则引发401错误。
    """
    user = getattr(request.state, "current_user", None)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户未认证或Token无效",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# --- API 路由 ---

@router.post("/register", response_model=TokenResponse)
@limiter.limit("5/minute")
def register(request: Request, user_data: UserRegister, db: Session = Depends(get_db)):
    """用户注册"""
    try:
        user = create_user(db, user_data.email, user_data.password)
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
@limiter.limit("10/minute")
def login(request: Request, user_data: UserLogin, db: Session = Depends(get_db)):
    """用户登录"""
    try:
        user = authenticate_user(db, user_data.email, user_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="邮箱或密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )
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
def get_current_user_info(current_user: User = Depends(get_current_user)):
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


@router.get("/chat-sessions")
def get_user_chat_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取用户的会话列表"""
    try:
        sessions = session_manager.get_user_sessions(db, current_user.id)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"获取用户会话失败: {e}")
        return {"sessions": []}


@router.get("/chat-sessions/{session_id}/messages")
def get_session_messages(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取指定会话的所有消息"""
    try:
        messages = session_manager.get_session_messages(db, session_id, current_user.id)
        return {"messages": messages}
    except Exception as e:
        logger.error(f"获取会话消息失败: {e}")
        return {"messages": []}


@router.get("/conversations")
def get_user_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取用户的聊天记录列表（兼容旧版本）"""
    try:
        sessions = session_manager.get_user_sessions(db, current_user.id)
        if sessions:
            conversations = []
            for session in sessions:
                conversations.append({
                    "id": session["id"],
                    "title": session["title"],
                    "preview": session.get("preview", ""),
                    "created_at": session["created_at"],
                    "updated_at": session.get("updated_at", session["created_at"]),
                    "message_count": session.get("message_count", 0),
                    "session_type": "chat_session"
                })
            return {"conversations": conversations}
        
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """删除指定的对话会话"""
    try:
        success = session_manager.delete_session(db, current_user.id, session_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话记录不存在或不属于当前用户"
            )
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """删除指定的消息"""
    try:
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="消息不存在或无权限删除"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除消息 {message_id} 时发生内部错误: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除消息失败"
        )


@router.get("/prompts", response_model=List[PromptResponse])
def get_user_prompts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取当前登录用户的所有自定义Prompt"""
    prompts = db.query(Prompt).filter(Prompt.user_id == current_user.id).all()
    return prompts

@router.post("/prompts", response_model=PromptResponse, status_code=status.HTTP_201_CREATED)
def create_user_prompt(
    prompt_data: PromptCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """为当前用户创建一个新的自定义Prompt"""
    new_prompt = Prompt(**prompt_data.model_dump(), user_id=current_user.id)
    db.add(new_prompt)
    db.commit()
    db.refresh(new_prompt)
    logger.info(f"用户 {current_user.email} 创建了新的Prompt: {new_prompt.name}")
    return new_prompt

@router.put("/prompts/{prompt_id}", response_model=PromptResponse)
def update_user_prompt(
    prompt_id: int,
    prompt_data: PromptUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """更新一个属于当前用户的Prompt"""
    prompt_to_update = db.query(Prompt).filter(
        Prompt.id == prompt_id,
        Prompt.user_id == current_user.id
    ).first()

    if not prompt_to_update:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prompt不存在或不属于当前用户"
        )

    prompt_to_update.name = prompt_data.name
    prompt_to_update.content = prompt_data.content
    db.commit()
    db.refresh(prompt_to_update)
    logger.info(f"用户 {current_user.email} 更新了Prompt: {prompt_to_update.name}")
    return prompt_to_update

@router.delete("/prompts/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_prompt(
    prompt_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """删除一个属于当前用户的Prompt"""
    prompt_to_delete = db.query(Prompt).filter(
        Prompt.id == prompt_id,
        Prompt.user_id == current_user.id
    ).first()

    if not prompt_to_delete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prompt不存在或不属于当前用户"
        )
    
    db.delete(prompt_to_delete)
    db.commit()
    logger.info(f"用户 {current_user.email} 删除了Prompt ID: {prompt_id}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
```

## `auth.py`

```python
# app/auth.py

from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from dotenv import load_dotenv
load_dotenv() # 加载环境变量
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

## `chatbot_pipeline.py`

```python
import asyncio
import time
import os
from typing import AsyncGenerator, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from . import config
from .prompt_manager import prompt_manager
from .logger_config import get_logger

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.orm import Session
from .models import Prompt, ChatSession

logger = get_logger(__name__)

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
            raise ValueError("API密钥或模型配置未找到。")
        
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

    async def ask_stream(
        self,
        question: str,
        db: Session,
        user_id: int,
        session_id: Optional[int] = None,
        prompt_id: Optional[int] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        流式处理用户问题并返回 AI 回复
        """
        try:
            yield StreamEvent(type=StreamEventType.PROCESSING, data={"message": "思考中..."}, timestamp=time.time())

            from .session_manager import session_manager
            
            current_session_id = session_id
            user_message_id = None

            # --- 创建新会话 ---
            if current_session_id is None:
                current_session_id = session_manager.create_new_session(db, user_id, question, prompt_id)
                if current_session_id:
                    yield StreamEvent(
                        type=StreamEventType.PROCESSING,
                        data={"message": "开始新对话", "session_id": current_session_id},
                        timestamp=time.time()
                    )
                else:
                    raise Exception("创建新会话失败")
            else:
                yield StreamEvent(
                    type=StreamEventType.PROCESSING,
                    data={"message": "继续对话", "session_id": current_session_id},
                    timestamp=time.time()
                )

            # --- 保存用户消息 ---
            if current_session_id:
                user_message_id = session_manager.add_message_to_session(
                    db, current_session_id, user_id, "user", question
                )
                if not user_message_id:
                    error_msg = f"保存用户消息失败。用户 {user_id}, 会话 {current_session_id}"
                    logger.error(error_msg)
                    yield StreamEvent(type=StreamEventType.ERROR, data={"error": error_msg}, timestamp=time.time())
                    return

            # --- 构建 system prompt ---
            system_message_content = ""

            # 1. 优先从 session.prompt_id 获取
            if current_session_id:
                chat_session = db.query(ChatSession).filter_by(id=current_session_id, user_id=user_id).first()
                if chat_session and chat_session.prompt_id:
                    user_prompt = db.query(Prompt).filter(
                        Prompt.id == chat_session.prompt_id,
                        Prompt.user_id == user_id
                    ).first()
                    if user_prompt:
                        system_message_content = user_prompt.content
                        logger.info(f"会话 {current_session_id} 使用了Prompt: {user_prompt.name} (ID: {chat_session.prompt_id})")

            # 2. 如果 session 没有绑定，尝试使用传入的 prompt_id
            if not system_message_content and prompt_id:
                user_prompt = db.query(Prompt).filter(
                    Prompt.id == prompt_id,
                    Prompt.user_id == user_id
                ).first()
                if user_prompt:
                    system_message_content = user_prompt.content
                    logger.info(f"用户 {user_id} 使用了自定义Prompt: {user_prompt.name} (ID: {prompt_id})")
                else:
                    logger.warning(f"用户 {user_id} 尝试使用无效的Prompt ID: {prompt_id}")

            # 3. fallback 到默认 prompt
            if not system_message_content:
                system_prompt_template = prompt_manager.get_template(config.SYSTEM_PROMPT_NAME)
                system_message_content = system_prompt_template.format()
                logger.info(f"用户 {user_id} 使用了默认Prompt。")

            # --- 构建历史上下文 ---
            chat_history = []
            if config.ENABLE_SHORT_TERM_MEMORY and current_session_id:
                session_messages = session_manager.get_session_context_for_ai(db, current_session_id, user_id, max_messages=10)
                for msg in session_messages:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))
            
            messages = [SystemMessage(content=system_message_content)]
            messages.extend(chat_history)

            yield StreamEvent(type=StreamEventType.GENERATION_START, data={"message": "开始生成回答"}, timestamp=time.time())

            # --- 调用 LLM ---
            complete_answer = ""
            api_key = os.getenv("API_KEY", "")
            if not api_key or "invalid" in api_key.lower():
                mock_response = f"这是一个模拟回复。你的问题是：{question}。"
                for char in mock_response:
                    complete_answer += char
                    yield StreamEvent(type=StreamEventType.GENERATION_CHUNK, data={"chunk": char}, timestamp=time.time())
                    await asyncio.sleep(0.02)
            else:
                try:
                    async for chunk in self.llm.astream(messages):
                        chunk_content = chunk.content if hasattr(chunk, 'content') else ""
                        if chunk_content:
                            complete_answer += chunk_content
                            yield StreamEvent(
                                type=StreamEventType.GENERATION_CHUNK,
                                data={"chunk": chunk_content},
                                timestamp=time.time()
                            )
                except Exception as api_error:
                    logger.error(f"LLM API 调用失败: {type(api_error).__name__} - {api_error}", exc_info=True)
                    error_message = "抱歉，AI服务当前网络繁忙或响应超时，请稍后再试。"
                    yield StreamEvent(type=StreamEventType.ERROR, data={"error": error_message}, timestamp=time.time())
                    return

            # --- 保存 AI 消息 ---
            ai_message_id = None
            if current_session_id:
                ai_message_id = session_manager.add_message_to_session(
                    db, current_session_id, user_id, "assistant", complete_answer.strip()
                )
                if not ai_message_id:
                    logger.error(f"保存AI回答失败。用户 {user_id}, 会话 {current_session_id}")

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
            logger.error(f"ask_stream 发生错误: {e}", exc_info=True)
            yield StreamEvent(type=StreamEventType.ERROR, data={"error": str(e)}, timestamp=time.time())
            
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

```

## `config.py`

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

## `database.py`

```python
# app/database.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from typing import Generator
from dotenv import load_dotenv
load_dotenv()
from .logger_config import get_logger
logger = get_logger(__name__)

class Base(DeclarativeBase):
    pass
# --- 从环境变量读取数据库配置 ---
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "mydb")

# --- 生产环境 PostgreSQL 配置 ---
# 使用 psycopg2 驱动
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- 用于本地开发的 SQLite 配置 (备用) ---
# 如果需要切换回SQLite，只需取消注释下面这行，并注释掉上面的PostgreSQL配置，init_database也要改回上面的init_database函数
# DATABASE_URL = "sqlite:///stream_chat_bot.db"

# 创建引擎
try:
    if DATABASE_URL.startswith("sqlite"):
        # SQLite 的特定配置
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            echo=False
        )
        logger.info("正在使用 SQLite 数据库 (用于本地开发)")
    else:
        # PostgreSQL 的配置
        engine = create_engine(
            DATABASE_URL,
            echo=False,
            pool_size=10,
            max_overflow=20
        )
        logger.info(f"正在连接 PostgreSQL 数据库: {DB_HOST}:{DB_PORT}/{DB_NAME}")

except Exception as e:
    logger.error(f"创建数据库引擎失败: {e}")
    raise

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

# def init_database():
#     """初始化数据库表"""
#     try:
#         # 检查数据库文件是否存在
#         db_file = 'stream_chat_bot.db'
#         is_new_db = not os.path.exists(db_file)
        
#         Base.metadata.create_all(bind=engine)
#         if is_new_db:
#             logger.info(f"新的SQLite数据库文件 '{db_file}' 已创建并初始化。")
#         else:
#             logger.info("数据库表初始化检查完成。")

#         # 检查新表是否创建成功
#         from sqlalchemy import text
#         db = SessionLocal()
#         try:
#             # 检查chat_sessions表
#             result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_sessions'"))
#             if result.fetchone():
#                 logger.info("✅ chat_sessions表已存在")
            
#             # 检查messages表
#             result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"))
#             if result.fetchone():
#                 logger.info("✅ messages表已存在")
                
#         except Exception as e:
#             logger.warning(f"检查新表时出错: {e}")
#         finally:
#             db.close()
            
#     except Exception as e:
#         logger.error(f"数据库初始化失败: {e}")
#         raise

def init_database():
    """初始化数据库表"""
    try:
        logger.info("正在初始化/检查数据库表...")
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表初始化完成。")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise
```

## `hot_reload_manager.py`

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

## `limiter.py`

```python
# app/limiter.py

import os
import redis.asyncio as redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
load_dotenv()
from .logger_config import get_logger
logger = get_logger(__name__)

# --- Redis 连接 ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# 构建 Redis URL
redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
logger.info(f"速率限制器正在连接到 Redis: {REDIS_HOST}:{REDIS_PORT}")

# --- 初始化 Limiter ---
# 使用异步 redis 客户端
limiter = Limiter(
    key_func=get_remote_address,  # 使用客户端的IP地址作为唯一标识
    storage_uri=redis_url,
    strategy="fixed-window",  # 固定时间窗口算法
    storage_options={"socket_connect_timeout": 3}
)
```

## `logger_config.py`

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

## `main.py`

```python
# main.py

import asyncio
import json
import time
import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from slowapi import _rate_limit_exceeded_handler   # 新增
from app.middleware import AuthMiddleware
from slowapi.errors import RateLimitExceeded
from app.limiter import limiter
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis.asyncio import Redis as AsyncRedis

from app.chatbot_pipeline import ChatbotPipeline
from app import config
from app.hot_reload_manager import hot_reload_manager
from app.database import SessionLocal, get_db
from app.models import User
from app.api_routes import router as api_router
from app.logger_config import get_logger
from .session_manager import session_manager
from typing import Optional, List, Dict, Any

logger = get_logger(__name__)
pipeline: Optional[ChatbotPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    # Redis 连接
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    try:
        r = AsyncRedis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", encoding="utf8", decode_responses=True)
        await r.ping()
        FastAPICache.init(RedisBackend(r), prefix="fastapi-cache")
        logger.info("FastAPI-Cache 已连接到 Redis")
    except Exception as e:
        logger.error(f"连接 Redis 或初始化缓存失败: {e}")

    # 应用启动逻辑
    logger.info("应用启动，正在初始化...")
    try:
        from app.database import init_database
        init_database()
        pipeline = ChatbotPipeline()
        if hot_reload_manager and config.ENABLE_HOT_RELOAD:
            hot_reload_manager.start()
        logger.info("核心服务初始化完成。")
    except Exception as e:
        logger.error(f"应用初始化失败: {e}", exc_info=True)
    
    yield

    # 应用关闭逻辑
    logger.info("应用关闭...")
    if hot_reload_manager:
        hot_reload_manager.stop()
    if pipeline and hasattr(pipeline, 'executor'):
        pipeline.executor.shutdown(wait=True)

app = FastAPI(
    title="企业级AI对话机器人",
    description="一个支持实时流式响应、具备记忆和可热重载角色的高级对话平台",
    lifespan=lifespan
)

app.add_middleware(AuthMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(api_router, prefix="/api", tags=["auth"])

project_root = Path(__file__).parent.parent
app.mount("/frontend", StaticFiles(directory=project_root / "frontend"), name="frontend")

@app.get("/")
async def get_homepage():
    return FileResponse(project_root / "frontend" / "index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

async def get_user_from_token(token: str, db: Session) -> Optional[User]:
    """辅助函数：从token验证并获取用户对象"""
    if not token:
        return None
    from .auth import verify_token, get_user_by_email
    email = verify_token(token)
    if not email:
        return None
    return get_user_by_email(db, email)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    db: Session = SessionLocal()
    authed_user: Optional[User] = None
    
    try:
        # 第一条消息必须是认证消息
        auth_data = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        auth_message = json.loads(auth_data)
        
        if auth_message.get("type") == "auth":
            token = auth_message.get("token", "")
            authed_user = await get_user_from_token(token, db)
            
            if not authed_user:
                await websocket.send_text(json.dumps({"type": "auth_error", "data": {"error": "无效的认证令牌"}}))
                await websocket.close()
                return
            
            await websocket.send_text(json.dumps({"type": "auth_success", "data": {"message": f"欢迎回来，{authed_user.email}！"}}))
            logger.info(f"用户 {authed_user.email} WebSocket认证成功")
        else:
            await websocket.send_text(json.dumps({"type": "auth_error", "data": {"error": "需要先进行认证"}}))
            await websocket.close()
            return
            
        # 进入主消息循环
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "question":
                question = message.get("content", "")
                session_id = message.get("session_id")
                
                # [SECURITY FIX] 在处理前，校验 session_id (如果存在) 是否属于当前认证用户
                if session_id:
                    from .models import ChatSession
                    session_check = db.query(ChatSession).filter(
                        ChatSession.id == session_id,
                        ChatSession.user_id == authed_user.id
                    ).first()
                    if not session_check:
                        logger.warning(f"安全警告: 用户 {authed_user.email} 尝试访问不属于自己的会话 {session_id}，WebSocket请求被拒绝。")
                        await websocket.send_text(json.dumps({"type": "error", "data": {"error": "无权访问该会话"}}))
                        continue
                
                if not pipeline:
                    await websocket.send_text(json.dumps({"type": "error", "data": {"error": "机器人核心引擎未就绪"}}))
                    continue

                logger.info(f"用户 {authed_user.email} 在会话 {session_id} 中提问: {question}")
                
                async for event in pipeline.ask_stream(question, db, authed_user.id, session_id):
                    await websocket.send_text(json.dumps({"type": event.type.value, "data": event.data}))

    except asyncio.TimeoutError:
        logger.warning("WebSocket认证超时")
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开 - 用户: {authed_user.email if authed_user else '未认证'}")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}", exc_info=True)
    finally:
        if db:
            db.close()
```

## `middleware.py`

```python
# app/middleware.py

from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import status
from fastapi.responses import JSONResponse

from .database import SessionLocal
from .auth import verify_token, get_user_by_email
from .logger_config import get_logger

logger = get_logger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """
        这个中间件会在每个请求被处理前运行。
        它负责从 token 中解析用户，并将其附加到 request.state。
        """
        request.state.current_user = None

        auth_header = request.headers.get("Authorization")
        token = request.cookies.get("access_token")

        auth_token = None
        if auth_header and auth_header.startswith("Bearer "):
            auth_token = auth_header.split(" ")[1]
        elif token:
            auth_token = token

        if auth_token:
            email = verify_token(auth_token)
            if email:
                db = SessionLocal()
                try:
                    user = get_user_by_email(db, email)
                    if user:
                        request.state.current_user = user
                        logger.debug(f"中间件认证成功: {user.email}")
                finally:
                    db.close()

        response = await call_next(request)
        return response

```

## `models.py`

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
    prompts = relationship("Prompt", back_populates="user", cascade="all, delete-orphan")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("Message", back_populates="chat_session", cascade="all, delete-orphan")
    prompt = relationship("Prompt") 
    
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

class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 建立与User模型的关系
    user = relationship("User", back_populates="prompts")
```

## `prompt_manager.py`

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

## `prompts/assistant_prompt.txt`

````
你是一个名为 "哈基米" 的企业级AI助手。

你的职责是：
1. 以友好、专业、乐于助人的语气与用户交流，可以使用Emoji 表情。
2. 能够基于对话历史（如果提供）进行多轮对话，理解上下文和代词指代。
3. 如果遇到不知道如何回答的问题，要诚实地说明，而不是编造答案。
4. 你的回答应力求简洁、清晰、有条理。
5. 在对话开始时，可以简单地问候用户。
````

## `session_manager.py`

```python
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from datetime import datetime

from .models import ChatSession, Message, Conversation, Prompt
from .logger_config import get_logger

logger = get_logger(__name__)

class SessionManager:
    """完整的会话管理器 - 支持真正的多轮对话会话"""

    def __init__(self):
        self.active_sessions: Dict[int, List[Dict]] = {}

    def create_new_session(self, db: Session, user_id: int, first_question: str, prompt_id: Optional[int] = None) -> Optional[int]:
        """创建新的会话，可选绑定 prompt_id"""
        try:
            title = first_question[:50] + "..." if len(first_question) > 50 else first_question
            chat_session = ChatSession(
                user_id=user_id,
                title=title,
                prompt_id=prompt_id  # 🔑 新增：保存角色绑定
            )
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)
            self.active_sessions[chat_session.id] = []
            logger.info(
                f"创建新会话: {chat_session.id} for 用户 {user_id}, 标题: {title}, prompt_id={prompt_id}"
            )
            return chat_session.id
        except Exception as e:
            logger.error(f"创建新会话失败: {e}")
            db.rollback()
            return None

    def get_session_messages(self, db: Session, session_id: int, user_id: int) -> List[Dict]:
        """获取会话的所有消息 (会校验会话归属)"""
        try:
            session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id
            ).first()
            if not session:
                logger.warning(f"会话 {session_id} 不存在或不属于用户 {user_id}")
                return []

            messages = db.query(Message).filter(
                Message.chat_session_id == session_id
            ).order_by(Message.created_at).all()

            return [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat() + "Z"
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"获取会话消息失败: {e}")
            return []

    def add_message_to_session(self, db: Session, session_id: int, user_id: int, role: str, content: str) -> Optional[int]:
        """添加消息到会话 (*安全校验*)"""
        try:
            session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id
            ).first()

            if not session:
                logger.error(
                    f"安全警告：用户 {user_id} 尝试向不属于自己的会话 {session_id} 添加消息，操作被拒绝。"
                )
                return None

            message = Message(chat_session_id=session_id, role=role, content=content)
            db.add(message)
            session.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(message)

            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = []

            self.active_sessions[session_id].append({
                "id": message.id,
                "role": role,
                "content": content,
                "timestamp": message.created_at
            })

            if len(self.active_sessions[session_id]) > 50:
                self.active_sessions[session_id] = self.active_sessions[session_id][-50:]

            logger.debug(f"添加消息到会话 {session_id}: {role} (用户 {user_id})")
            return message.id
        except Exception as e:
            logger.error(f"添加消息到会话 {session_id} 失败: {e}")
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
                last_message = db.query(Message).filter(
                    Message.chat_session_id == session.id
                ).order_by(Message.created_at.desc()).first()
                preview = (
                    last_message.content[:100] + "..."
                    if last_message and len(last_message.content) > 100
                    else (last_message.content if last_message else "")
                )
                session_list.append({
                    "id": session.id,
                    "title": session.title,
                    "preview": preview,
                    "created_at": session.created_at.isoformat() + "Z",
                    "updated_at": session.updated_at.isoformat() + "Z",
                    "message_count": len(session.messages),
                    "prompt_id": session.prompt_id  # 🔑 新增：返回绑定角色
                })
            return session_list
        except Exception as e:
            logger.error(f"获取用户会话列表失败: {e}")
            return []

    def delete_session(self, db: Session, user_id: int, session_id: int) -> bool:
        """删除会话（会校验会话归属）"""
        try:
            session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id
            ).first()
            if session:
                db.delete(session)
                db.commit()
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                logger.info(f"删除会话成功: 用户 {user_id}, 会话 {session_id}")
                return True
            else:
                logger.warning(f"未找到要删除的会话: 用户 {user_id}, 会话 {session_id}")
                return False
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            db.rollback()
            return False

    def get_session_context_for_ai(self, db: Session, session_id: int, user_id: int, max_messages: int = 10) -> List[Dict]:
        """获取会话上下文供AI使用 (会校验会话归属)"""
        messages = self.get_session_messages(db, session_id, user_id)
        return messages[-max_messages:] if len(messages) > max_messages else messages

    def get_legacy_conversations(self, db: Session, user_id: int) -> List[Dict]:
        """获取旧的对话记录"""
        try:
            conversations = db.query(Conversation).filter(
                Conversation.user_id == user_id
            ).order_by(Conversation.created_at.desc()).limit(50).all()
            return [
                {
                    "id": conv.id,
                    "title": conv.question[:50] + "..." if len(conv.question) > 50 else conv.question,
                    "preview": conv.answer[:100] + "..." if len(conv.answer) > 100 else conv.answer,
                    "created_at": conv.created_at.isoformat() + "Z",
                }
                for conv in conversations
            ]
        except Exception as e:
            logger.error(f"获取旧对话记录失败: {e}")
            return []

session_manager = SessionManager()

```

