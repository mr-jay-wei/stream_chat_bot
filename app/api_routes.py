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