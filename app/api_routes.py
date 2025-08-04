# app/api_routes.py

from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db
from .auth import (
    authenticate_user, create_user, create_access_token, 
    verify_token, ACCESS_TOKEN_EXPIRE_MINUTES
)
from .user_service import UserService
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
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token: Optional[str] = Cookie(None, alias="access_token"),
    db: AsyncSession = Depends(get_db)
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
    user = await get_user_by_email(db, email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在"
        )
    
    return user

@router.post("/register", response_model=TokenResponse)
async def register(user_data: UserRegister, db: AsyncSession = Depends(get_db)):
    """用户注册"""
    try:
        # 创建用户
        user = await create_user(db, user_data.email, user_data.password)
        
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
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):
    """用户登录"""
    try:
        # 验证用户
        user = await authenticate_user(db, user_data.email, user_data.password)
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
async def get_current_user_info(current_user = Depends(get_current_user)):
    """获取当前用户信息"""
    return UserInfo(
        id=current_user.id,
        email=current_user.email,
        is_active=current_user.is_active
    )

@router.post("/logout")
async def logout():
    """用户登出（客户端需要删除token）"""
    return {"message": "登出成功"}

@router.get("/chat-sessions")
async def get_user_chat_sessions(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """获取用户的对话会话列表"""
    try:
        sessions = await UserService.get_user_chat_sessions(db, current_user.id, limit=50)
        
        chat_list = []
        for session in sessions:
            # 获取最后一条消息作为预览
            preview = ""
            if session.messages:
                last_message = session.messages[-1]
                preview = last_message.content[:100] + "..." if len(last_message.content) > 100 else last_message.content
            
            chat_item = {
                "id": session.id,
                "title": session.title,
                "preview": preview,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "message_count": len(session.messages)
            }
            chat_list.append(chat_item)
        
        return {"sessions": chat_list}
        
    except Exception as e:
        logger.error(f"获取用户对话会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取对话会话失败"
        )

@router.get("/chat-sessions/{session_id}/messages")
async def get_session_messages(
    session_id: int,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """获取指定会话的所有消息"""
    try:
        messages = await UserService.get_session_messages(db, session_id, current_user.id)
        
        message_list = []
        for message in messages:
            message_item = {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat()
            }
            message_list.append(message_item)
        
        return {"messages": message_list}
        
    except Exception as e:
        logger.error(f"获取会话消息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取会话消息失败"
        )

# 保留旧的API以兼容现有代码
@router.get("/conversations")
async def get_user_conversations(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """获取用户的聊天记录列表（兼容旧版本）"""
    try:
        conversations = await UserService.get_user_conversations(db, current_user.id, limit=50)
        
        # 按对话分组（这里简化处理，每个问答对作为一个对话）
        chat_list = []
        for conv in conversations:
            chat_item = {
                "id": conv.id,
                "title": conv.question[:50] + "..." if len(conv.question) > 50 else conv.question,
                "preview": conv.answer[:100] + "..." if len(conv.answer) > 100 else conv.answer,
                "created_at": conv.created_at.isoformat(),
                "question": conv.question,
                "answer": conv.answer
            }
            chat_list.append(chat_item)
        
        return {"conversations": chat_list}
        
    except Exception as e:
        logger.error(f"获取用户对话记录失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取对话记录失败"
        )

@router.delete("/chat-sessions/{session_id}")
async def delete_chat_session(
    session_id: int,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """删除指定的对话会话"""
    try:
        # 验证会话属于当前用户
        session = await UserService.get_chat_session_by_id(db, session_id, current_user.id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话会话不存在"
            )
        
        # 删除会话（级联删除所有消息）
        success = await UserService.delete_chat_session(db, session_id, current_user.id)
        if success:
            logger.info(f"用户 {current_user.email} 删除对话会话 {session_id}")
            return {"message": "对话会话删除成功"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="删除对话会话失败"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除对话会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除对话会话失败"
        )

@router.delete("/messages/{message_id}")
async def delete_message(
    message_id: int,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """删除指定的消息"""
    try:
        # 删除消息（确保属于当前用户）
        success = await UserService.delete_message(db, message_id, current_user.id)
        if success:
            logger.info(f"用户 {current_user.email} 删除消息 {message_id}")
            return {"message": "消息删除成功"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="消息不存在或无权限删除"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除消息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除消息失败"
        )