# app/api_routes.py

from datetime import timedelta
from typing import Optional,  Any
from fastapi import APIRouter, Depends, HTTPException, status, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from .models import User
from fastapi import Request

from .limiter import limiter # 导入我们的limiter实例
from fastapi_cache.decorator import cache
from starlette.responses import Response

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

# [CACHE] 自定义缓存键生成器，确保每个用户的缓存是独立的
def key_builder(
    func: Any,
    namespace: str = "",
    *,
    request: Request,
    response: Response | None = None,
    **kwargs: Any,
) -> str:
    # 获取当前用户信息
    # 注意：这里我们不能直接用Depends，需要从request中获取
    current_user = getattr(request.state, "current_user", None)
    
    # 如果能获取到用户ID，就加入到缓存键中
    if current_user and hasattr(current_user, 'id'):
        cache_key = f"{namespace}:{func.__module__}:{func.__name__}:user_{current_user.id}"
    else:
        # 否则，使用IP地址作为后备
        cache_key = f"{namespace}:{func.__module__}:{func.__name__}:ip_{request.client.host}"
        
    return cache_key

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
@limiter.limit("5/minute") # 每分钟最多5次
def register(request: Request, user_data: UserRegister, db: Session = Depends(get_db)):
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
@limiter.limit("10/minute") # 登录允许更频繁一些，每分钟10次
def login(request: Request, user_data: UserLogin, db: Session = Depends(get_db)):
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
@cache(expire=60, key_builder=key_builder) # 缓存60秒，并使用自定义key生成器
def get_user_chat_sessions(
    request: Request, # 确保request被传入
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # [CACHE] 将用户信息附加到request.state，供key_builder使用
    request.state.current_user = current_user

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
@cache(expire=60, key_builder=key_builder) # 同样为旧API也加上缓存
def get_user_conversations(
    request: Request, # 确保request被传入
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # [CACHE] 将用户信息附加到request.state
    request.state.current_user = current_user
    
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