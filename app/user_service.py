# app/user_service.py

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload

from .models import User, ChatSession, Message, Conversation
from .logger_config import get_logger

logger = get_logger(__name__)

class UserService:
    """用户服务类"""
    
    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
        """根据ID获取用户"""
        try:
            result = await db.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"获取用户失败: {e}")
            return None
    
    @staticmethod
    async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        try:
            result = await db.execute(select(User).where(User.email == email))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"获取用户失败: {e}")
            return None
    
    # === 新的对话会话管理方法 ===
    
    @staticmethod
    async def create_chat_session(db: AsyncSession, user_id: int, title: str) -> Optional[ChatSession]:
        """创建新的对话会话"""
        try:
            chat_session = ChatSession(
                user_id=user_id,
                title=title
            )
            db.add(chat_session)
            await db.commit()
            await db.refresh(chat_session)
            
            logger.info(f"用户 {user_id} 创建新对话会话: {title}")
            return chat_session
            
        except Exception as e:
            logger.error(f"创建对话会话失败: {e}")
            await db.rollback()
            return None
    
    @staticmethod
    async def get_user_chat_sessions(db: AsyncSession, user_id: int, limit: int = 50) -> List[ChatSession]:
        """获取用户的对话会话列表"""
        try:
            result = await db.execute(
                select(ChatSession)
                .where(ChatSession.user_id == user_id)
                .order_by(desc(ChatSession.updated_at))
                .limit(limit)
                .options(selectinload(ChatSession.messages))
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"获取用户对话会话失败: {e}")
            return []
    
    @staticmethod
    async def get_chat_session_by_id(db: AsyncSession, session_id: int, user_id: int) -> Optional[ChatSession]:
        """根据ID获取对话会话（确保属于指定用户）"""
        try:
            result = await db.execute(
                select(ChatSession)
                .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
                .options(selectinload(ChatSession.messages))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"获取对话会话失败: {e}")
            return None
    
    @staticmethod
    async def add_message_to_session(db: AsyncSession, session_id: int, role: str, content: str) -> Optional[Message]:
        """向对话会话添加消息"""
        try:
            message = Message(
                chat_session_id=session_id,
                role=role,
                content=content
            )
            db.add(message)
            
            # 更新会话的最后更新时间
            await db.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            result = await db.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            session = result.scalar_one_or_none()
            if session:
                session.updated_at = message.created_at
            
            await db.commit()
            await db.refresh(message)
            
            logger.debug(f"向会话 {session_id} 添加 {role} 消息")
            return message
            
        except Exception as e:
            logger.error(f"添加消息失败: {e}")
            await db.rollback()
            return None
    
    @staticmethod
    async def get_session_messages(db: AsyncSession, session_id: int, user_id: int) -> List[Message]:
        """获取对话会话的所有消息"""
        try:
            # 首先验证会话属于该用户
            session = await UserService.get_chat_session_by_id(db, session_id, user_id)
            if not session:
                return []
            
            result = await db.execute(
                select(Message)
                .where(Message.chat_session_id == session_id)
                .order_by(Message.created_at)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"获取会话消息失败: {e}")
            return []
    
    @staticmethod
    async def update_session_title(db: AsyncSession, session_id: int, user_id: int, title: str) -> bool:
        """更新对话会话标题"""
        try:
            result = await db.execute(
                select(ChatSession)
                .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
            )
            session = result.scalar_one_or_none()
            
            if session:
                session.title = title
                await db.commit()
                logger.info(f"更新会话 {session_id} 标题为: {title}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"更新会话标题失败: {e}")
            await db.rollback()
            return False
    
    @staticmethod
    async def delete_chat_session(db: AsyncSession, session_id: int, user_id: int) -> bool:
        """删除对话会话（确保属于指定用户）"""
        try:
            result = await db.execute(
                select(ChatSession)
                .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
            )
            session = result.scalar_one_or_none()
            
            if session:
                await db.delete(session)
                await db.commit()
                logger.info(f"删除会话 {session_id} 成功")
                return True
            return False
            
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            await db.rollback()
            return False
    
    @staticmethod
    async def delete_message(db: AsyncSession, message_id: int, user_id: int) -> bool:
        """删除消息（确保属于指定用户的会话）"""
        try:
            # 通过JOIN查询确保消息属于用户的会话
            result = await db.execute(
                select(Message)
                .join(ChatSession)
                .where(
                    Message.id == message_id,
                    ChatSession.user_id == user_id
                )
            )
            message = result.scalar_one_or_none()
            
            if message:
                await db.delete(message)
                await db.commit()
                logger.info(f"删除消息 {message_id} 成功")
                return True
            return False
            
        except Exception as e:
            logger.error(f"删除消息失败: {e}")
            await db.rollback()
            return False
    
    # === 兼容旧版本的方法（逐步废弃） ===
    
    @staticmethod
    async def get_user_conversations(db: AsyncSession, user_id: int, limit: int = 50) -> List[Conversation]:
        """获取用户的对话记录（旧版本兼容）"""
        try:
            result = await db.execute(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(desc(Conversation.created_at))
                .limit(limit)
            )
            conversations = result.scalars().all()
            return list(reversed(conversations))  # 按时间正序返回
        except Exception as e:
            logger.error(f"获取用户对话记录失败: {e}")
            return []
    
    @staticmethod
    async def add_conversation(db: AsyncSession, user_id: int, question: str, answer: str) -> Optional[Conversation]:
        """添加对话记录（旧版本兼容）"""
        try:
            conversation = Conversation(
                user_id=user_id,
                question=question,
                answer=answer
            )
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            
            logger.info(f"用户 {user_id} 添加对话记录成功")
            return conversation
            
        except Exception as e:
            logger.error(f"添加对话记录失败: {e}")
            await db.rollback()
            return None