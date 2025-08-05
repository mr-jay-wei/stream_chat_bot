# app/user_service.py

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .models import User, Conversation
# ChatSession, Message 模型不存在，暂时注释掉
from .logger_config import get_logger

logger = get_logger(__name__)

class UserService:
    """用户服务类"""
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """根据ID获取用户"""
        try:
            return db.query(User).filter(User.id == user_id).first()
        except Exception as e:
            logger.error(f"获取用户失败: {e}")
            return None
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        try:
            return db.query(User).filter(User.email == email).first()
        except Exception as e:
            logger.error(f"获取用户失败: {e}")
            return None
    
    # === 新的对话会话管理方法（暂时注释掉，因为ChatSession模型不存在） ===
    
    # @staticmethod
    # def create_chat_session(db: Session, user_id: int, title: str) -> Optional[ChatSession]:
    #     """创建新的对话会话"""
    #     try:
    #         chat_session = ChatSession(
    #             user_id=user_id,
    #             title=title
    #         )
    #         db.add(chat_session)
    #         db.commit()
    #         db.refresh(chat_session)
    #         
    #         logger.info(f"用户 {user_id} 创建新对话会话: {title}")
    #         return chat_session
    #         
    #     except Exception as e:
    #         logger.error(f"创建对话会话失败: {e}")
    #         db.rollback()
    #         return None
    
    # @staticmethod
    # def get_user_chat_sessions(db: Session, user_id: int, limit: int = 50) -> List[ChatSession]:
    #     """获取用户的对话会话列表（暂时注释掉，因为ChatSession模型不存在）"""
    #     try:
    #         return []  # 暂时返回空列表
    #     except Exception as e:
    #         logger.error(f"获取用户对话会话失败: {e}")
    #         return []
    
    # @staticmethod
    # def get_chat_session_by_id(db: Session, session_id: int, user_id: int) -> Optional[ChatSession]:
    #     """根据ID获取对话会话（确保属于指定用户）（暂时注释掉，因为ChatSession模型不存在）"""
    #     try:
    #         return None  # 暂时返回None
    #     except Exception as e:
    #         logger.error(f"获取对话会话失败: {e}")
    #         return None
    
    # @staticmethod
    # def add_message_to_session(db: Session, session_id: int, role: str, content: str) -> Optional[Message]:
    #     """向对话会话添加消息（暂时注释掉，因为Message模型不存在）"""
    #     try:
    #         return None  # 暂时返回None
    #     except Exception as e:
    #         logger.error(f"添加消息失败: {e}")
    #         db.rollback()
    #         return None
    
    # @staticmethod
    # def get_session_messages(db: Session, session_id: int, user_id: int) -> List[Message]:
    #     """获取对话会话的所有消息（暂时注释掉，因为Message模型不存在）"""
    #     try:
    #         return []  # 暂时返回空列表
    #     except Exception as e:
    #         logger.error(f"获取会话消息失败: {e}")
    #         return []
    
    # @staticmethod
    # def update_session_title(db: Session, session_id: int, user_id: int, title: str) -> bool:
    #     """更新对话会话标题（暂时注释掉，因为ChatSession模型不存在）"""
    #     try:
    #         return False  # 暂时返回False
    #     except Exception as e:
    #         logger.error(f"更新会话标题失败: {e}")
    #         db.rollback()
    #         return False
    
    # @staticmethod
    # def delete_chat_session(db: Session, session_id: int, user_id: int) -> bool:
    #     """删除对话会话（确保属于指定用户）（暂时注释掉，因为ChatSession模型不存在）"""
    #     try:
    #         return False  # 暂时返回False
    #     except Exception as e:
    #         logger.error(f"删除会话失败: {e}")
    #         db.rollback()
    #         return False
    
    # @staticmethod
    # def delete_message(db: Session, message_id: int, user_id: int) -> bool:
    #     """删除消息（确保属于指定用户的会话）（暂时注释掉，因为Message模型不存在）"""
    #     try:
    #         return False  # 暂时返回False
    #     except Exception as e:
    #         logger.error(f"删除消息失败: {e}")
    #         db.rollback()
    #         return False
    
    # === 兼容旧版本的方法（逐步废弃） ===
    
    @staticmethod
    def get_user_conversations(db: Session, user_id: int, limit: int = 50) -> List[Conversation]:
        """获取用户的对话记录（旧版本兼容）"""
        try:
            conversations = db.query(Conversation).filter(
                Conversation.user_id == user_id
            ).order_by(desc(Conversation.created_at)).limit(limit).all()
            return list(reversed(conversations))  # 按时间正序返回
        except Exception as e:
            logger.error(f"获取用户对话记录失败: {e}")
            return []
    
    @staticmethod
    def add_conversation(db: Session, user_id: int, question: str, answer: str) -> Optional[Conversation]:
        """添加对话记录（旧版本兼容）"""
        try:
            conversation = Conversation(
                user_id=user_id,
                question=question,
                answer=answer
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            
            logger.info(f"用户 {user_id} 添加对话记录成功")
            return conversation
            
        except Exception as e:
            logger.error(f"添加对话记录失败: {e}")
            db.rollback()
            return None