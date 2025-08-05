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
                    "timestamp": msg.created_at
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
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
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