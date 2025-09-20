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