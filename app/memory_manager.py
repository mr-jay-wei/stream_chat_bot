# app/memory_manager.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
from sqlalchemy.ext.asyncio import AsyncSession

from . import config
from .logger_config import get_logger
from .user_service import UserService
from .models import Message

# 配置日志
logger = get_logger(__name__)

@dataclass
class ConversationTurn:
    """单轮对话记录"""
    question: str
    answer: str
    timestamp: datetime
    
    def __post_init__(self):
        """计算字符长度"""
        self.length = len(self.question) + len(self.answer)

class MemoryManager:
    """
    基于数据库的记忆管理器 - 支持用户数据隔离和会话管理
    """
    def __init__(self):
        self._lock = threading.Lock()
        logger.info("数据库记忆管理器初始化完成。")
    
    async def add_message_to_session(self, db: AsyncSession, session_id: int, role: str, content: str) -> None:
        """向对话会话添加消息"""
        try:
            # 检查消息长度
            if len(content) > config.SINGLE_CONVERSATION_MAX_LENGTH:
                logger.warning(f"消息过长 ({len(content)} 字符)，将被截断。")
                content = content[:config.SINGLE_CONVERSATION_MAX_LENGTH-100] + "...[内容过长已截断]"
            
            # 保存到数据库
            await UserService.add_message_to_session(db, session_id, role, content)
            logger.debug(f"向会话 {session_id} 添加 {role} 消息")
            
        except Exception as e:
            logger.error(f"添加消息失败: {e}")
    
    async def get_session_context(self, db: AsyncSession, session_id: int, user_id: int, max_turns: Optional[int] = None) -> List[ConversationTurn]:
        """获取对话会话的上下文（转换为ConversationTurn格式以兼容现有代码）"""
        try:
            messages = await UserService.get_session_messages(db, session_id, user_id)
            
            # 将消息转换为问答对
            turns = []
            current_question = None
            
            for message in messages:
                if message.role == 'user':
                    current_question = message.content
                elif message.role == 'assistant' and current_question:
                    turn = ConversationTurn(
                        question=current_question,
                        answer=message.content,
                        timestamp=message.created_at
                    )
                    turns.append(turn)
                    current_question = None
            
            # 如果指定了最大轮数，返回最近的几轮
            if max_turns and len(turns) > max_turns:
                turns = turns[-max_turns:]
            
            return turns
            
        except Exception as e:
            logger.error(f"获取会话上下文失败: {e}")
            return []
    
    async def create_new_session(self, db: AsyncSession, user_id: int, first_question: str) -> Optional[int]:
        """创建新的对话会话"""
        try:
            # 生成会话标题（使用问题的前50个字符）
            title = first_question[:50] + "..." if len(first_question) > 50 else first_question
            
            session = await UserService.create_chat_session(db, user_id, title)
            if session:
                return session.id
            return None
            
        except Exception as e:
            logger.error(f"创建新会话失败: {e}")
            return None
    
    async def get_memory_stats(self, db: AsyncSession, user_id: int) -> Dict[str, Any]:
        """获取用户记忆统计信息"""
        try:
            sessions = await UserService.get_user_chat_sessions(db, user_id)
            total_messages = sum(len(session.messages) for session in sessions)
            total_length = sum(
                sum(len(msg.content) for msg in session.messages) 
                for session in sessions
            )
            
            return {
                "total_sessions": len(sessions),
                "total_messages": total_messages,
                "total_length": total_length,
                "max_length": config.SHORT_TERM_MEMORY_MAX_LENGTH,
                "usage_percentage": (total_length / config.SHORT_TERM_MEMORY_MAX_LENGTH) * 100 if config.SHORT_TERM_MEMORY_MAX_LENGTH > 0 else 0,
                "cleanup_strategy": config.MEMORY_CLEANUP_STRATEGY
            }
        except Exception as e:
            logger.error(f"获取记忆统计失败: {e}")
            return {
                "total_sessions": 0,
                "total_messages": 0,
                "total_length": 0,
                "max_length": config.SHORT_TERM_MEMORY_MAX_LENGTH,
                "usage_percentage": 0,
                "cleanup_strategy": config.MEMORY_CLEANUP_STRATEGY
            }

# 全局单例
memory_manager = MemoryManager()