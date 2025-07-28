# rag/memory_manager.py

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from . import config


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    question: str
    answer: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.metadata is None:
            self.metadata = {}
        
        # 计算字符长度
        self.char_length = len(self.question) + len(self.answer)
        
        # 截断过长的内容
        if self.char_length > config.SINGLE_CONVERSATION_MAX_LENGTH:
            max_q_len = config.SINGLE_CONVERSATION_MAX_LENGTH // 2
            max_a_len = config.SINGLE_CONVERSATION_MAX_LENGTH - max_q_len
            
            if len(self.question) > max_q_len:
                self.question = self.question[:max_q_len-3] + "..."
            
            if len(self.answer) > max_a_len:
                self.answer = self.answer[:max_a_len-3] + "..."
            
            # 重新计算长度
            self.char_length = len(self.question) + len(self.answer)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """从字典创建对象"""
        return cls(**data)
    
    def get_formatted_time(self) -> str:
        """获取格式化的时间字符串"""
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")


class ShortTermMemoryManager:
    """短期记忆管理器"""
    
    def __init__(self):
        """初始化记忆管理器"""
        self.conversations: List[ConversationTurn] = []
        self.total_char_length = 0
        self.max_length = config.SHORT_TERM_MEMORY_MAX_LENGTH
        self.min_rounds = config.MIN_CONVERSATION_ROUNDS
        self.cleanup_strategy = config.MEMORY_CLEANUP_STRATEGY
        self.sliding_window_size = config.SLIDING_WINDOW_SIZE
        
        print(f"短期记忆管理器已初始化 (最大长度: {self.max_length:,} 字符)")
    
    def add_conversation(self, question: str, answer: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加一轮对话到记忆中
        
        Args:
            question: 用户问题
            answer: AI回答
            metadata: 额外的元数据
        """
        if not config.ENABLE_SHORT_TERM_MEMORY:
            return
        
        # 创建对话记录
        conversation = ConversationTurn(
            question=question,
            answer=answer,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # 添加到列表
        self.conversations.append(conversation)
        self.total_char_length += conversation.char_length
        
        print(f"📝 添加对话记录 (长度: {conversation.char_length} 字符, 总长度: {self.total_char_length:,} 字符)")
        
        # 检查是否需要清理
        self._cleanup_if_needed()
    
    def _cleanup_if_needed(self) -> None:
        """根据策略清理记忆"""
        if self.total_char_length <= self.max_length:
            return
        
        if self.cleanup_strategy == "auto":
            self._auto_cleanup()
        elif self.cleanup_strategy == "sliding_window":
            self._sliding_window_cleanup()
        # manual策略不自动清理
    
    def _auto_cleanup(self) -> None:
        """
        自动清理策略：严格控制总长度不超过max_length
        优先级：长度限制 > 轮数保留
        如果单轮对话超长，会截取该轮对话的内容
        """
        removed_count = 0
        truncated_count = 0
        
        # 第一阶段：移除整轮对话直到满足长度要求或只剩一轮
        while (self.total_char_length > self.max_length and len(self.conversations) > 1):
            removed_conversation = self.conversations.pop(0)
            self.total_char_length -= removed_conversation.char_length
            removed_count += 1
        
        # 第二阶段：如果还是超长且只剩一轮对话，截取该轮对话
        if self.total_char_length > self.max_length and len(self.conversations) == 1:
            last_conversation = self.conversations[0]
            
            # 计算需要截取多少字符
            excess_chars = self.total_char_length - self.max_length
            target_length = last_conversation.char_length - excess_chars
            
            if target_length > 0:
                # 按比例截取问题和答案
                total_original_length = len(last_conversation.question) + len(last_conversation.answer)
                question_ratio = len(last_conversation.question) / total_original_length
                answer_ratio = len(last_conversation.answer) / total_original_length
                
                target_question_length = int(target_length * question_ratio)
                target_answer_length = target_length - target_question_length
                
                # 截取问题和答案
                if target_question_length > 3:  # 保留至少3个字符用于"..."
                    truncated_question = last_conversation.question[:target_question_length-3] + "..."
                else:
                    truncated_question = "..."
                
                if target_answer_length > 3:  # 保留至少3个字符用于"..."
                    truncated_answer = last_conversation.answer[:target_answer_length-3] + "..."
                else:
                    truncated_answer = "..."
                
                # 更新对话内容
                old_length = last_conversation.char_length
                last_conversation.question = truncated_question
                last_conversation.answer = truncated_answer
                last_conversation.char_length = len(truncated_question) + len(truncated_answer)
                
                # 更新总长度
                self.total_char_length = self.total_char_length - old_length + last_conversation.char_length
                truncated_count = 1
                
                print(f"⚠️  最后一轮对话过长，已截取 {old_length - last_conversation.char_length} 字符")
            else:
                # 如果目标长度太小，直接清空该轮对话
                self.conversations.clear()
                self.total_char_length = 0
                removed_count += 1
                print(f"⚠️  单轮对话超出限制太多，已清空所有记忆")
        
        # 第三阶段：如果还有多轮对话但仍超长，继续移除（理论上不应该发生）
        while self.total_char_length > self.max_length and len(self.conversations) > 0:
            removed_conversation = self.conversations.pop(0)
            self.total_char_length -= removed_conversation.char_length
            removed_count += 1
        
        # 输出清理结果
        if removed_count > 0 or truncated_count > 0:
            messages = []
            if removed_count > 0:
                messages.append(f"移除了 {removed_count} 轮旧对话")
            if truncated_count > 0:
                messages.append(f"截取了 {truncated_count} 轮对话内容")
            
            print(f"🧹 自动清理完成：{', '.join(messages)} (当前总长度: {self.total_char_length:,} 字符)")
        
        # 最终验证：确保绝对不超过限制
        if self.total_char_length > self.max_length:
            print(f"❌ 警告：清理后仍超出限制 ({self.total_char_length:,} > {self.max_length:,})")
            # 紧急处理：直接清空
            self.conversations.clear()
            self.total_char_length = 0
            print(f"🚨 紧急清空所有记忆以避免超出限制")
    
    def _sliding_window_cleanup(self) -> None:
        """滑动窗口清理策略：保持固定数量的对话"""
        if len(self.conversations) <= self.sliding_window_size:
            return
        
        # 计算需要移除的对话数量
        excess_count = len(self.conversations) - self.sliding_window_size
        
        # 移除最旧的对话
        for _ in range(excess_count):
            removed_conversation = self.conversations.pop(0)
            self.total_char_length -= removed_conversation.char_length
        
        print(f"🪟 滑动窗口清理了 {excess_count} 轮旧对话 (保留最近 {self.sliding_window_size} 轮)")
    
    def get_recent_conversations(self, count: Optional[int] = None) -> List[ConversationTurn]:
        """
        获取最近的对话记录
        
        Args:
            count: 获取的对话轮数，None表示获取所有
            
        Returns:
            对话记录列表
        """
        if count is None:
            return self.conversations.copy()
        
        return self.conversations[-count:] if count > 0 else []
    
    def get_conversation_context(self, include_count: Optional[int] = None) -> str:
        """
        获取对话上下文字符串，用于提供给LLM
        
        Args:
            include_count: 包含的对话轮数，None表示包含所有
            
        Returns:
            格式化的对话上下文
        """
        conversations = self.get_recent_conversations(include_count)
        
        if not conversations:
            return ""
        
        context_parts = []
        for i, conv in enumerate(conversations, 1):
            context_parts.append(f"第{i}轮对话:")
            context_parts.append(f"用户: {conv.question}")
            context_parts.append(f"助手: {conv.answer}")
            context_parts.append("")  # 空行分隔
        
        return "\n".join(context_parts).strip()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Returns:
            统计信息字典
        """
        if not self.conversations:
            return {
                "total_conversations": 0,
                "total_char_length": 0,
                "memory_usage_percent": 0.0,
                "oldest_conversation": None,
                "newest_conversation": None,
                "average_conversation_length": 0
            }
        
        return {
            "total_conversations": len(self.conversations),
            "total_char_length": self.total_char_length,
            "memory_usage_percent": (self.total_char_length / self.max_length) * 100,
            "oldest_conversation": self.conversations[0].get_formatted_time(),
            "newest_conversation": self.conversations[-1].get_formatted_time(),
            "average_conversation_length": self.total_char_length // len(self.conversations)
        }
    
    def clear_memory(self) -> int:
        """
        清空所有记忆
        
        Returns:
            清除的对话轮数
        """
        cleared_count = len(self.conversations)
        self.conversations.clear()
        self.total_char_length = 0
        
        print(f"🗑️ 已清空所有记忆 (清除了 {cleared_count} 轮对话)")
        return cleared_count
    
    def remove_old_conversations(self, keep_count: int) -> int:
        """
        手动移除旧对话，保留指定数量的最新对话
        
        Args:
            keep_count: 保留的对话轮数
            
        Returns:
            移除的对话轮数
        """
        if keep_count >= len(self.conversations):
            return 0
        
        # 计算需要移除的数量
        remove_count = len(self.conversations) - keep_count
        
        # 移除最旧的对话
        removed_conversations = self.conversations[:remove_count]
        self.conversations = self.conversations[remove_count:]
        
        # 更新总长度
        removed_length = sum(conv.char_length for conv in removed_conversations)
        self.total_char_length -= removed_length
        
        print(f"🧹 手动移除了 {remove_count} 轮旧对话 (当前总长度: {self.total_char_length:,} 字符)")
        return remove_count
    
    def search_conversations(self, keyword: str, limit: int = 10) -> List[Tuple[int, ConversationTurn]]:
        """
        在对话历史中搜索关键词
        
        Args:
            keyword: 搜索关键词
            limit: 返回结果数量限制
            
        Returns:
            匹配的对话记录列表，包含索引和对话对象
        """
        results = []
        keyword_lower = keyword.lower()
        
        for i, conv in enumerate(self.conversations):
            if (keyword_lower in conv.question.lower() or 
                keyword_lower in conv.answer.lower()):
                results.append((i, conv))
                
                if len(results) >= limit:
                    break
        
        return results
    
    def export_conversations(self, file_path: str) -> bool:
        """
        导出对话记录到JSON文件
        
        Args:
            file_path: 导出文件路径
            
        Returns:
            导出是否成功
        """
        try:
            export_data = {
                "export_time": datetime.now().isoformat(),
                "total_conversations": len(self.conversations),
                "total_char_length": self.total_char_length,
                "conversations": [conv.to_dict() for conv in self.conversations]
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"📤 对话记录已导出到: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ 导出对话记录失败: {e}")
            return False
    
    def import_conversations(self, file_path: str, append: bool = False) -> bool:
        """
        从JSON文件导入对话记录
        
        Args:
            file_path: 导入文件路径
            append: 是否追加到现有记录（False表示替换）
            
        Returns:
            导入是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            imported_conversations = [
                ConversationTurn.from_dict(conv_data) 
                for conv_data in import_data['conversations']
            ]
            
            if not append:
                self.clear_memory()
            
            # 添加导入的对话
            for conv in imported_conversations:
                self.conversations.append(conv)
                self.total_char_length += conv.char_length
            
            # 清理如果需要
            self._cleanup_if_needed()
            
            print(f"📥 已导入 {len(imported_conversations)} 轮对话记录")
            return True
            
        except Exception as e:
            print(f"❌ 导入对话记录失败: {e}")
            return False


# 创建全局记忆管理器实例
memory_manager = ShortTermMemoryManager()