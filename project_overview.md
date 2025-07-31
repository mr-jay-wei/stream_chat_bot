# 项目概览: stream_chat_bot

本文档由`generate_project_overview.py`自动生成，包含了项目的结构树和所有可读文件的内容。

## 项目结构

```
stream_chat_bot/
├── app
│   ├── prompts
│   │   └── assistant_prompt.txt
│   ├── __init__.py
│   ├── chatbot_pipeline.py
│   ├── config.py
│   ├── hot_reload_manager.py
│   ├── memory_manager.py
│   └── prompt_manager.py
├── log
├── static
│   ├── index.html
│   ├── main.js
│   └── style.css
├── test
├── .env_example
├── .gitignore
├── .python-version
├── chatbot_web_demo.py
├── pyproject.toml
└── README.md
```

---

# 文件内容

## `.env_example`

```
CLOUD_INFINI_API_KEY = ""
CLOUD_BASE_URL = ""
CLOUD_MODEL_NAME = ""
DeepSeek_api_key = ""
DeepSeek_base_url = ""
DeepSeek_model_name = ""
```

## `.gitignore`

```
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info
.vscode

# Virtual environments
.venv

```

## `.python-version`

```
3.12

```

## `app/__init__.py`

```python
[文件为空]
```

## `app/chatbot_pipeline.py`

```python
# chatbot_core/chatbot_pipeline.py

import asyncio
import time
import os
from typing import AsyncGenerator, Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

# 导入我们的模块化组件
from . import config
from .prompt_manager import prompt_manager
from .memory_manager import memory_manager

# 导入LangChain核心组件
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 导入流式事件定义
from dataclasses import dataclass
from enum import Enum

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
    """
    企业级对话机器人核心管道 (V1.1 - 支持热重载回调)
    """
    def __init__(self):
        print("正在初始化企业级对话机器人...")
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self._setup_llm()
        print("企业级对话机器人初始化完成。")

    def _setup_llm(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL")
        model_name = os.getenv("OPENROUTER_MODEL_NAME")

        if not all([api_key, base_url, model_name]):
            raise ValueError("API密钥或模型配置未找到。请检查.env文件。")
        
        print(f"  - 配置大语言模型: {model_name}")
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.001,
            streaming=True
        )
            
    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def ask_stream(self, question: str, session_id: str = "default") -> AsyncGenerator[StreamEvent, None]:
        """
        核心的流式对话方法
        """
        try:
            yield StreamEvent(type=StreamEventType.PROCESSING, data={"message": "思考中..."}, timestamp=time.time())

            # 每次调用都重新获取最新的模板，确保热重载生效
            system_prompt_template = prompt_manager.get_template(config.SYSTEM_PROMPT_NAME)
            system_message_content = system_prompt_template.format()

            chat_history = []
            if config.ENABLE_SHORT_TERM_MEMORY:
                for turn in memory_manager.get_recent_conversations():
                    chat_history.append(HumanMessage(content=turn.question))
                    chat_history.append(AIMessage(content=turn.answer))

            messages = [SystemMessage(content=system_message_content)]
            messages.extend(chat_history)
            messages.append(HumanMessage(content=question))

            yield StreamEvent(type=StreamEventType.GENERATION_START, data={"message": "开始生成回答"}, timestamp=time.time())

            complete_answer = ""
            if hasattr(self.llm, 'astream'):
                async for chunk in self.llm.astream(messages):
                    chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    if chunk_content:
                        complete_answer += chunk_content
                        yield StreamEvent(type=StreamEventType.GENERATION_CHUNK, data={"chunk": chunk_content}, timestamp=time.time())
            else:
                response = await self._run_in_executor(self.llm.invoke, messages)
                answer = response.content if hasattr(response, 'content') else str(response)
                complete_answer = answer.strip()
                for char in complete_answer:
                    yield StreamEvent(type=StreamEventType.GENERATION_CHUNK, data={"chunk": char}, timestamp=time.time())
                    # await asyncio.sleep(0.02)

            if config.ENABLE_SHORT_TERM_MEMORY:
                memory_manager.add_conversation(question, complete_answer.strip())

            yield StreamEvent(type=StreamEventType.GENERATION_END, data={"message": "生成完成"}, timestamp=time.time())
            yield StreamEvent(type=StreamEventType.COMPLETE, data={"message": "对话完成"}, timestamp=time.time())

        except Exception as e:
            yield StreamEvent(type=StreamEventType.ERROR, data={"error": str(e)}, timestamp=time.time())
            
    def __del__(self):
        """析构函数，在对象销毁时清理资源。"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
```

## `app/config.py`

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
```

## `app/hot_reload_manager.py`

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
            print(f"🔄 检测到提示词文件修改: {prompt_name}")
            
            # 清除所有相关缓存
            prompt_manager._prompt_cache.pop(prompt_name, None)
            prompt_manager._template_cache.pop(prompt_name, None)
            
            # 重新加载提示词（这会重新填充缓存）
            prompt_manager.load_prompt(prompt_name)
            print(f"✅ 自动重载完成: {prompt_name}")
            
            # 调用回调函数
            if self.callback:
                self.callback("modified", prompt_name)
                
        except Exception as e:
            print(f"❌ 自动重载失败 {prompt_name}: {e}")
    
    def on_created(self, event):
        """文件创建事件处理"""
        if event.is_directory:
            return
        
        prompt_name = self._get_prompt_name(event.src_path)
        if not prompt_name:
            return
        
        try:
            print(f"➕ 检测到新提示词文件: {prompt_name}")
            
            # 加载新提示词
            prompt_manager.load_prompt(prompt_name)
            print(f"✅ 自动加载完成: {prompt_name}")
            
            # 调用回调函数
            if self.callback:
                self.callback("created", prompt_name)
                
        except Exception as e:
            print(f"❌ 自动加载失败 {prompt_name}: {e}")
    
    def on_deleted(self, event):
        """文件删除事件处理"""
        if event.is_directory:
            return
        
        prompt_name = self._get_prompt_name(event.src_path)
        if not prompt_name:
            return
        
        try:
            print(f"🗑️ 检测到提示词文件删除: {prompt_name}")
            
            # 从缓存中移除
            prompt_manager._prompt_cache.pop(prompt_name, None)
            prompt_manager._template_cache.pop(prompt_name, None)
            print(f"✅ 缓存清理完成: {prompt_name}")
            
            # 调用回调函数
            if self.callback:
                self.callback("deleted", prompt_name)
                
        except Exception as e:
            print(f"❌ 缓存清理失败 {prompt_name}: {e}")


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
            
            print(f"🔍 热重载监控已设置，监控目录: {self.watch_directory}")
            
        except Exception as e:
            print(f"❌ 设置文件监控器失败: {e}")
            self.enable_hot_reload = False
    
    def _on_file_change(self, event_type: str, prompt_name: str):
        """文件变化回调处理"""
        # 通知所有注册的回调函数
        for callback in self.callbacks:
            try:
                callback(event_type, prompt_name)
            except Exception as e:
                print(f"❌ 回调函数执行失败: {e}")
    
    def start(self):
        """启动热重载监控"""
        if not self.enable_hot_reload:
            print("⚠️ 热重载功能未启用")
            return False
        
        if self.is_running:
            print("⚠️ 热重载监控已在运行中")
            return True
        
        # 如果observer已经停止，需要重新创建
        if self.observer and not self.observer.is_alive():
            self._setup_file_watcher()
        
        if not self.observer:
            print("❌ 文件监控器初始化失败")
            return False
        
        try:
            self.observer.start()
            self.is_running = True
            print(f"🔥 热重载监控已启动，正在监控: {self.watch_directory}")
            return True
            
        except Exception as e:
            print(f"❌ 启动热重载监控失败: {e}")
            # 尝试重新创建observer
            self._setup_file_watcher()
            if self.observer:
                try:
                    self.observer.start()
                    self.is_running = True
                    print(f"🔥 热重载监控已重新启动，正在监控: {self.watch_directory}")
                    return True
                except Exception as e2:
                    print(f"❌ 重新启动也失败: {e2}")
            return False
    
    def stop(self):
        """停止热重载监控"""
        if not self.observer or not self.is_running:
            return
        
        try:
            self.observer.stop()
            self.observer.join(timeout=5)  # 等待最多5秒
            self.is_running = False
            print("🛑 热重载监控已停止")
            
        except Exception as e:
            print(f"❌ 停止热重载监控失败: {e}")
    
    def add_callback(self, callback: Callable[[str, str], None]):
        """
        添加文件变化回调函数
        
        Args:
            callback: 回调函数，参数为(event_type, prompt_name)
        """
        self.callbacks.add(callback)
        print(f"📝 已添加热重载回调函数")
    
    def remove_callback(self, callback: Callable[[str, str], None]):
        """
        移除文件变化回调函数
        
        Args:
            callback: 要移除的回调函数
        """
        self.callbacks.discard(callback)
        print(f"🗑️ 已移除热重载回调函数")
    
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
    print("⚠️ 未安装watchdog库，热重载功能不可用")
    print("   安装命令: uv add watchdog")


# 创建全局热重载管理器实例
hot_reload_manager = HotReloadManager(
    enable_hot_reload=WATCHDOG_AVAILABLE and getattr(config, 'ENABLE_HOT_RELOAD', True)
) if WATCHDOG_AVAILABLE else None


def enable_hot_reload():
    """启用热重载功能"""
    if not WATCHDOG_AVAILABLE:
        print("❌ watchdog库未安装，无法启用热重载功能")
        print("   安装命令: uv add watchdog")
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

## `app/memory_manager.py`

```python
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
```

## `app/prompt_manager.py`

```python
# rag/prompt_manager.py

import os
from pathlib import Path
from typing import Dict, Optional, Any
from langchain_core.prompts import PromptTemplate


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
            
            print(f"提示词已保存到: {prompt_file}")
            
        except Exception as e:
            raise RuntimeError(f"保存提示词文件失败 {prompt_file}: {e}")
    
    def clear_cache(self) -> None:
        """清除所有缓存。"""
        self._prompt_cache.clear()
        self._template_cache.clear()
        print("提示词缓存已清除")
    
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
                print(f"✅ 重新加载: {prompt_name}")
            except Exception as e:
                print(f"❌ 重新加载失败 {prompt_name}: {e}")
        
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

## `app/prompts/assistant_prompt.txt`

```
你是一个名为 "AI-Jay" 的企业级AI助手。

你的职责是：
1. 以友好、专业、乐于助人的语气与用户交流，可以使用Emoji 表情。
2. 能够基于对话历史（如果提供）进行多轮对话，理解上下文和代词指代。
3. 如果遇到不知道如何回答的问题，要诚实地说明，而不是编造答案。
4. 你的回答应力求简洁、清晰、有条理。
5. 在对话开始时，可以简单地问候用户。
```

## `chatbot_web_demo.py`

```python
# chatbot_web_demo.py

import asyncio
import json
import logging
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
# 导入我们的对话机器人核心
from app.chatbot_pipeline import ChatbotPipeline, StreamEventType, StreamEvent
from app import config
# 配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from app.hot_reload_manager import hot_reload_manager
# 全局单例
pipeline: ChatbotPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI应用的生命周期管理器。
    在应用启动时执行yield之前的部分，在应用关闭时执行yield之后的部分。
    """
    # --- 应用启动时执行 ---
    global pipeline
    logger.info("应用启动，正在初始化对话机器人...")
    try:
        pipeline = ChatbotPipeline()
        logger.info("对话机器人初始化完成。")
        
        # 启动热重载
        # from app.hot_reload_manager import hot_reload_manager
        if hot_reload_manager and config.ENABLE_HOT_RELOAD:
            hot_reload_manager.start()
            
    except Exception as e:
        logger.error(f"Pipeline初始化失败: {e}", exc_info=True)
        # 即使失败，也需要yield一次，让FastAPI知道启动流程已（不成功地）走完
    
    yield  # <--- 这是关键的分割点

    # --- 应用关闭时执行 ---
    logger.info("应用关闭...")
    # from app.hot_reload_manager import hot_reload_manager
    if hot_reload_manager:
        hot_reload_manager.stop()
        
    if pipeline and hasattr(pipeline, 'executor'):
        logger.info("正在清理线程池...")
        pipeline.executor.shutdown(wait=True)
        logger.info("线程池已关闭。")

app = FastAPI(
    title="企业级AI对话机器人", 
    description="一个支持实时流式响应、具备记忆和可热重载角色的高级对话平台",
    lifespan=lifespan # <--- 在这里注册
)

# --- 静态文件服务 ---
# 挂载static目录，让FastAPI能直接提供HTML, CSS, JS文件
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_homepage():
    """
    当用户访问根路径时，返回我们的主HTML文件。
    """
    return FileResponse('static/index.html')


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket连接已建立")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "question":
                question = message.get("content", "")
                if not pipeline:
                    # 如果pipeline未初始化成功，发送错误
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "机器人核心引擎未准备就绪，请检查服务器日志。"}
                    }))
                    continue

                logger.info(f"收到问题: {question}")
                
                async for event in pipeline.ask_stream(question):
                    response = { "type": event.type.value, "data": event.data }
                    await websocket.send_text(json.dumps(response))
                    
    except WebSocketDisconnect:
        logger.info("WebSocket连接已断开")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}", exc_info=True)
        if websocket.client_state == 1: # OPEN
             await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": f"服务器内部错误: {str(e)}"}
            }))

if __name__ == "__main__":
    import uvicorn
    print("🤖 启动企业级AI对话机器人Web演示...")
    print("🌐 访问地址: http://localhost:8003")
    print("🔥 提示词热重载已激活，尝试修改 app/prompts/assistant_prompt.txt 并刷新对话！")
    
    uvicorn.run("chatbot_web_demo:app", host="0.0.0.0", port=8003, reload=True)
```

## `pyproject.toml`

```
[project]
name = "rag-example"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "chromadb>=1.0.15",
    "dotenv>=0.9.9",
    "fastapi>=0.116.1",
    "jieba>=0.42.1",
    "langchain>=0.3.26",
    "langchain-chroma>=0.2.5",
    "langchain-community>=0.3.27",
    "langchain-huggingface>=0.3.1",
    "langchain-openai>=0.3.28",
    "openai>=1.97.0",
    "python-multipart>=0.0.20",
    "rank-bm25>=0.2.2",
    "sentence-transformers>=5.0.0",
    "sse-starlette>=3.0.2",
    "uvicorn>=0.35.0",
    "watchdog>=6.0.0",
    "websockets>=14.0",
]

```

## `README.md`

```markdown
# 🤖 企业级AI对话机器人平台 (V1.0)

本项目是一个功能完备、架构先进、支持实时流式响应、具备短期记忆和可热重载角色功能的企业级对话式AI平台。它从一个复杂的RAG系统中精炼而来，剥离了检索增强的特定逻辑，专注于提供一个通用的、高性能的对话机器人核心框架，可被轻松定制为任何角色。

## ✨ 核心特性

-   **🤖 动态角色扮演 (Dynamic Role-Playing)**:
    通过修改简单的`.txt`提示词文件，可以**实时改变**机器人的性格、职责和说话风格，无需重启服务，极大地提升了AI角色的可运营性。

-   **⚡ 实时流式响应 (Real-time Streaming)**:
    基于WebSocket和`asyncio`，直接对接LLM的流式接口，实现最低延迟的“打字机”效果，提供极致的现代Web交互体验。

-   **🧠 短期对话记忆 (Short-Term Memory)**:
    能够自动保存和管理对话历史，理解上下文和代词指代（如“它”、“他”），进行流畅、连贯的多轮对话，并拥有智能的内存清理策略。

-   **🔥 提示词热重载 (Prompt Hot-Reloading)**:
    运营或产品人员可以直接修改提示词文件，效果**立即生效**。这使得Prompt Engineering的过程从“编码-重启-测试”的繁琐循环，变成了“修改-保存-对话”的丝滑体验。

-   **🏗️ 高度模块化架构 (Highly Modular Architecture)**:
    核心功能（LLM调用、记忆、提示词）被清晰地分离到独立的管理器中，代码高内聚、低耦合，易于维护、测试和未来扩展。

-   **🚀 全栈开箱即用 (Full-Stack Out-of-the-Box)**:
    提供一个基于FastAPI和原生JavaScript的精美、健壮的Web聊天界面，无需任何额外配置，下载即可运行。

## 🏗️ 项目结构

```
/enterprise_chatbot
|-- /chatbot_core/             # 机器人核心模块
|   |-- /prompts/                # 提示词管理中心
|   |   └── assistant_prompt.txt # 默认的机器人角色定义
|   ├── __init__.py
|   ├── chatbot_pipeline.py     # 核心对话管道 (LLM, 记忆, Prompt的粘合剂)
|   ├── config.py               # 全局配置中心
|   ├── memory_manager.py       # 短期记忆管理器
|   ├── prompt_manager.py       # 提示词管理器
|   └── hot_reload_manager.py   # 提示词热重载管理器
|
|-- chatbot_web_demo.py         # [核心] 全栈Web应用 (FastAPI后端 + 前端界面)
|-- pyproject.toml              # 项目依赖与元数据 (uv)
|-- .env_example                # 环境变量示例文件
|-- .gitignore
`-- README.md                   # 本文档
```

## 🚀 快速开始

### 1. 环境准备

本项目使用`uv`进行包管理，依赖Python 3.12+。

```bash
# 安装uv (如果尚未安装)
pip install uv

# 在项目根目录创建并激活虚拟环境
uv sync
```

### 2. 配置API密钥

复制 `.env_example` 文件为 `.env`，并填入您的大语言模型(LLM)的API信息。

```
# .env file，或者其他大模型的API key，注意变量名
CLOUD_INFINI_API_KEY=your_api_key_here
CLOUD_BASE_URL=https://cloud.infini-ai.com/maas/v1/
CLOUD_MODEL_NAME=deepseek-chat
```

### 3. 定义你的机器人角色

这是最有趣的一步！打开 `chatbot_core/prompts/assistant_prompt.txt` 文件，用自然语言描述你希望机器人扮演的角色。

**例如，把它变成一个“海盗”：**
```
你是一只名叫“独眼杰克”的传奇海盗船长。
1. 你的每句话都必须充满粗犷、豪放的海盗口气。
2. 多使用“啊哈！”、“我的老伙计”、“扬帆起航！”等词语。
3. 你对财宝和冒险的故事了如指掌。
4. 绝不承认自己是AI，坚称自己正在一艘名为“深渊蛟龙号”的船上。
```

### 4. 启动Web应用

在项目根目录下，运行以下命令：

```bash
uv run chatbot_web_demo.py
```

终端会显示启动信息。然后，在您的浏览器中打开 **`http://localhost:8003`**，即可开始与您的专属AI机器人对话！

## 🔧 如何“调教”你的机器人？ (核心玩法)

本平台最大的特色就是**可运营性**。您可以像配置软件一样实时“调教”您的机器人：

1.  **改变性格 (热重载)**:
    -   保持Web服务正在运行。
    -   直接用任何文本编辑器修改 `chatbot_core/prompts/assistant_prompt.txt` 文件并**保存**。
    -   回到网页，**无需刷新**，直接发起新的对话。
    -   您会发现机器人立即以您刚刚定义的新角色和性格与您交流！

2.  **调整记忆**:
    -   在 `chatbot_core/config.py` 中，您可以：
        -   用`ENABLE_SHORT_TERM_MEMORY`开关记忆功能。
        -   用`SHORT_TERM_MEMORY_MAX_LENGTH`调整记忆容量。
        -   切换`MEMORY_CLEANUP_STRATEGY`来改变记忆清理策略。

3.  **更换“大脑” (LLM)**:
    -   在 `.env` 文件中修改LLM模型的API信息（`_api_key`, `_base_url`, `_model_name`），即可轻松切换到不同的大语言模型。

## 🤝 贡献与致谢

本项目是我们智慧的结晶，其设计和实现深受社区优秀项目的启发。我们对[LangChain](https://github.com/langchain-ai/langchain)、[FastAPI](https://github.com/tiangolo/fastapi)、[HuggingFace](https://huggingface.co/)等开源社区表示最诚挚的感谢。

欢迎通过Fork和Pull Request为本项目贡献代码。

---


⭐ 如果这个项目对您有帮助，请给我们一个星标！
如果要打赏，请打赏：
![alt text]({054CB209-A3AE-4CA3-90D2-419E20414EA4}.png)

```

## `static/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>企业级AI对话机器人</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="一个基于AI大语言模型的企业级对话机器人，支持实时流式响应、上下文记忆和动态角色配置。">
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <h1>🤖 企业级AI对话机器人</h1>
        <div id="connectionStatus" class="connection-status disconnected">正在连接...</div>
        <div id="chatContainer" class="chat-container">
            <div class="message status-message">欢迎！我是AI-Jay，随时准备为您服务。</div>
        </div>
        <div class="input-container">
            <input type="text" id="questionInput" placeholder="请输入您的问题..." />
            <button id="sendButton" disabled>发送</button>
        </div>
    </div>
    <script src="/static/main.js"></script>
</body>
</html>
```

## `static/main.js`

```javascript
// static/main.js

// 立即执行函数，避免污染全局作用域
(() => {
    let ws = null;
    const chatContainer = document.getElementById('chatContainer');
    const questionInput = document.getElementById('questionInput');
    const sendButton = document.getElementById('sendButton');
    const connectionStatus = document.getElementById('connectionStatus');
    
    function connectWebSocket() {
        ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = () => {
            console.log('WebSocket连接已建立');
            connectionStatus.textContent = '✅ 已连接';
            connectionStatus.className = 'connection-status connected';
            sendButton.disabled = false;
        };
        
        ws.onmessage = (event) => {
            const eventData = JSON.parse(event.data);
            handleStreamEvent(eventData);
        };
        
        ws.onclose = () => {
            console.log('WebSocket连接已关闭');
            connectionStatus.textContent = '❌ 连接断开，3秒后尝试重连...';
            connectionStatus.className = 'connection-status disconnected';
            sendButton.disabled = true;
            setTimeout(connectWebSocket, 3000);
        };
        
        ws.onerror = (error) => console.error('WebSocket错误:', error);
    }
    
    function addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        messageDiv.textContent = content;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return messageDiv;
    }
    
    let currentBotMessageDiv = null;

    function handleStreamEvent(event) {
        switch (event.type) {
            case 'processing':
                addMessage(`[${event.data.message}]`, 'status');
                break;
            case 'generation_start':
                currentBotMessageDiv = addMessage('', 'bot');
                break;
            case 'generation_chunk':
                if (currentBotMessageDiv) {
                    currentBotMessageDiv.textContent += event.data.chunk;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                break;
            case 'generation_end':
            case 'complete':
                currentBotMessageDiv = null;
                sendButton.disabled = false;
                sendButton.textContent = '发送';
                break;
            case 'error':
                addMessage(`[错误]: ${event.data.error}`, 'status');
                sendButton.disabled = false;
                sendButton.textContent = '发送';
                break;
        }
    }
    
    function sendQuestion() {
        const question = questionInput.value.trim();
        if (!question || !ws || ws.readyState !== WebSocket.OPEN) return;
        
        addMessage(question, 'user');
        ws.send(JSON.stringify({ type: 'question', content: question }));
        
        questionInput.value = '';
        sendButton.disabled = true;
        sendButton.textContent = '思考中...';
    }
    
    sendButton.addEventListener('click', sendQuestion);
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendQuestion();
    });
    
    connectWebSocket();
})();
```

## `static/style.css`

```css
/* static/style.css */

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f5f5f5;
}

.container {
    background: white;
    border-radius: 10px;
    padding: 30px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

h1 {
    color: #333;
    text-align: center;
    margin-bottom: 30px;
}

.chat-container {
    height: 400px;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 15px;
    overflow-y: auto;
    background-color: #fafafa;
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
}

.message {
    margin-bottom: 15px;
    padding: 10px 15px;
    border-radius: 18px;
    max-width: 80%;
    word-wrap: break-word;
    line-height: 1.5;
}

.user-message {
    background-color: #007bff;
    color: white;
    margin-left: auto;
    text-align: left;
    align-self: flex-end;
}

.bot-message {
    background-color: #e9ecef;
    color: #333;
    margin-right: auto;
    text-align: left;
    align-self: flex-start;
}

.status-message {
    background-color: #fff3cd;
    color: #856404;
    font-style: italic;
    text-align: center;
    border: 1px solid #ffeaa7;
    max-width: 100%;
    align-self: center;
}

.input-container {
    display: flex;
    gap: 10px;
}

#questionInput {
    flex: 1;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: 25px;
    font-size: 16px;
    padding-left: 20px;
}

#questionInput:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
}

#sendButton {
    padding: 12px 24px;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 25px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s;
}

#sendButton:hover {
    background-color: #0056b3;
}

#sendButton:disabled {
    background-color: #6c757d;
    cursor: not-allowed;
}

.connection-status {
    text-align: center;
    padding: 10px;
    margin-bottom: 20px;
    border-radius: 6px;
    font-weight: 500;
}

.connected {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.disconnected {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
```

