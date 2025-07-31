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
from .logger_config import get_logger

# 配置日志
logger = get_logger(__name__)

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
        logger.info("正在初始化企业级对话机器人...")
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self._setup_llm()
        logger.info("企业级对话机器人初始化完成。")

    def _setup_llm(self):
        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL")
        model_name = os.getenv("MODEL_NAME")

        if not all([api_key, base_url, model_name]):
            logger.error("API密钥或模型配置未找到。请检查.env文件。")
            raise ValueError("API密钥或模型配置未找到。请检查.env文件。")
        
        logger.info(f"配置大语言模型: {model_name}")
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