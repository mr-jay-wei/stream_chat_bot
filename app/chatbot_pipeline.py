# app/chatbot_pipeline.py

import asyncio
import time
import os
from typing import AsyncGenerator, Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from . import config
from .prompt_manager import prompt_manager
from .logger_config import get_logger

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dataclasses import dataclass
from enum import Enum

logger = get_logger(__name__)

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
            raise ValueError("API密钥或模型配置未找到。")
        
        logger.info(f"配置大语言模型: {model_name}")
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.001,
            streaming=True,
            max_tokens=4000,
            request_timeout=60
        )
            
    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def ask_stream(self, question: str, db, user_id: int, session_id: Optional[int] = None) -> AsyncGenerator[StreamEvent, None]:
        try:
            yield StreamEvent(type=StreamEventType.PROCESSING, data={"message": "思考中..."}, timestamp=time.time())

            from .session_manager import session_manager
            
            current_session_id = session_id
            user_message_id = None

            if current_session_id is None:
                current_session_id = session_manager.create_new_session(db, user_id, question)
                if current_session_id:
                    yield StreamEvent(type=StreamEventType.PROCESSING, data={"message": "开始新对话", "session_id": current_session_id}, timestamp=time.time())
                else:
                    raise Exception("创建新会话失败")
            else:
                yield StreamEvent(type=StreamEventType.PROCESSING, data={"message": "继续对话", "session_id": current_session_id}, timestamp=time.time())

            if current_session_id:
                # [SECURITY FIX] 调用 add_message_to_session 时传入 user_id
                user_message_id = session_manager.add_message_to_session(db, current_session_id, user_id, "user", question)
                if user_message_id:
                    logger.info(f"用户提问已保存到会话 {current_session_id}: 消息ID {user_message_id}")
                else:
                    # 如果保存失败（例如因为权限问题），则停止处理
                    error_msg = f"保存用户消息失败，可能是权限问题。用户 {user_id}, 会话 {current_session_id}"
                    logger.error(error_msg)
                    yield StreamEvent(type=StreamEventType.ERROR, data={"error": error_msg}, timestamp=time.time())
                    return

            system_prompt_template = prompt_manager.get_template(config.SYSTEM_PROMPT_NAME)
            system_message_content = system_prompt_template.format()

            chat_history = []
            if config.ENABLE_SHORT_TERM_MEMORY and current_session_id:
                session_messages = session_manager.get_session_context_for_ai(db, current_session_id, user_id, max_messages=10)
                for msg in session_messages:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))
            
            messages = [SystemMessage(content=system_message_content)]
            messages.extend(chat_history)

            yield StreamEvent(type=StreamEventType.GENERATION_START, data={"message": "开始生成回答"}, timestamp=time.time())

            complete_answer = ""
            # ... (LLM 调用逻辑保持不变)
            api_key = os.getenv("API_KEY", "")
            if not api_key or "invalid" in api_key.lower():
                mock_response = f"这是一个模拟回复。你的问题是：{question}。"
                for char in mock_response:
                    complete_answer += char
                    yield StreamEvent(type=StreamEventType.GENERATION_CHUNK, data={"chunk": char}, timestamp=time.time())
                    await asyncio.sleep(0.02)
            else:
                async for chunk in self.llm.astream(messages):
                    chunk_content = chunk.content if hasattr(chunk, 'content') else ""
                    if chunk_content:
                        complete_answer += chunk_content
                        yield StreamEvent(type=StreamEventType.GENERATION_CHUNK, data={"chunk": chunk_content}, timestamp=time.time())

            ai_message_id = None
            if current_session_id:
                # [SECURITY FIX] 调用 add_message_to_session 时传入 user_id
                ai_message_id = session_manager.add_message_to_session(db, current_session_id, user_id, "assistant", complete_answer.strip())
                if not ai_message_id:
                     logger.error(f"保存AI回答失败，可能是权限问题。用户 {user_id}, 会话 {current_session_id}")

            yield StreamEvent(type=StreamEventType.GENERATION_END, data={"message": "生成完成"}, timestamp=time.time())
            yield StreamEvent(
                type=StreamEventType.COMPLETE, 
                data={
                    "message": "对话完成", 
                    "session_id": current_session_id,
                    "user_message_id": user_message_id,
                    "ai_message_id": ai_message_id
                }, 
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"ask_stream 发生错误: {e}", exc_info=True)
            yield StreamEvent(type=StreamEventType.ERROR, data={"error": str(e)}, timestamp=time.time())
            
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)