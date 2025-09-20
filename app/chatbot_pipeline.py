import asyncio
import time
import os
from typing import AsyncGenerator, Any, Optional
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
from sqlalchemy.orm import Session
from .models import Prompt, ChatSession

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

    async def ask_stream(
        self,
        question: str,
        db: Session,
        user_id: int,
        session_id: Optional[int] = None,
        prompt_id: Optional[int] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        流式处理用户问题并返回 AI 回复
        """
        try:
            from .session_manager import session_manager # 修正导入路径
            
            current_session_id = session_id

            if current_session_id is None:
                current_session_id = session_manager.create_new_session(db, user_id, question, prompt_id)
                if current_session_id:
                    # --- 核心改动：确保关键事件优先发送 ---
                    event = StreamEvent(
                        type=StreamEventType.PROCESSING,
                        data={"message": "已创建新会话", "session_id": current_session_id},
                        timestamp=time.time()
                    )
                    yield event
                    # 给事件循环一个处理的机会
                    await asyncio.sleep(0.01)
                else:
                    raise Exception("创建新会话失败")

            # --- 第二个 processing 事件可以照常发送 ---
            yield StreamEvent(type=StreamEventType.PROCESSING, data={"message": "思考中..."}, timestamp=time.time())

            # --- 保存用户消息 ---
            if current_session_id:
                user_message_id = session_manager.add_message_to_session(
                    db, current_session_id, user_id, "user", question
                )
                if not user_message_id:
                    error_msg = f"保存用户消息失败。用户 {user_id}, 会话 {current_session_id}"
                    logger.error(error_msg)
                    yield StreamEvent(type=StreamEventType.ERROR, data={"error": error_msg}, timestamp=time.time())
                    return

            # --- 构建 system prompt ---
            system_message_content = ""

            # 1. 优先从 session.prompt_id 获取
            if current_session_id:
                chat_session = db.query(ChatSession).filter_by(id=current_session_id, user_id=user_id).first()
                if chat_session and chat_session.prompt_id:
                    user_prompt = db.query(Prompt).filter(
                        Prompt.id == chat_session.prompt_id,
                        Prompt.user_id == user_id
                    ).first()
                    if user_prompt:
                        system_message_content = user_prompt.content
                        logger.info(f"会话 {current_session_id} 使用了Prompt: {user_prompt.name} (ID: {chat_session.prompt_id})")

            # 2. 如果 session 没有绑定，尝试使用传入的 prompt_id
            if not system_message_content and prompt_id:
                user_prompt = db.query(Prompt).filter(
                    Prompt.id == prompt_id,
                    Prompt.user_id == user_id
                ).first()
                if user_prompt:
                    system_message_content = user_prompt.content
                    logger.info(f"用户 {user_id} 使用了自定义Prompt: {user_prompt.name} (ID: {prompt_id})")
                else:
                    logger.warning(f"用户 {user_id} 尝试使用无效的Prompt ID: {prompt_id}")

            # 3. fallback 到默认 prompt
            if not system_message_content:
                system_prompt_template = prompt_manager.get_template(config.SYSTEM_PROMPT_NAME)
                system_message_content = system_prompt_template.format()
                logger.info(f"用户 {user_id} 使用了默认Prompt。")

            # --- 构建历史上下文 ---
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

            # --- 调用 LLM ---
            complete_answer = ""
            api_key = os.getenv("API_KEY", "")
            if not api_key or "invalid" in api_key.lower():
                mock_response = f"这是一个模拟回复。你的问题是：{question}。"
                for char in mock_response:
                    complete_answer += char
                    yield StreamEvent(type=StreamEventType.GENERATION_CHUNK, data={"chunk": char}, timestamp=time.time())
                    await asyncio.sleep(0.02)
            else:
                try:
                    async for chunk in self.llm.astream(messages):
                        chunk_content = chunk.content if hasattr(chunk, 'content') else ""
                        if chunk_content:
                            complete_answer += chunk_content
                            yield StreamEvent(
                                type=StreamEventType.GENERATION_CHUNK,
                                data={"chunk": chunk_content},
                                timestamp=time.time()
                            )
                except Exception as api_error:
                    logger.error(f"LLM API 调用失败: {type(api_error).__name__} - {api_error}", exc_info=True)
                    error_message = "抱歉，AI服务当前网络繁忙或响应超时，请稍后再试。"
                    yield StreamEvent(type=StreamEventType.ERROR, data={"error": error_message}, timestamp=time.time())
                    return

            # --- 保存 AI 消息 ---
            ai_message_id = None
            if current_session_id:
                ai_message_id = session_manager.add_message_to_session(
                    db, current_session_id, user_id, "assistant", complete_answer.strip()
                )
                if not ai_message_id:
                    logger.error(f"保存AI回答失败。用户 {user_id}, 会话 {current_session_id}")

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
