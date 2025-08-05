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
# from .memory_manager import memory_manager  # 已移动到test文件夹
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
            streaming=True,
            max_tokens=4000,  # 增加最大输出长度
            request_timeout=60  # 增加请求超时时间
        )
            
    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def ask_stream(self, question: str, db, user_id: int, session_id: Optional[int] = None) -> AsyncGenerator[StreamEvent, None]:
        """
        核心的流式对话方法 - 支持用户数据隔离和会话管理
        """
        try:
            yield StreamEvent(type=StreamEventType.PROCESSING, data={"message": "思考中..."}, timestamp=time.time())

            # 导入会话管理器
            from .session_manager import session_manager
            
            # 处理会话逻辑
            current_session_id = session_id
            
            if session_id is None:
                # 创建新会话
                current_session_id = session_manager.create_new_session(db, user_id, question)
                if current_session_id:
                    yield StreamEvent(type=StreamEventType.PROCESSING, data={"message": "开始新对话", "session_id": current_session_id}, timestamp=time.time())
                else:
                    raise Exception("创建新会话失败")
            else:
                # 继续现有会话
                current_session_id = session_id
                yield StreamEvent(type=StreamEventType.PROCESSING, data={"message": "继续对话", "session_id": current_session_id}, timestamp=time.time())

            # 每次调用都重新获取最新的模板，确保热重载生效
            system_prompt_template = prompt_manager.get_template(config.SYSTEM_PROMPT_NAME)
            system_message_content = system_prompt_template.format()

            # 获取会话历史
            chat_history = []
            if config.ENABLE_SHORT_TERM_MEMORY and current_session_id:
                try:
                    # 获取会话的历史消息
                    session_messages = session_manager.get_session_context_for_ai(db, current_session_id, user_id, max_messages=10)
                    
                    # 转换为LangChain消息格式
                    for msg in session_messages:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history.append(AIMessage(content=msg["content"]))
                    
                    logger.debug(f"加载了会话 {current_session_id} 的 {len(session_messages)} 条历史消息")
                        
                except Exception as e:
                    logger.error(f"获取会话历史失败: {e}")
                    # 如果获取失败，继续使用空的chat_history
            
            # 先将用户问题添加到会话
            if current_session_id:
                session_manager.add_message_to_session(db, current_session_id, "user", question)

            messages = [SystemMessage(content=system_message_content)]
            messages.extend(chat_history)
            messages.append(HumanMessage(content=question))

            yield StreamEvent(type=StreamEventType.GENERATION_START, data={"message": "开始生成回答"}, timestamp=time.time())

            complete_answer = ""
            
            # 检查API密钥是否有效，如果无效则使用模拟回复
            api_key = os.getenv("API_KEY", "")
            if not api_key or api_key.endswith(".com") or "invalid" in api_key.lower():
                # 使用模拟回复进行测试
                logger.warning("API密钥无效，使用模拟回复进行测试")
                mock_response = f"这是一个模拟回复，用于测试数据库保存功能。你的问题是：{question}。当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}"
                
                # 模拟流式回复
                for i, char in enumerate(mock_response):
                    complete_answer += char
                    yield StreamEvent(type=StreamEventType.GENERATION_CHUNK, data={"chunk": char}, timestamp=time.time())
                    # 添加小延迟模拟真实的流式效果
                    await asyncio.sleep(0.02)
            else:
                # 使用真实的LLM API
                try:
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
                            await asyncio.sleep(0.02)
                except Exception as api_error:
                    logger.error(f"LLM API调用失败: {api_error}")
                    # API失败时使用备用回复
                    fallback_response = f"抱歉，AI服务暂时不可用。你的问题是：{question}。请稍后重试。"
                    for char in fallback_response:
                        complete_answer += char
                        yield StreamEvent(type=StreamEventType.GENERATION_CHUNK, data={"chunk": char}, timestamp=time.time())
                        await asyncio.sleep(0.02)

            # 将AI回答添加到会话
            try:
                if current_session_id:
                    # 使用新的会话管理系统保存AI回答
                    message_id = session_manager.add_message_to_session(db, current_session_id, "assistant", complete_answer.strip())
                    if message_id:
                        logger.info(f"AI回答已保存到会话 {current_session_id}: 消息ID {message_id}")
                    else:
                        logger.error(f"保存AI回答失败: 会话 {current_session_id}")
                else:
                    logger.error("无效的会话ID，无法保存AI回答")
                    
            except Exception as save_error:
                logger.error(f"保存AI回答失败: {save_error}")
                try:
                    db.rollback()
                except:
                    pass

            yield StreamEvent(type=StreamEventType.GENERATION_END, data={"message": "生成完成"}, timestamp=time.time())
            yield StreamEvent(type=StreamEventType.COMPLETE, data={"message": "对话完成", "session_id": session_id}, timestamp=time.time())

        except Exception as e:
            yield StreamEvent(type=StreamEventType.ERROR, data={"error": str(e)}, timestamp=time.time())
            
    def __del__(self):
        """析构函数，在对象销毁时清理资源。"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)