# main.py

import asyncio
import json
import time
import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from slowapi import _rate_limit_exceeded_handler   # 新增
from app.middleware import AuthMiddleware
from slowapi.errors import RateLimitExceeded
from app.limiter import limiter
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis.asyncio import Redis as AsyncRedis

from app.chatbot_pipeline import ChatbotPipeline
from app import config
from app.hot_reload_manager import hot_reload_manager
from app.database import SessionLocal, get_db
from app.models import User
from app.api_routes import router as api_router
from app.logger_config import get_logger
from .session_manager import session_manager
from typing import Optional, List, Dict, Any

logger = get_logger(__name__)
pipeline: Optional[ChatbotPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    # Redis 连接
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    try:
        r = AsyncRedis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", encoding="utf8", decode_responses=True)
        await r.ping()
        FastAPICache.init(RedisBackend(r), prefix="fastapi-cache")
        logger.info("FastAPI-Cache 已连接到 Redis")
    except Exception as e:
        logger.error(f"连接 Redis 或初始化缓存失败: {e}")

    # 应用启动逻辑
    logger.info("应用启动，正在初始化...")
    try:
        from app.database import init_database
        init_database()
        pipeline = ChatbotPipeline()
        if hot_reload_manager and config.ENABLE_HOT_RELOAD:
            hot_reload_manager.start()
        logger.info("核心服务初始化完成。")
    except Exception as e:
        logger.error(f"应用初始化失败: {e}", exc_info=True)
    
    yield

    # 应用关闭逻辑
    logger.info("应用关闭...")
    if hot_reload_manager:
        hot_reload_manager.stop()
    if pipeline and hasattr(pipeline, 'executor'):
        pipeline.executor.shutdown(wait=True)

app = FastAPI(
    title="企业级AI对话机器人",
    description="一个支持实时流式响应、具备记忆和可热重载角色的高级对话平台",
    lifespan=lifespan
)

app.add_middleware(AuthMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(api_router, prefix="/api", tags=["auth"])

project_root = Path(__file__).parent.parent
app.mount("/frontend", StaticFiles(directory=project_root / "frontend"), name="frontend")

@app.get("/")
async def get_homepage():
    return FileResponse(project_root / "frontend" / "index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

async def get_user_from_token(token: str, db: Session) -> Optional[User]:
    """辅助函数：从token验证并获取用户对象"""
    if not token:
        return None
    from .auth import verify_token, get_user_by_email
    email = verify_token(token)
    if not email:
        return None
    return get_user_by_email(db, email)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    db: Session = SessionLocal()
    authed_user: Optional[User] = None
    
    try:
        # 第一条消息必须是认证消息
        auth_data = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        auth_message = json.loads(auth_data)
        
        if auth_message.get("type") == "auth":
            token = auth_message.get("token", "")
            authed_user = await get_user_from_token(token, db)
            
            if not authed_user:
                await websocket.send_text(json.dumps({"type": "auth_error", "data": {"error": "无效的认证令牌"}}))
                await websocket.close()
                return
            
            await websocket.send_text(json.dumps({"type": "auth_success", "data": {"message": f"欢迎回来，{authed_user.email}！"}}))
            logger.info(f"用户 {authed_user.email} WebSocket认证成功")
        else:
            await websocket.send_text(json.dumps({"type": "auth_error", "data": {"error": "需要先进行认证"}}))
            await websocket.close()
            return
            
        # 进入主消息循环
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "question":
                question = message.get("content", "")
                session_id = message.get("session_id")
                
                # [SECURITY FIX] 在处理前，校验 session_id (如果存在) 是否属于当前认证用户
                if session_id:
                    from .models import ChatSession
                    session_check = db.query(ChatSession).filter(
                        ChatSession.id == session_id,
                        ChatSession.user_id == authed_user.id
                    ).first()
                    if not session_check:
                        logger.warning(f"安全警告: 用户 {authed_user.email} 尝试访问不属于自己的会话 {session_id}，WebSocket请求被拒绝。")
                        await websocket.send_text(json.dumps({"type": "error", "data": {"error": "无权访问该会话"}}))
                        continue
                
                if not pipeline:
                    await websocket.send_text(json.dumps({"type": "error", "data": {"error": "机器人核心引擎未就绪"}}))
                    continue

                logger.info(f"用户 {authed_user.email} 在会话 {session_id} 中提问: {question}")
                
                async for event in pipeline.ask_stream(question, db, authed_user.id, session_id):
                    await websocket.send_text(json.dumps({"type": event.type.value, "data": event.data}))

    except asyncio.TimeoutError:
        logger.warning("WebSocket认证超时")
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开 - 用户: {authed_user.email if authed_user else '未认证'}")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}", exc_info=True)
    finally:
        if db:
            db.close()