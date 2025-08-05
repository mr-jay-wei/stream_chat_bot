# chatbot_web_demo.py

import asyncio
import json
import time
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session

# 导入我们的对话机器人核心
from app.chatbot_pipeline import ChatbotPipeline, StreamEventType, StreamEvent
from app import config
from app.hot_reload_manager import hot_reload_manager

# 导入数据库和认证相关
from app.database import engine, get_db
from app.models import Base, User, Conversation, ChatSession
from app.api_routes import router as api_router

# 导入新的日志配置
from app.logger_config import get_logger

# 配置日志
logger = get_logger(__name__)
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
    logger.info("应用启动，正在初始化...")
    try:
        # 初始化数据库
        from app.database import init_database
        await init_database()
        logger.info("数据库初始化完成。")
        
        # 初始化对话机器人
        pipeline = ChatbotPipeline()
        logger.info("对话机器人初始化完成。")
        
        # 启动热重载
        if hot_reload_manager and config.ENABLE_HOT_RELOAD:
            hot_reload_manager.start()
            
    except Exception as e:
        logger.error(f"应用初始化失败: {e}", exc_info=True)
        # 即使失败，也需要yield一次，让FastAPI知道启动流程已（不成功地）走完
    
    yield  # <--- 这是关键的分割点

    # --- 应用关闭时执行 ---
    logger.info("应用关闭...")
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

# 注册API路由
app.include_router(api_router, prefix="/api", tags=["auth"])

# --- 静态文件服务 ---
# 挂载static目录，让FastAPI能直接提供HTML, CSS, JS文件
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_homepage():
    """
    当用户访问根路径时，返回我们的主HTML文件。
    """
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    """
    健康检查端点，用于Docker和负载均衡器检查服务状态。
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "chatbot-api",
        "version": "1.0.0"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket连接已建立")
    
    user = None
    db = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理认证消息
            if message.get("type") == "auth":
                token = message.get("token", "")
                if not token:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "未提供认证令牌"}
                    }))
                    continue
                
                # 验证token
                try:
                    from jose import jwt, JWTError
                    import os
                    
                    SECRET_KEY = os.getenv("SECRET_KEY")
                    ALGORITHM = os.getenv("ALGORITHM", "HS256")
                    
                    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                    email = payload.get("sub")
                    
                    if not email:
                        await websocket.send_text(json.dumps({
                            "type": "auth_error",
                            "data": {"error": "无效的认证令牌"}
                        }))
                        continue
                        
                    # 获取数据库会话和用户信息
                    async for db_session in get_db():
                        db = db_session
                        # 使用异步查询
                        from sqlalchemy import select
                        result = await db.execute(select(User).where(User.email == email))
                        user = result.scalar_one_or_none()
                        break
                    
                except JWTError:
                    await websocket.send_text(json.dumps({
                        "type": "auth_error",
                        "data": {"error": "无效的认证令牌"}
                    }))
                    continue
                except Exception as e:
                    logger.error(f"WebSocket认证错误: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "auth_error",
                        "data": {"error": "认证失败"}
                    }))
                    continue
                
                if not user:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "用户不存在"}
                    }))
                    continue
                
                # 认证成功
                await websocket.send_text(json.dumps({
                    "type": "auth_success",
                    "data": {"message": f"欢迎回来，{user.email}！"}
                }))
                logger.info(f"用户 {user.email} WebSocket认证成功")
                continue
            
            # 处理对话消息
            if message.get("type") == "question":
                # 检查用户是否已认证
                if not user or not db:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "请先进行身份认证"}
                    }))
                    continue
                
                question = message.get("content", "")
                session_id = message.get("session_id")  # 可选的会话ID
                
                if not pipeline:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "机器人核心引擎未准备就绪，请检查服务器日志。"}
                    }))
                    continue

                logger.info(f"用户 {user.email} 在会话 {session_id} 中提问: {question}")
                
                # 使用用户ID和会话ID进行对话
                async for event in pipeline.ask_stream(question, db, user.id, session_id):
                    response = {"type": event.type.value, "data": event.data}
                    await websocket.send_text(json.dumps(response))
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接已断开 - 用户: {user.email if user else '未认证'}")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}", exc_info=True)
        if websocket.client_state == 1:  # OPEN
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": f"服务器内部错误: {str(e)}"}
            }))

if __name__ == "__main__":
    import uvicorn
    logger.info("🤖 启动企业级AI对话机器人Web演示...")
    logger.info("🌐 访问地址: http://localhost:8003")
    logger.info("🔥 提示词热重载已激活，尝试修改 app/prompts/assistant_prompt.txt 并刷新对话！")
    
    # 显示日志统计信息
    from app.logger_config import logger_config
    log_stats = logger_config.get_log_stats()
    logger.info(f"📊 日志系统已启动，日志目录: {log_stats['log_directory']}")
    
    uvicorn.run("chatbot_web_demo:app", host="0.0.0.0", port=8003, reload=True)