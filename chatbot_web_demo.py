# chatbot_web_demo.py

import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# 导入我们的对话机器人核心
from app.chatbot_pipeline import ChatbotPipeline, StreamEventType, StreamEvent
from app import config
from app.hot_reload_manager import hot_reload_manager

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
    logger.info("🤖 启动企业级AI对话机器人Web演示...")
    logger.info("🌐 访问地址: http://localhost:8003")
    logger.info("🔥 提示词热重载已激活，尝试修改 app/prompts/assistant_prompt.txt 并刷新对话！")
    
    # 显示日志统计信息
    from app.logger_config import logger_config
    log_stats = logger_config.get_log_stats()
    logger.info(f"📊 日志系统已启动，日志目录: {log_stats['log_directory']}")
    
    uvicorn.run("chatbot_web_demo:app", host="0.0.0.0", port=8003, reload=True)