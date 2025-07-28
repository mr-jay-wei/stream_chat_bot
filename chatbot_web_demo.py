# chatbot_web_demo.py

import asyncio
import json
import logging
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
# 导入我们的对话机器人核心
from app.chatbot_pipeline import ChatbotPipeline, StreamEventType, StreamEvent
from app import config
# 配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# app = FastAPI(title="企业级AI对话机器人", description="一个支持实时流式响应、具备记忆和可热重载角色的高级对话平台")

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
        from app.hot_reload_manager import hot_reload_manager
        if hot_reload_manager and config.ENABLE_HOT_RELOAD:
            hot_reload_manager.start()
            
    except Exception as e:
        logger.error(f"Pipeline初始化失败: {e}", exc_info=True)
        # 即使失败，也需要yield一次，让FastAPI知道启动流程已（不成功地）走完
    
    yield  # <--- 这是关键的分割点

    # --- 应用关闭时执行 ---
    logger.info("应用关闭...")
    from app.hot_reload_manager import hot_reload_manager
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

'''
@app.on_event("startup")
async def startup_event():
    """应用启动时执行，创建并初始化Pipeline。"""
    global pipeline
    logger.info("应用启动，正在初始化对话机器人...")
    pipeline = ChatbotPipeline()
    logger.info("对话机器人初始化完成。")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行，显式调用清理逻辑。"""
    global pipeline
    logger.info("应用关闭...")
    if pipeline:
        # 调用析构函数中的逻辑，确保资源被释放
        pipeline.__del__()
    logger.info("应用已关闭。")
'''
@app.get("/")
async def get_homepage():
    # ... (HTML, CSS, JS部分保持不变)
    # ... 为节省篇幅，此处省略，请使用您之前的版本 ...
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>企业级AI对话机器人</title>
        <meta charset="utf-8">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
            .container { background: white; border-radius: 10px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; margin-bottom: 30px; }
            .chat-container { height: 400px; border: 1px solid #ddd; border-radius: 8px; padding: 15px; overflow-y: auto; background-color: #fafafa; margin-bottom: 20px; }
            .message { margin-bottom: 15px; padding: 10px 15px; border-radius: 18px; max-width: 80%; word-wrap: break-word; }
            .user-message { background-color: #007bff; color: white; margin-left: auto; text-align: left; }
            .bot-message { background-color: #e9ecef; color: #333; margin-right: auto; text-align: left; }
            .status-message { background-color: #fff3cd; color: #856404; font-style: italic; text-align: center; border: 1px solid #ffeaa7; }
            .input-container { display: flex; gap: 10px; }
            #questionInput { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 25px; font-size: 16px; }
            #sendButton { padding: 12px 24px; background-color: #007bff; color: white; border: none; border-radius: 25px; cursor: pointer; font-size: 16px; transition: background-color 0.3s; }
            #sendButton:hover { background-color: #0056b3; }
            #sendButton:disabled { background-color: #6c757d; cursor: not-allowed; }
            .connection-status { text-align: center; padding: 10px; margin-bottom: 20px; border-radius: 6px; }
            .connected { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .disconnected { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
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
        <script>
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
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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