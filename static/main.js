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