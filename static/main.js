// static/main.js

// 立即执行函数，避免污染全局作用域
(() => {
    // 全局状态
    let ws = null;
    let currentUser = null;
    let accessToken = null;
    let currentSessionId = null;
    let chatSessions = [];
    
    // DOM元素
    const authContainer = document.getElementById('authContainer');
    const chatApp = document.getElementById('chatApp');
    const loginTab = document.getElementById('loginTab');
    const registerTab = document.getElementById('registerTab');
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const authMessage = document.getElementById('authMessage');
    const userEmail = document.getElementById('userEmail');
    const logoutButton = document.getElementById('logoutButton');
    const newChatButton = document.getElementById('newChatButton');
    const chatHistoryList = document.getElementById('chatHistoryList');
    const chatContainer = document.getElementById('chatContainer');
    const questionInput = document.getElementById('questionInput');
    const sendButton = document.getElementById('sendButton');
    const connectionStatus = document.getElementById('connectionStatus');
    
    // 初始化
    function init() {
        // 检查是否已有token
        const savedToken = localStorage.getItem('access_token');
        if (savedToken) {
            accessToken = savedToken;
            verifyTokenAndShowChat();
        } else {
            showAuthInterface();
        }
        
        // 绑定事件
        bindEvents();
    }
    
    // 绑定事件
    function bindEvents() {
        // 认证标签切换
        loginTab.addEventListener('click', () => switchAuthTab('login'));
        registerTab.addEventListener('click', () => switchAuthTab('register'));
        
        // 表单提交
        loginForm.addEventListener('submit', handleLogin);
        registerForm.addEventListener('submit', handleRegister);
        
        // 登出和新建对话
        logoutButton.addEventListener('click', handleLogout);
        newChatButton.addEventListener('click', startNewChat);
        
        // 发送消息
        sendButton.addEventListener('click', sendQuestion);
        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuestion();
            }
        });
    }
    
    // 切换认证标签
    function switchAuthTab(tab) {
        if (tab === 'login') {
            loginTab.classList.add('active');
            registerTab.classList.remove('active');
            loginForm.classList.remove('hidden');
            registerForm.classList.add('hidden');
        } else {
            registerTab.classList.add('active');
            loginTab.classList.remove('active');
            registerForm.classList.remove('hidden');
            loginForm.classList.add('hidden');
        }
        clearAuthMessage();
    }
    
    // 处理登录
    async function handleLogin(e) {
        e.preventDefault();
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;
        
        try {
            showAuthMessage('正在登录...', 'info');
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password }),
            });
            
            const data = await response.json();
            
            if (response.ok) {
                accessToken = data.access_token;
                currentUser = { email: data.user_email };
                localStorage.setItem('access_token', accessToken);
                showAuthMessage('登录成功！', 'success');
                setTimeout(() => showChatInterface(), 1000);
            } else {
                showAuthMessage(data.detail || '登录失败', 'error');
            }
        } catch (error) {
            console.error('登录错误:', error);
            showAuthMessage('网络错误，请稍后重试', 'error');
        }
    }
    
    // 处理注册
    async function handleRegister(e) {
        e.preventDefault();
        const email = document.getElementById('registerEmail').value;
        const password = document.getElementById('registerPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        
        if (password !== confirmPassword) {
            showAuthMessage('两次输入的密码不一致', 'error');
            return;
        }
        
        try {
            showAuthMessage('正在注册...', 'info');
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password }),
            });
            
            const data = await response.json();
            
            if (response.ok) {
                accessToken = data.access_token;
                currentUser = { email: data.user_email };
                localStorage.setItem('access_token', accessToken);
                showAuthMessage('注册成功！', 'success');
                setTimeout(() => showChatInterface(), 1000);
            } else {
                showAuthMessage(data.detail || '注册失败', 'error');
            }
        } catch (error) {
            console.error('注册错误:', error);
            showAuthMessage('网络错误，请稍后重试', 'error');
        }
    }
    
    // 验证token并显示聊天界面
    async function verifyTokenAndShowChat() {
        try {
            const response = await fetch('/api/me', {
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                },
            });
            
            if (response.ok) {
                const userData = await response.json();
                currentUser = userData;
                showChatInterface();
            } else {
                // token无效，清除并显示登录界面
                localStorage.removeItem('access_token');
                accessToken = null;
                showAuthInterface();
            }
        } catch (error) {
            console.error('验证token错误:', error);
            localStorage.removeItem('access_token');
            accessToken = null;
            showAuthInterface();
        }
    }
    
    // 显示认证界面
    function showAuthInterface() {
        authContainer.classList.remove('hidden');
        chatApp.classList.add('hidden');
    }
    
    // 显示聊天界面
    function showChatInterface() {
        authContainer.classList.add('hidden');
        chatApp.classList.remove('hidden');
        userEmail.textContent = currentUser.email;
        
        // 清空聊天记录 - 确保用户数据隔离
        clearChatHistory();
        
        // 加载用户的聊天会话
        loadChatSessions();
        
        connectWebSocket();
    }
    
    // 处理登出
    function handleLogout() {
        localStorage.removeItem('access_token');
        accessToken = null;
        currentUser = null;
        currentSessionId = null;
        chatSessions = [];
        
        if (ws) {
            ws.close();
            ws = null;
        }
        
        // 清空聊天记录 - 确保用户数据隔离
        clearChatHistory();
        clearChatHistoryList();
        
        showAuthInterface();
    }
    
    // 开始新对话
    function startNewChat() {
        currentSessionId = null;
        clearChatHistory();
        
        // 清除所有聊天记录项的active状态
        document.querySelectorAll('.chat-history-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // 聚焦到输入框
        questionInput.focus();
    }
    
    // 加载聊天会话
    async function loadChatSessions() {
        try {
            const response = await fetch('/api/conversations', {
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                },
            });
            
            if (response.ok) {
                const data = await response.json();
                chatSessions = data.conversations || [];
                renderChatSessionsList();
            } else {
                console.error('加载聊天会话失败');
            }
        } catch (error) {
            console.error('加载聊天会话错误:', error);
        }
    }
    
    // 渲染聊天会话列表
    function renderChatSessionsList() {
        chatHistoryList.innerHTML = '';
        
        if (chatSessions.length === 0) {
            chatHistoryList.innerHTML = '<div style="padding: 16px; text-align: center; color: #8e8ea0; font-size: 14px;">暂无聊天记录</div>';
            return;
        }
        
        chatSessions.forEach(session => {
            const sessionItem = document.createElement('div');
            sessionItem.className = 'chat-history-item';
            sessionItem.dataset.sessionId = session.id;
            
            const formatTime = (isoString) => {
                const date = new Date(isoString);
                const now = new Date();
                const diffMs = now - date;
                const diffMins = Math.floor(diffMs / 60000);
                const diffHours = Math.floor(diffMs / 3600000);
                const diffDays = Math.floor(diffMs / 86400000);
                
                if (diffMins < 1) return '刚刚';
                if (diffMins < 60) return `${diffMins}分钟前`;
                if (diffHours < 24) return `${diffHours}小时前`;
                if (diffDays < 7) return `${diffDays}天前`;
                return date.toLocaleDateString();
            };
            
            sessionItem.innerHTML = `
                <div class="chat-content">
                    <div class="chat-title">${session.title}</div>
                    <div class="chat-preview">${session.preview}</div>
                    <div class="chat-time">${formatTime(session.updated_at)} • ${session.message_count}条消息</div>
                </div>
                <div class="chat-actions">
                    <button class="delete-session-btn" title="删除对话">🗑️</button>
                </div>
            `;
            
            // 点击对话内容区域加载对话
            const chatContent = sessionItem.querySelector('.chat-content');
            chatContent.addEventListener('click', () => loadChatSession(session));
            
            // 点击删除按钮删除对话
            const deleteBtn = sessionItem.querySelector('.delete-session-btn');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // 阻止事件冒泡
                deleteChatSession(session);
            });
            chatHistoryList.appendChild(sessionItem);
        });
    }
    
    // 加载特定对话会话
    async function loadChatSession(session) {
        currentSessionId = session.id;
        
        // 更新active状态
        document.querySelectorAll('.chat-history-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-session-id="${session.id}"]`).classList.add('active');
        
        // 清空当前聊天并加载历史消息
        clearChatHistory();
        
        try {
            // 检查是否是新的会话类型
            if (session.session_type === 'chat_session') {
                // 使用新的API获取会话消息
                const response = await fetch(`/api/chat-sessions/${session.id}/messages`, {
                    headers: {
                        'Authorization': `Bearer ${accessToken}`,
                    },
                });
                
                if (response.ok) {
                    const data = await response.json();
                    
                    // 显示历史消息
                    data.messages.forEach(message => {
                        addMessage(message.content, message.role === 'user' ? 'user' : 'bot', message.id);
                    });
                    
                    console.log(`已加载会话: ${session.title} (${data.messages.length} 条消息)`);
                } else {
                    console.error('加载会话消息失败');
                    addMessage('加载历史消息失败', 'status');
                }
            } else {
                // 兼容旧的对话记录格式
                if (session.question && session.answer) {
                    addMessage(session.question, 'user');
                    addMessage(session.answer, 'bot');
                    console.log(`已加载旧对话: ${session.title}`);
                }
            }
        } catch (error) {
            console.error('加载对话时出错:', error);
            addMessage('加载历史消息失败', 'status');
        }
    }
    
    // 清空聊天记录列表
    function clearChatHistoryList() {
        chatHistoryList.innerHTML = '';
    }
    
    // 显示认证消息
    function showAuthMessage(message, type) {
        authMessage.textContent = message;
        authMessage.className = `auth-message ${type}`;
        authMessage.style.display = 'block';
    }
    
    // 清除认证消息
    function clearAuthMessage() {
        authMessage.style.display = 'none';
        authMessage.textContent = '';
        authMessage.className = 'auth-message';
    }
    
    // 清空聊天记录 - 确保用户数据隔离
    function clearChatHistory() {
        chatContainer.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">🤖</div>
                <h2>欢迎使用AI助手</h2>
                <p>我是AI-Jay，随时准备为您服务。您可以问我任何问题！</p>
            </div>
        `;
        currentBotMessageDiv = null;
    }
    
    // WebSocket连接
    function connectWebSocket() {
        if (!accessToken) return;
        
        ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = () => {
            console.log('WebSocket连接已建立');
            connectionStatus.textContent = '正在认证...';
            connectionStatus.className = 'connection-status disconnected';
            
            // 发送认证消息
            ws.send(JSON.stringify({
                type: 'auth',
                token: accessToken
            }));
        };
        
        ws.onmessage = (event) => {
            const eventData = JSON.parse(event.data);
            handleWebSocketMessage(eventData);
        };
        
        ws.onclose = () => {
            console.log('WebSocket连接已关闭');
            connectionStatus.textContent = '❌ 连接断开，3秒后尝试重连...';
            connectionStatus.className = 'connection-status disconnected';
            sendButton.disabled = true;
            setTimeout(() => {
                if (accessToken && currentUser) {
                    connectWebSocket();
                }
            }, 3000);
        };
        
        ws.onerror = (error) => console.error('WebSocket错误:', error);
    }
    
    // 处理WebSocket消息
    function handleWebSocketMessage(event) {
        console.log('收到WebSocket消息:', event); // 添加调试信息
        switch (event.type) {
            case 'auth_success':
                connectionStatus.textContent = '✅ 已连接';
                connectionStatus.className = 'connection-status connected';
                sendButton.disabled = false;
                break;
            case 'processing':
                console.log('处理processing事件:', event.data); // 添加调试信息
                if (event.data.session_id && !currentSessionId) {
                    // 新会话创建成功，更新当前会话ID
                    currentSessionId = event.data.session_id;
                }
                // 添加临时状态消息，标记为可删除
                addStatusMessage(`[${event.data.message}]`);
                break;
            case 'generation_start':
                // 清除所有临时状态消息
                clearStatusMessages();
                currentBotMessageDiv = addMessage('', 'bot');
                break;
            case 'generation_chunk':
                if (currentBotMessageDiv) {
                    const content = currentBotMessageDiv.querySelector('.message-content');
                    content.textContent += event.data.chunk;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                break;
            case 'generation_end':
            case 'complete':
                currentBotMessageDiv = null;
                sendButton.disabled = false;
                sendButton.innerHTML = '<span class="send-icon">➤</span>';
                
                // 如果有会话ID，更新当前会话ID
                if (event.data.session_id) {
                    currentSessionId = event.data.session_id;
                }
                
                // 重新加载聊天会话列表以显示新的对话或更新
                loadChatSessions();
                break;
            case 'error':
                // 清除临时状态消息
                clearStatusMessages();
                addMessage(`[错误]: ${event.data.error}`, 'status');
                sendButton.disabled = false;
                sendButton.innerHTML = '<span class="send-icon">➤</span>';
                break;
        }
    }
    
    // 添加消息到聊天容器
    function addMessage(content, type, messageId = null) {
        // 如果是第一条消息，清除欢迎界面
        const welcomeMessage = chatContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        if (messageId) {
            messageDiv.dataset.messageId = messageId;
        }
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        // 设置头像
        if (type === 'user') {
            avatarDiv.textContent = '👤';
        } else if (type === 'bot') {
            avatarDiv.textContent = '🤖';
        } else {
            avatarDiv.textContent = 'ℹ️';
        }
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        // 为用户消息和AI消息添加删除按钮（状态消息不添加）
        if ((type === 'user' || type === 'bot') && messageId) {
            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'message-actions';
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-message-btn';
            deleteBtn.innerHTML = '🗑️';
            deleteBtn.title = '删除消息';
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteMessage(messageId, messageDiv);
            });
            
            actionsDiv.appendChild(deleteBtn);
            messageDiv.appendChild(actionsDiv);
        }
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        return messageDiv;
    }
    
    // 添加临时状态消息（会在AI开始回复时自动删除）
    function addStatusMessage(content) {
        // 如果是第一条消息，清除欢迎界面
        const welcomeMessage = chatContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message status-message temp-status';
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.textContent = 'ℹ️';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        return messageDiv;
    }
    
    // 清除所有临时状态消息
    function clearStatusMessages() {
        const tempMessages = chatContainer.querySelectorAll('.temp-status');
        tempMessages.forEach(message => {
            message.remove();
        });
    }
    
    let currentBotMessageDiv = null;
    
    // 发送问题
    function sendQuestion() {
        const question = questionInput.value.trim();
        if (!question || !ws || ws.readyState !== WebSocket.OPEN) return;
        
        addMessage(question, 'user');
        
        // 发送消息，包含当前会话ID（如果有的话）
        const messageData = {
            type: 'question',
            content: question
        };
        
        if (currentSessionId) {
            messageData.session_id = currentSessionId;
        }
        
        ws.send(JSON.stringify(messageData));
        
        questionInput.value = '';
        sendButton.disabled = true;
        sendButton.innerHTML = '⏳';
        
        // 如果是新对话，清除其他对话的active状态
        if (!currentSessionId) {
            document.querySelectorAll('.chat-history-item').forEach(item => {
                item.classList.remove('active');
            });
        }
    }
    
    // 删除对话会话
    async function deleteChatSession(session) {
        if (!confirm(`确定要删除对话"${session.title}"吗？此操作无法撤销。`)) {
            return;
        }
        
        try {
            const response = await fetch(`/api/chat-sessions/${session.id}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                },
            });
            
            if (response.ok) {
                // 如果删除的是当前会话，清空聊天区域
                if (currentSessionId === session.id) {
                    currentSessionId = null;
                    clearChatHistory();
                }
                
                // 重新加载会话列表
                await loadChatSessions();
                
                console.log(`对话会话 "${session.title}" 删除成功`);
            } else {
                const data = await response.json();
                alert(`删除失败: ${data.detail || '未知错误'}`);
            }
        } catch (error) {
            console.error('删除对话会话错误:', error);
            alert('删除失败，请稍后重试');
        }
    }
    
    // 删除单条消息
    async function deleteMessage(messageId, messageElement) {
        if (!confirm('确定要删除这条消息吗？此操作无法撤销。')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/messages/${messageId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                },
            });
            
            if (response.ok) {
                // 从DOM中移除消息元素
                messageElement.remove();
                
                // 重新加载会话列表以更新消息计数
                await loadChatSessions();
                
                console.log(`消息 ${messageId} 删除成功`);
            } else {
                const data = await response.json();
                alert(`删除失败: ${data.detail || '未知错误'}`);
            }
        } catch (error) {
            console.error('删除消息错误:', error);
            alert('删除失败，请稍后重试');
        }
    }
    
    // 启动应用
    init();
})();