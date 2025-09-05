// static/main.js

(() => {
    // 全局状态
    let ws = null;
    let currentUser = null;
    let accessToken = null;
    let currentSessionId = null;
    let chatSessions = [];
    let currentUserMessageDiv = null; // 用于追踪新会话的用户消息DOM
    let currentBotMessageDiv = null;  // 用于追踪新会话的机器人消息DOM
    
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
    
    function init() {
        const savedToken = localStorage.getItem('access_token');
        if (savedToken) {
            accessToken = savedToken;
            verifyTokenAndShowChat();
        } else {
            showAuthInterface();
        }
        bindEvents();
    }
    
    function bindEvents() {
        loginTab.addEventListener('click', () => switchAuthTab('login'));
        registerTab.addEventListener('click', () => switchAuthTab('register'));
        loginForm.addEventListener('submit', handleLogin);
        registerForm.addEventListener('submit', handleRegister);
        logoutButton.addEventListener('click', handleLogout);
        newChatButton.addEventListener('click', startNewChat);
        sendButton.addEventListener('click', sendQuestion);
        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuestion();
            }
        });
    }
    
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
    
    async function handleLogin(e) {
        e.preventDefault();
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;
        
        try {
            showAuthMessage('正在登录...', 'info');
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
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
            showAuthMessage('网络错误，请稍后重试', 'error');
        }
    }
    
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
                headers: { 'Content-Type': 'application/json' },
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
            showAuthMessage('网络错误，请稍后重试', 'error');
        }
    }
    
    async function verifyTokenAndShowChat() {
        try {
            const response = await fetch('/api/me', { headers: { 'Authorization': `Bearer ${accessToken}` } });
            if (response.ok) {
                currentUser = await response.json();
                showChatInterface();
            } else {
                handleLogout();
            }
        } catch (error) {
            handleLogout();
        }
    }
    
    function showAuthInterface() {
        authContainer.classList.remove('hidden');
        chatApp.classList.add('hidden');
    }
    
    function showChatInterface() {
        authContainer.classList.add('hidden');
        chatApp.classList.remove('hidden');
        userEmail.textContent = currentUser.email;
        clearChatHistory();
        loadChatSessions();
        connectWebSocket();
    }
    
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
        clearChatHistory();
        clearChatHistoryList();
        showAuthInterface();
    }
    
    function startNewChat() {
        currentSessionId = null;
        clearChatHistory();
        document.querySelectorAll('.chat-history-item').forEach(item => item.classList.remove('active'));
        questionInput.focus();
    }
    
    async function loadChatSessions() {
        try {
            const response = await fetch('/api/conversations', { headers: { 'Authorization': `Bearer ${accessToken}` } });
            if (response.ok) {
                const data = await response.json();
                chatSessions = data.conversations || [];
                renderChatSessionsList();
            }
        } catch (error) {
            console.error('加载聊天会话错误:', error);
        }
    }
    
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
                if (diffMins < 1) return '刚刚';
                if (diffMins < 60) return `${diffMins}分钟前`;
                const diffHours = Math.floor(diffMs / 3600000);
                if (diffHours < 24) return `${diffHours}小时前`;
                const diffDays = Math.floor(diffMs / 86400000);
                if (diffDays < 7) return `${diffDays}天前`;
                return date.toLocaleDateString();
            };
            
            sessionItem.innerHTML = `
                <div class="chat-content">
                    <div class="chat-title">${session.title}</div>
                    <div class="chat-preview">${session.preview}</div>
                    <div class="chat-time">${formatTime(session.updated_at || session.created_at)} • ${session.message_count || ''}条消息</div>
                </div>
                <div class="chat-actions">
                    <button class="delete-session-btn" title="删除对话">🗑️</button>
                </div>
            `;
            
            sessionItem.querySelector('.chat-content').addEventListener('click', () => loadChatSession(session));
            sessionItem.querySelector('.delete-session-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteChatSession(session);
            });
            chatHistoryList.appendChild(sessionItem);
        });
    }
    
    async function loadChatSession(session) {
        // [FIX] 如果点击的是当前已加载的会话，则不执行任何操作
        if (currentSessionId === session.id) {
            console.log("已经是当前会话，无需重新加载。");
            return;
        }
        currentSessionId = session.id;
        
        document.querySelectorAll('.chat-history-item').forEach(item => item.classList.remove('active'));
        document.querySelector(`[data-session-id="${session.id}"]`).classList.add('active');
        
        clearChatHistory();
        
        try {
            const apiEndpoint = session.session_type === 'chat_session' ? `/api/chat-sessions/${session.id}/messages` : `/api/chat-sessions/${session.id}/messages`; // 统一使用新API
            const response = await fetch(apiEndpoint, { headers: { 'Authorization': `Bearer ${accessToken}` } });
            
            if (response.ok) {
                const data = await response.json();
                data.messages.forEach(message => addMessage(message.content, message.role === 'user' ? 'user' : 'bot', message.id));
            } else {
                addMessage('加载历史消息失败', 'status');
            }
        } catch (error) {
            addMessage('加载历史消息失败', 'status');
        }
    }
    
    function clearChatHistoryList() {
        chatHistoryList.innerHTML = '';
    }
    
    function showAuthMessage(message, type) {
        authMessage.textContent = message;
        authMessage.className = `auth-message ${type}`;
        authMessage.style.display = 'block';
    }
    
    function clearAuthMessage() {
        authMessage.style.display = 'none';
    }
    
    function clearChatHistory() {
        chatContainer.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">🤖</div>
                <h2>欢迎使用AI助手</h2>
                <p>我是AI-Jay，随时准备为您服务。您可以问我任何问题！</p>
            </div>
        `;
        currentBotMessageDiv = null;
        currentUserMessageDiv = null;
    }
    
    function connectWebSocket() {
        if (!accessToken) return;
        if (ws && ws.readyState === WebSocket.OPEN) return;
        
        ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = () => {
            connectionStatus.textContent = '正在认证...';
            ws.send(JSON.stringify({ type: 'auth', token: accessToken }));
        };
        
        ws.onmessage = (event) => handleWebSocketMessage(JSON.parse(event.data));
        
        ws.onclose = () => {
            connectionStatus.textContent = '❌ 连接断开，3秒后尝试重连...';
            connectionStatus.className = 'connection-status disconnected';
            sendButton.disabled = true;
            setTimeout(() => { if (accessToken && currentUser) connectWebSocket(); }, 3000);
        };
        
        ws.onerror = (error) => console.error('WebSocket错误:', error);
    }
    
    function handleWebSocketMessage(event) {
        switch (event.type) {
            case 'auth_success':
                connectionStatus.textContent = '✅ 已连接';
                connectionStatus.className = 'connection-status connected';
                sendButton.disabled = false;
                break;
            case 'processing':
                if (event.data.session_id && !currentSessionId) {
                    currentSessionId = event.data.session_id;
                }
                addStatusMessage(`[${event.data.message}]`);
                break;
            case 'generation_start':
                clearStatusMessages();
                currentBotMessageDiv = addMessage('', 'bot');
                break;
            case 'generation_chunk':
                if (currentBotMessageDiv) {
                    currentBotMessageDiv.querySelector('.message-content').textContent += event.data.chunk;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                break;
            case 'complete':
                const { user_message_id, ai_message_id } = event.data;
                if (user_message_id && currentUserMessageDiv) {
                    addDeleteButtonToMessage(currentUserMessageDiv, user_message_id);
                }
                if (ai_message_id && currentBotMessageDiv) {
                    addDeleteButtonToMessage(currentBotMessageDiv, ai_message_id);
                }
                currentUserMessageDiv = null;
                currentBotMessageDiv = null;
                sendButton.disabled = false;
                sendButton.innerHTML = '<span class="send-icon">➤</span>';
                if (event.data.session_id) {
                    currentSessionId = event.data.session_id;
                }
                loadChatSessions();
                break;
            case 'error':
                clearStatusMessages();
                addMessage(`[错误]: ${event.data.error}`, 'status');
                sendButton.disabled = false;
                sendButton.innerHTML = '<span class="send-icon">➤</span>';
                break;
        }
    }

    function addMessage(content, type, messageId = null) {
        const welcomeMessage = chatContainer.querySelector('.welcome-message');
        if (welcomeMessage) welcomeMessage.remove();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        if (messageId) messageDiv.dataset.messageId = messageId;
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${type === 'user' ? '👤' : (type === 'bot' ? '🤖' : 'ℹ️')}</div>
            <div class="message-content">${content}</div>
        `;
        
        if ((type === 'user' || type === 'bot') && messageId) {
            addDeleteButtonToMessage(messageDiv, messageId);
        }
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return messageDiv;
    }

    function addDeleteButtonToMessage(messageDiv, messageId) {
        if (!messageDiv || !messageId) return;
        messageDiv.dataset.messageId = messageId;
        
        if (messageDiv.querySelector('.message-actions')) return; // 避免重复添加

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
    
    function addStatusMessage(content) {
        const welcomeMessage = chatContainer.querySelector('.welcome-message');
        if (welcomeMessage) welcomeMessage.remove();
        const messageDiv = addMessage(content, 'status');
        messageDiv.classList.add('temp-status');
        return messageDiv;
    }
    
    function clearStatusMessages() {
        chatContainer.querySelectorAll('.temp-status').forEach(m => m.remove());
    }
    
    function sendQuestion() {
        const question = questionInput.value.trim();
        if (!question || !ws || ws.readyState !== WebSocket.OPEN) return;
        
        currentUserMessageDiv = addMessage(question, 'user');
        
        const messageData = { type: 'question', content: question };
        if (currentSessionId) messageData.session_id = currentSessionId;
        
        ws.send(JSON.stringify(messageData));
        
        questionInput.value = '';
        sendButton.disabled = true;
        sendButton.innerHTML = '⏳';
        
        if (!currentSessionId) {
            document.querySelectorAll('.chat-history-item').forEach(item => item.classList.remove('active'));
        }
    }
    
    async function deleteChatSession(session) {
        if (!confirm(`确定要删除对话"${session.title}"吗？`)) return;
        
        try {
            const response = await fetch(`/api/chat-sessions/${session.id}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${accessToken}` },
            });
            if (response.ok) {
                if (currentSessionId === session.id) startNewChat();
                await loadChatSessions();
            } else {
                const data = await response.json();
                alert(`删除失败: ${data.detail || '未知错误'}`);
            }
        } catch (error) {
            alert('删除失败，请稍后重试');
        }
    }
    
    async function deleteMessage(messageId, messageElement) {
        if (!confirm('确定要删除这条消息吗？')) return;
        
        try {
            const response = await fetch(`/api/messages/${messageId}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${accessToken}` },
            });
            if (response.ok) {
                messageElement.remove();
                await loadChatSessions();
            } else {
                const data = await response.json();
                alert(`删除失败: ${data.detail || '未知错误'}`);
            }
        } catch (error) {
            alert('删除失败，请稍后重试');
        }
    }
    
    init();
})();