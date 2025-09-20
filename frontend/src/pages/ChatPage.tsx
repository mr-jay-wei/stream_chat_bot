import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { useWebSocket } from '../hooks/useWebSocket';
import apiClient from '../api/apiClient';
import NewChatModal from '../components/NewChatModal';
import PromptsManagerModal from '../components/PromptsManagerModal';
import MessageItem, { Message } from '../components/MessageItem';
import { deleteMessage } from '../api/chat';

export interface ChatSession {
  id: number;
  title: string;
  user_id: number;
  created_at: string;
  updated_at: string;
  prompt_id: number | null;
}
export interface Prompt {
    id: number;
    user_id: number;
    name: string;
    content: string;
}
export interface WebSocketEvent {
    type: 'auth_success' | 'auth_error' | 'processing' | 'generation_start' | 'generation_chunk' | 'generation_end' | 'complete' | 'error';
    data: any;
}

const ChatPage: React.FC = () => {
    const { user, token, logout } = useAuth();
    const { isConnected, lastMessage, sendMessage } = useWebSocket(token);

    const [sessions, setSessions] = useState<ChatSession[]>([]);
    const [prompts, setPrompts] = useState<Prompt[]>([]);
    const [currentSessionId, setCurrentSessionId] = useState<number | null>(null);
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isSending, setIsSending] = useState(false);
    const [nextPromptId, setNextPromptId] = useState<number | null>(null);
    const [currentPrompt, setCurrentPrompt] = useState<Prompt | null>(null);

    const [isNewChatModalOpen, setIsNewChatModalOpen] = useState(false);
    const [isPromptsManagerModalOpen, setIsPromptsManagerModalOpen] = useState(false);
    
    const chatContainerRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const promptMap = new Map<number, string>();
    prompts.forEach(p => promptMap.set(p.id, p.name));
    promptMap.set(0, "哈基米");

    const fetchData = async () => {
        try {
            const [sessionsRes, promptsRes] = await Promise.all([
                apiClient.get<{ sessions: ChatSession[] }>('/chat-sessions'),
                apiClient.get<Prompt[]>('/prompts')
            ]);
            setSessions(sessionsRes.data.sessions);
            setPrompts(promptsRes.data);
        } catch (error) {
            console.error("Failed to fetch data", error);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);

    const loadSessionMessages = async (sessionId: number) => {
        try {
            const response = await apiClient.get<{ messages: Message[] }>(`/chat-sessions/${sessionId}/messages`);
            const sessionData = sessions.find(s => s.id === sessionId);
            
            setMessages(response.data.messages);
            setCurrentSessionId(sessionId);
            setNextPromptId(null);

            if (sessionData) {
                const prompt = prompts.find(p => p.id === sessionData.prompt_id);
                setCurrentPrompt(prompt || null);
            }
        } catch (error) {
            console.error("Failed to load session messages", error);
        }
    };
    
    useEffect(() => {
        if (!lastMessage) return;
        switch (lastMessage.type) {
            case 'processing':
                if (lastMessage.data.session_id && currentSessionId === null) {
                    const newSessionId = lastMessage.data.session_id;
                    setCurrentSessionId(newSessionId);
                    // 在收到新会话ID后，我们还需要更新左侧列表以包含这个新会话
                    // 同时，我们把前端临时创建的用户消息替换为后端返回的真实消息
                    setMessages(prev => prev.map(m => m.id > 0 ? m : {...m, chat_session_id: newSessionId}));
                    fetchData();
                }
                break;
            case 'generation_start':
                setMessages(prev => [...prev, { id: Date.now(), role: 'assistant', content: '', chat_session_id: currentSessionId! }]);
                break;
            case 'generation_chunk':
                setMessages(prev => {
                    const newMessages = [...prev];
                    const lastMsg = newMessages[newMessages.length - 1];
                    if (lastMsg && lastMsg.role === 'assistant') {
                        lastMsg.content += lastMessage.data.chunk;
                    }
                    return newMessages;
                });
                break;
            case 'complete':
                // AI消息完成后，后端会返回真实的消息ID，我们用它来更新
                if (lastMessage.data.ai_message_id) {
                    setMessages(prev => prev.map(m => (m.content === lastMessage.data.final_content && m.role === 'assistant') ? { ...m, id: lastMessage.data.ai_message_id } : m));
                }
                setIsSending(false);
                break;
            case 'error':
                 alert(`发生错误: ${lastMessage.data.error}`);
                 setIsSending(false);
                 break;
        }
    }, [lastMessage, currentSessionId, prompts]);

    useEffect(() => {
        chatContainerRef.current?.scrollTo(0, chatContainerRef.current.scrollHeight);
    }, [messages]);

    const handleSend = () => {
        if (!input.trim() || isSending) return;
        const tempId = Date.now(); // 使用一个临时的唯一ID
        const userMessage: Message = { id: tempId, role: 'user', content: input, chat_session_id: currentSessionId! };
        setMessages(prev => [...prev, userMessage]);
        const messagePayload: { type: string; content: string; session_id: number | null; prompt_id?: number | null } = { type: 'question', content: input, session_id: currentSessionId };
        if (currentSessionId === null) {
            messagePayload.prompt_id = nextPromptId;
            const prompt = prompts.find(p => p.id === nextPromptId);
            setCurrentPrompt(prompt || null);
        }
        sendMessage(messagePayload);
        setInput('');
        setIsSending(true);
        setNextPromptId(null);
    };

    const startNewChat = (promptId: number | null) => {
        setCurrentSessionId(null);
        setMessages([]);
        setNextPromptId(promptId);
        setIsNewChatModalOpen(false);
        inputRef.current?.focus();
        const prompt = prompts.find(p => p.id === promptId);
        setCurrentPrompt(prompt || null);
    };
    
    const handleDeleteSession = async (sessionId: number) => {
        if (window.confirm("确定要删除这个对话吗？")) {
            try {
                await apiClient.delete(`/chat-sessions/${sessionId}`);
                if (currentSessionId === sessionId) {
                    setCurrentSessionId(null);
                    setMessages([]);
                    setCurrentPrompt(null);
                }
                fetchData();
            } catch (error) {
                alert("删除失败");
            }
        }
    };
    
    const handleDeleteMessage = async (messageId: number) => {
        try {
            await deleteMessage(messageId);
            setMessages(prevMessages => prevMessages.filter(msg => msg.id !== messageId));
            fetchData();
        } catch (error) {
            console.error('Failed to delete message:', error);
            alert('删除消息失败');
        }
    };

    const currentChatTitle = currentPrompt ? currentPrompt.name : (currentSessionId !== null ? '哈基米' : '哈基米');
    
    const renderMessages = () => {
        return messages.map((msg, index) => {
            const showAvatar = index === 0 || messages[index - 1].role !== msg.role;
            return (
                <MessageItem
                    key={msg.id || index}
                    message={msg}
                    showAvatar={showAvatar}
                    onDelete={handleDeleteMessage}
                />
            );
        });
    };

    return (
        <div className="chat-app">
            <div className="sidebar">
                <div className="sidebar-header">
                    <button className="sidebar-btn" onClick={() => setIsNewChatModalOpen(true)}>+ 新建对话</button>
                    <button className="sidebar-btn" onClick={() => setIsPromptsManagerModalOpen(true)}>⚙️ 管理角色</button>
                </div>
                <div className="chat-history">
                    <div className="chat-history-header">聊天记录</div>
                    <div className="chat-history-list">
                        {sessions.map(session => (
                            <div key={session.id} className={`chat-history-item ${currentSessionId === session.id ? 'active' : ''}`} onClick={() => loadSessionMessages(session.id)}>
                                <div className="chat-item-content">
                                    <div className="chat-title">{session.title}</div>
                                    <div className="chat-prompt-tag">
                                        {promptMap.get(session.prompt_id || 0) || '哈基米'}
                                    </div>
                                </div>
                                <button className="delete-session-btn" onClick={(e) => {e.stopPropagation(); handleDeleteSession(session.id)}}>🗑️</button>
                            </div>
                        ))}
                    </div>
                </div>
                <div className="sidebar-footer">
                    <div className="user-info">
                        <div className="user-email">{user?.email}</div>
                        <button onClick={logout} className="logout-button">登出</button>
                    </div>
                </div>
            </div>

            <div className="main-content">
                <div className="chat-header">
                    <h1>
                        <img src="/images/my-logo.png" alt="Logo" className="header-logo" /> 
                        {currentChatTitle}
                    </h1>
                    <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>{isConnected ? '✅ 已连接' : '❌ 连接断开'}</div>
                </div>
                <div className="chat-container" ref={chatContainerRef}>
                    {messages.length === 0 ? (
                        <div className="welcome-message">
                            <img src="/images/my-logo.png" alt="Welcome Logo" className="welcome-logo" />
                            <p>{nextPromptId !== null ? `正在与 ${currentPrompt?.name || '哈基米'} 开始新对话，请输入...` : "我是哈基米，选择一个对话或新建对话开始吧！"}</p>
                        </div>
                    ) : (
                        renderMessages()
                    )}
                </div>
                <div className="input-container">
                    <div className="input-wrapper">
                        <input ref={inputRef} type="text" id="questionInput" placeholder="请输入您的问题..." value={input} onChange={(e) => setInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSend()} />
                        <button id="sendButton" onClick={handleSend} disabled={isSending || !input.trim()}>➤</button>
                    </div>
                </div>
            </div>

            {isNewChatModalOpen && ( <NewChatModal prompts={prompts} onClose={() => setIsNewChatModalOpen(false)} onSelectPrompt={startNewChat} /> )}
            {isPromptsManagerModalOpen && ( <PromptsManagerModal onClose={() => { setIsPromptsManagerModalOpen(false); fetchData(); }} /> )}
        </div>
    );
};

export default ChatPage;