import { useState, useEffect, useRef, useCallback } from 'react';

// --- 内联类型定义 ---
export interface WebSocketEvent {
    type: 'auth_success' | 'auth_error' | 'processing' | 'generation_start' | 'generation_chunk' | 'generation_end' | 'complete' | 'error';
    data: any;
}

const getWebSocketURL = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws`;
};

export const useWebSocket = (token: string | null) => {
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState<WebSocketEvent | null>(null);
    const ws = useRef<WebSocket | null>(null);

    const connect = useCallback(() => {
        if (!token || (ws.current && ws.current.readyState === WebSocket.OPEN)) {
            return;
        }

        ws.current = new WebSocket(getWebSocketURL());

        ws.current.onopen = () => {
            console.log('WebSocket Connected');
            setIsConnected(true);
            // 发送认证消息
            ws.current?.send(JSON.stringify({ type: 'auth', token }));
        };

        ws.current.onmessage = (event) => {
            try {
                const message: WebSocketEvent = JSON.parse(event.data);
                 if (message.type === 'auth_success') {
                    console.log("WebSocket Authenticated!");
                }
                setLastMessage(message);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        ws.current.onerror = (error) => {
            console.error('WebSocket Error:', error);
        };

        ws.current.onclose = () => {
            console.log('WebSocket Disconnected');
            setIsConnected(false);
            // 这里可以添加自动重连逻辑
            setTimeout(() => {
                console.log("Attempting to reconnect WebSocket...");
                connect();
            }, 3000); // 3秒后尝试重连
        };
    }, [token]);

    useEffect(() => {
        if (token) {
            connect();
        }
        return () => {
            if (ws.current) {
                // 清理onclose事件监听器，防止在组件卸载后还执行重连
                ws.current.onclose = null; 
                ws.current.close();
            }
        };
    }, [token, connect]);

    const sendMessage = (message: object) => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(message));
        } else {
            console.error('WebSocket is not connected.');
        }
    };

    return { isConnected, lastMessage, sendMessage };
};