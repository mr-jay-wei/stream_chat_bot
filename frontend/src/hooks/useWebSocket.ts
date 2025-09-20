import { useState, useEffect, useRef, useCallback } from 'react';

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
    
    // ws.current 将永远指向最新的 WebSocket 实例
    const ws = useRef<WebSocket | null>(null);
    
    const reconnectTimer = useRef<number | null>(null);

    // connect 函数现在不依赖任何外部变量，只负责连接逻辑
    const connect = useCallback(() => {
        if (!token) return;

        // 清理旧连接
        if (ws.current) {
            ws.current.onclose = null;
            ws.current.close();
        }

        const socket = new WebSocket(getWebSocketURL());
        ws.current = socket;

        socket.onopen = () => {
            console.log('WebSocket Connected');
            setIsConnected(true);
            socket.send(JSON.stringify({ type: 'auth', token }));
        };

        socket.onmessage = (event) => {
            try {
                const message: WebSocketEvent = JSON.parse(event.data);
                setLastMessage(message);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        socket.onerror = (error) => {
            console.error('WebSocket Error:', error);
            socket.close(); // 发生错误时主动关闭，会触发 onclose
        };

        socket.onclose = () => {
            console.log('WebSocket Disconnected');
            setIsConnected(false);

            // 只有当当前socket实例是ws.current指向的实例时，才进行重连
            // 这可以防止旧socket的onclose事件干扰新连接
            if (ws.current === socket) {
                if (reconnectTimer.current) {
                    clearTimeout(reconnectTimer.current);
                }
                if (token) {
                     reconnectTimer.current = window.setTimeout(() => {
                        console.log("Attempting to reconnect WebSocket...");
                        connect();
                    }, 3000);
                }
            }
        };
    }, [token]);

    useEffect(() => {
        connect();
        return () => {
            if (reconnectTimer.current) {
                clearTimeout(reconnectTimer.current);
            }
            if (ws.current) {
                ws.current.onclose = null; 
                ws.current.close();
            }
        };
    }, [connect]);

    // --- 核心修正：sendMessage 不再依赖于旧闭包 ---
    // sendMessage 函数在每次调用时，都直接从 ws.current 获取最新的socket实例
    const sendMessage = (message: object) => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(message));
        } else {
            console.error('WebSocket is not connected. Message not sent:', message);
            // 可以在这里增加一个消息队列，等重连成功后再发送
        }
    };

    return { isConnected, lastMessage, sendMessage };
};