// mobile/services/WebSocketClient.js
import 'react-native-url-polyfill/auto';
import apiClient from '../api/index';

const WEBSOCKET_URL = apiClient.defaults.baseURL.replace('http', 'ws').replace('/api', '/ws');

class WebSocketClient {
  // ... 内部代码完全不变 ...
  constructor() {
    this.ws = null;
    this.listeners = {};
  }
  on(event, callback) {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
  }
  removeListener(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event] = this.listeners[event].filter(l => l !== callback);
    }
  }
  emit(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => callback(data));
    }
  }
  connect(token) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
    this.ws = new WebSocket(WEBSOCKET_URL);
    this.ws.onopen = () => {
      this.emit('open');
      this.sendMessage({ type: 'auth', token });
    };
    this.ws.onmessage = event => {
      try {
        this.emit('message', JSON.parse(event.data));
      } catch (e) {
        console.error('Parse error', e);
      }
    };
    this.ws.onerror = error => {
      this.emit('error', error);
    };
    this.ws.onclose = event => {
      this.emit('close');
      this.ws = null;
    };
  }
  sendMessage(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.error('WS not connected.');
    }
  }
  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// 【改动点】: 创建实例并使用命名导出
export const webSocketClient = new WebSocketClient();
