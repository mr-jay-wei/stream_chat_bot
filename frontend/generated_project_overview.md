# 项目概览: frontend

本文档由`generate_project_overview.py`自动生成，包含了项目的结构树和所有可读文件的内容。

## 项目结构

```
frontend/
├── public
│   └── images
├── src
│   ├── api
│   │   ├── apiClient.ts
│   │   └── chat.ts
│   ├── assets
│   ├── components
│   │   ├── MessageItem.tsx
│   │   ├── Modal.tsx
│   │   ├── NewChatModal.tsx
│   │   └── PromptsManagerModal.tsx
│   ├── context
│   │   └── AuthContext.tsx
│   ├── hooks
│   │   └── useWebSocket.ts
│   ├── pages
│   │   ├── AuthPage.tsx
│   │   └── ChatPage.tsx
│   ├── App.css
│   ├── App.tsx
│   ├── index.css
│   ├── main.tsx
│   ├── style.css
│   └── vite-env.d.ts
├── .gitignore
├── eslint.config.js
├── index.html
├── package.json
├── README.md
├── tsconfig.app.json
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
```

---

# 文件内容

## `.gitignore`

```
# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
lerna-debug.log*

node_modules
dist
dist-ssr
*.local

# Editor directories and files
.vscode/*
!.vscode/extensions.json
.idea
.DS_Store
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?

```

## `eslint.config.js`

```javascript
import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs['recommended-latest'],
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
  },
])

```

## `index.html`

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vite + React + TS</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>

```

## `package.json`

```json
{
  "name": "frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "lint": "eslint .",
    "preview": "vite preview"
  },
  "dependencies": {
    "axios": "^1.12.2",
    "react": "^19.1.1",
    "react-dom": "^19.1.1"
  },
  "devDependencies": {
    "@eslint/js": "^9.35.0",
    "@types/react": "^19.1.13",
    "@types/react-dom": "^19.1.9",
    "@vitejs/plugin-react": "^5.0.2",
    "eslint": "^9.35.0",
    "eslint-plugin-react-hooks": "^5.2.0",
    "eslint-plugin-react-refresh": "^0.4.20",
    "globals": "^16.4.0",
    "typescript": "~5.8.3",
    "typescript-eslint": "^8.43.0",
    "vite": "^7.1.6"
  }
}

```

## `README.md`

````text
\# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

#\# Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

\`\`\`js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
\`\`\`

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

\`\`\`js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
\`\`\`

````

## `src/api/apiClient.ts`

```typescript
import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api', // Vite会帮我们代理到后端
});

// 请求拦截器：在每次发送请求前，都检查一下有没有token，有就带上
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export default apiClient;
```

## `src/api/chat.ts`

```typescript
// frontend/src/api/chat.ts

import apiClient from './apiClient';

// 删除单条消息的函数
export const deleteMessage = (messageId: number) => {
  return apiClient.delete(`/messages/${messageId}`);
};
```

## `src/App.css`

```css
#root {
  max-width: 1280px;
  margin: 0 auto;
  padding: 2rem;
  text-align: center;
}

.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.react:hover {
  filter: drop-shadow(0 0 2em #61dafbaa);
}

@keyframes logo-spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

@media (prefers-reduced-motion: no-preference) {
  a:nth-of-type(2) .logo {
    animation: logo-spin infinite 20s linear;
  }
}

.card {
  padding: 2em;
}

.read-the-docs {
  color: #888;
}

```

## `src/App.tsx`

```
import React from 'react';
import { useAuth } from './context/AuthContext';
import AuthPage from './pages/AuthPage';
import ChatPage from './pages/ChatPage';

function AppContent() {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        fontSize: '1.5rem',
        color: '#555',
      }}>
        正在加载...
      </div>
    );
  }

  return user ? <ChatPage /> : <AuthPage />;
}

function App() {
  // AppContent 会通过 useAuth() 自动从 main.tsx 注入的 AuthProvider 获取状态
  return <AppContent />;
}

export default App;
```

## `src/components/MessageItem.tsx`

```
// frontend/src/components/MessageItem.tsx

import React from 'react';

// --- 类型定义 ---
export interface Message {
  id: number;
  chat_session_id: number;
  role: 'user' | 'assistant';
  content: string;
}

interface MessageItemProps {
  message: Message;
  showAvatar: boolean;
  onDelete: (messageId: number) => void;
}

const MessageItem: React.FC<MessageItemProps> = ({ message, showAvatar, onDelete }) => {
  const messageClass = `message ${message.role}-message ${showAvatar ? '' : 'no-avatar'}`;

  // 只有当消息ID是数字时（意味着它已经保存在数据库），才显示删除按钮
  const canBeDeleted = typeof message.id === 'number' && message.id > 0;

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // 防止触发其他点击事件
    if (window.confirm('确定要删除这条消息吗？')) {
      onDelete(message.id);
    }
  };

  return (
    <div className={messageClass}>
      <div className="message-avatar">
        {showAvatar && (
          message.role === 'user' ? '👤' : <img src="/images/my-logo.png" alt="Bot" className="avatar-logo" />
        )}
      </div>
      <div className="message-content">
        {message.content}
        {canBeDeleted && (
          <button className="delete-message-btn" title="删除消息" onClick={handleDeleteClick}>
            🗑️
          </button>
        )}
      </div>
    </div>
  );
};

export default MessageItem;
```

## `src/components/Modal.tsx`

```
import React, { ReactNode } from 'react';

interface ModalProps {
  title: string;
  children: ReactNode;
  onClose: () => void;
}

const Modal: React.FC<ModalProps> = ({ title, children, onClose }) => {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{title}</h2>
          <button onClick={onClose} className="modal-close-btn">&times;</button>
        </div>
        <div className="modal-body">
          {children}
        </div>
      </div>
    </div>
  );
};

export default Modal;
```

## `src/components/NewChatModal.tsx`

```
import React from 'react';
import Modal from './Modal';

// --- 内联类型定义 ---
export interface Prompt {
    id: number;
    user_id: number;
    name: string;
    content: string;
}

interface NewChatModalProps {
  prompts: Prompt[];
  onClose: () => void;
  onSelectPrompt: (promptId: number | null) => void;
}

const NewChatModal: React.FC<NewChatModalProps> = ({ prompts, onClose, onSelectPrompt }) => {
  return (
    <Modal title="选择一个角色开始新对话" onClose={onClose}>
      <div className="prompt-list">
        <div className="prompt-item" onClick={() => onSelectPrompt(null)}>
          <div className="prompt-name">哈基米</div>
          <div className="prompt-content">使用系统默认的哈基米助手。</div>
        </div>
        {prompts.map(prompt => (
          <div key={prompt.id} className="prompt-item" onClick={() => onSelectPrompt(prompt.id)}>
            <div className="prompt-name">{prompt.name}</div>
            <div className="prompt-content">{prompt.content.substring(0, 100)}...</div>
          </div>
        ))}
      </div>
    </Modal>
  );
};

export default NewChatModal;
```

## `src/components/PromptsManagerModal.tsx`

```
// frontend/src/components/PromptsManagerModal.tsx

import React, { useState, useEffect } from 'react';
import Modal from './Modal';
import apiClient from '../api/apiClient';

// (类型定义部分保持不变)
export interface Prompt {
    id: number;
    user_id: number;
    name: string;
    content: string;
}

const PromptsManagerModal: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [editingPrompt, setEditingPrompt] = useState<Partial<Prompt> | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchPrompts = async () => {
    setIsLoading(true);
    try {
        const response = await apiClient.get<Prompt[]>('/prompts');
        setPrompts(response.data);
    } catch (error) {
        console.error("Failed to fetch prompts", error);
        alert("加载角色列表失败");
    } finally {
        setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPrompts();
  }, []);

  const handleSave = async () => {
    if (!editingPrompt || !editingPrompt.name?.trim() || !editingPrompt.content?.trim()) {
      alert('角色名称和设定不能为空');
      return;
    }
    try {
      if (editingPrompt.id) {
        await apiClient.put(`/prompts/${editingPrompt.id}`, { name: editingPrompt.name, content: editingPrompt.content });
      } else {
        await apiClient.post('/prompts', { name: editingPrompt.name, content: editingPrompt.content });
      }
      setEditingPrompt(null);
      fetchPrompts();
    } catch (error) {
      alert('保存失败');
    }
  };

  const handleDelete = async (id: number) => {
    if (window.confirm('确定要删除这个角色吗? 这将永久移除它。')) {
      try {
        await apiClient.delete(`/prompts/${id}`);
        fetchPrompts();
      } catch (error) {
        alert('删除失败');
      }
    }
  };

  return (
    <Modal title="管理我的角色" onClose={onClose}>
      {editingPrompt ? (
        <div className="prompt-form">
          <input
            type="text"
            placeholder="角色名称"
            value={editingPrompt.name || ''}
            onChange={(e) => setEditingPrompt({ ...editingPrompt, name: e.target.value })}
          />
          <textarea
            placeholder="角色设定 (例如：你是一位严格的雅思口语考官...)"
            value={editingPrompt.content || ''}
            onChange={(e) => setEditingPrompt({ ...editingPrompt, content: e.target.value })}
          />
          <div className="prompt-form-actions">
            <button className="cancel-btn" onClick={() => setEditingPrompt(null)}>取消</button>
            <button className="save-btn" onClick={handleSave}>保存</button>
          </div>
        </div>
      ) : (
        <>
          {/* --- 关键改动：使用了新的 btn-primary 样式 --- */}
          <button className="btn-primary" onClick={() => setEditingPrompt({ name: '', content: '' })}>+ 新建角色</button>
          {isLoading ? (
              <p style={{textAlign: 'center', margin: '20px'}}>正在加载...</p>
          ) : (
            <div className="prompt-list" style={{marginTop: '20px'}}>
                {prompts.length === 0 ? (
                    <p style={{textAlign: 'center', color: '#666'}}>你还没有创建任何角色。</p>
                ) : (
                    prompts.map(prompt => (
                    <div key={prompt.id} className="prompt-item">
                        <div className="prompt-item-header">
                            <div className="prompt-name">{prompt.name}</div>
                            <div className="prompt-actions">
                                <button title="编辑" onClick={() => setEditingPrompt(prompt)}>✏️</button>
                                <button title="删除" onClick={() => handleDelete(prompt.id)}>🗑️</button>
                            </div>
                        </div>
                        <div className="prompt-content">{prompt.content}</div>
                    </div>
                    ))
                )}
            </div>
          )}
        </>
      )}
    </Modal>
  );
};

export default PromptsManagerModal;
```

## `src/context/AuthContext.tsx`

```
import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';
import apiClient from '../api/apiClient';

// 类型定义只存在于此文件内部，不对外导出
interface User {
  id: number;
  email: string;
}

// ---------------- 其他代码完全不变 ----------------

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  login: (token: string, user: User) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const bootstrapAuth = async () => {
      const storedToken = localStorage.getItem('access_token');
      if (storedToken) {
        try {
          apiClient.defaults.headers.common['Authorization'] = `Bearer ${storedToken}`;
          const response = await apiClient.get<User>('/me');
          setUser(response.data);
          setToken(storedToken);
        } catch (error) {
          console.error("Token is invalid, cleaning up.", error);
          localStorage.removeItem('access_token');
          delete apiClient.defaults.headers.common['Authorization'];
        }
      }
      setIsLoading(false);
    };
    bootstrapAuth();
  }, []);

  const login = (newToken: string, newUser: User) => {
    localStorage.setItem('access_token', newToken);
    apiClient.defaults.headers.common['Authorization'] = `Bearer ${newToken}`;
    setToken(newToken);
    setUser(newUser);
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    delete apiClient.defaults.headers.common['Authorization'];
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
```

## `src/hooks/useWebSocket.ts`

```typescript
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
```

## `src/index.css`

```css
:root {
  font-family: system-ui, Avenir, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  font-weight: 400;

  color-scheme: light dark;
  color: rgba(255, 255, 255, 0.87);
  background-color: #242424;

  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

a {
  font-weight: 500;
  color: #646cff;
  text-decoration: inherit;
}
a:hover {
  color: #535bf2;
}

body {
  margin: 0;
  display: flex;
  place-items: center;
  min-width: 320px;
  min-height: 100vh;
}

h1 {
  font-size: 3.2em;
  line-height: 1.1;
}

button {
  border-radius: 8px;
  border: 1px solid transparent;
  padding: 0.6em 1.2em;
  font-size: 1em;
  font-weight: 500;
  font-family: inherit;
  background-color: #1a1a1a;
  cursor: pointer;
  transition: border-color 0.25s;
}
button:hover {
  border-color: #646cff;
}
button:focus,
button:focus-visible {
  outline: 4px auto -webkit-focus-ring-color;
}

@media (prefers-color-scheme: light) {
  :root {
    color: #213547;
    background-color: #ffffff;
  }
  a:hover {
    color: #747bff;
  }
  button {
    background-color: #f9f9f9;
  }
}

```

## `src/main.tsx`

```
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './style.css'
import { AuthProvider } from './context/AuthContext'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AuthProvider>
      <App />
    </AuthProvider>
  </React.StrictMode>,
)
```

## `src/pages/AuthPage.tsx`

```
import React, { useState } from 'react';
import apiClient from '../api/apiClient';
import { useAuth } from '../context/AuthContext';

// 在这里为 AuthPage.tsx 自己定义 User 类型
interface User {
  id: number;
  email: string;
}

type AuthMode = 'login' | 'register';

const AuthPage: React.FC = () => {
  const [mode, setMode] = useState<AuthMode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage(null);

    if (mode === 'register' && password !== confirmPassword) {
      setMessage({ text: '两次输入的密码不一致', type: 'error' });
      return;
    }

    const url = mode === 'login' ? '/login' : '/register';
    
    try {
      const response = await apiClient.post<{ access_token: string; user_email: string }>(url, { email, password });
      const { access_token } = response.data;
      
      const authApi = apiClient;
      authApi.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
      const meResponse = await authApi.get<User>('/me');

      login(access_token, meResponse.data);
      setMessage({ text: `${mode === 'login' ? '登录' : '注册'}成功！`, type: 'success' });

    } catch (error: any) {
      setMessage({ text: error.response?.data?.detail || '操作失败', type: 'error' });
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-form">
        <h1>
            <img src="/images/my-logo.png" alt="Logo" className="header-logo" />
            哈基米
        </h1>
        <div className="auth-tabs">
          <button className={`auth-tab ${mode === 'login' ? 'active' : ''}`} onClick={() => setMode('login')}>登录</button>
          <button className={`auth-tab ${mode === 'register' ? 'active' : ''}`} onClick={() => setMode('register')}>注册</button>
        </div>
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="email">邮箱</label>
            <input type="email" id="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </div>
          <div className="form-group">
            <label htmlFor="password">密码</label>
            <input type="password" id="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
          </div>
          {mode === 'register' && (
            <div className="form-group">
              <label htmlFor="confirmPassword">确认密码</label>
              <input type="password" id="confirmPassword" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} required />
            </div>
          )}
          <button type="submit" className="auth-button">{mode === 'login' ? '登录' : '注册'}</button>
        </form>
        {message && <div className={`auth-message ${message.type}`}>{message.text}</div>}
      </div>
    </div>
  );
};

export default AuthPage;
```

## `src/pages/ChatPage.tsx`

```
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
```

## `src/style.css`

```css
/* src/style.css */

:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --background-light: #f7f7f8;
    --background-dark: #202123;
    --text-light: #ffffff;
    --text-dark: #333333;
    --border-color: #e5e5e5;
  }
  
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: var(--background-light);
    height: 100vh;
    overflow: hidden;
  }
  
  #root {
    height: 100%;
  }
  
  .hidden {
    display: none !important;
  }
  
  /* 认证界面样式 */
  .auth-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
  }
  
  .auth-form {
    background: white;
    border-radius: 12px;
    padding: 40px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    width: 100%;
    max-width: 400px;
  }
  
  .auth-form h1 {
    color: var(--text-dark);
    text-align: center;
    margin-bottom: 30px;
    font-size: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .auth-tabs {
    display: flex;
    margin-bottom: 30px;
    border-bottom: 1px solid var(--border-color);
  }
  
  .auth-tab {
    flex: 1;
    padding: 12px;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 16px;
    color: #666;
    border-bottom: 2px solid transparent;
    transition: all 0.3s;
  }
  
  .auth-tab.active {
    color: var(--primary-color);
    border-bottom-color: var(--primary-color);
  }
  
  .form-group {
    margin-bottom: 20px;
  }
  
  .form-group label {
    display: block;
    margin-bottom: 5px;
    color: var(--text-dark);
    font-weight: 500;
  }
  
  .form-group input {
    width: 100%;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: 16px;
  }
  
  .auth-button {
    width: 100%;
    padding: 12px;
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
    font-weight: 500;
  }
  
  .auth-message {
    margin-top: 15px;
    padding: 10px;
    border-radius: 6px;
    text-align: center;
    font-size: 14px;
  }
  
  .auth-message.success {
    background-color: #d4edda;
    color: #155724;
  }
  
  .auth-message.error {
    background-color: #f8d7da;
    color: #721c24;
  }
  
  
  /* 聊天应用布局 */
  .chat-app {
    display: flex;
    height: 100vh;
  }
  
  /* 左侧边栏 */
  .sidebar {
    width: 280px;
    background: var(--background-dark);
    color: var(--text-light);
    display: flex;
    flex-direction: column;
    border-right: 1px solid #4d4d4f;
  }
  
  .sidebar-header {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    border-bottom: 1px solid #4d4d4f;
  }
  
  .sidebar-btn {
    width: 100%;
    padding: 12px;
    background: transparent;
    color: white;
    border: 1px solid #4d4d4f;
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-size: 14px;
    transition: background-color 0.2s;
  }
  
  .sidebar-btn:hover {
    background: #40414f;
  }
  
  .chat-history {
    flex: 1;
    overflow-y: auto;
  }
  
  .chat-history-header {
    padding: 16px;
    font-size: 14px;
    color: #8e8ea0;
    font-weight: 500;
  }
  
  .chat-history-list {
    padding: 8px;
  }
  
  .chat-history-item {
    display: flex;
    align-items: center;
    padding: 12px;
    margin-bottom: 4px;
    border-radius: 6px;
    transition: background-color 0.2s;
    cursor: pointer;
    position: relative;
  }
  
  .chat-history-item:hover {
    background: #40414f;
  }
  
  .chat-history-item.active {
    background: #40414f;
  }
  
  .chat-history-item .chat-title {
    font-size: 14px;
    color: white;
    margin-bottom: 4px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  
  .delete-session-btn {
    background: none;
    border: none;
    color: #8e8ea0;
    cursor: pointer;
    font-size: 14px;
    opacity: 0;
    transition: all 0.2s;
    position: absolute;
    right: 10px;
    top: 50%;
    transform: translateY(-50%);
  }
  .chat-history-item:hover .delete-session-btn {
      opacity: 1;
  }
  .delete-session-btn:hover {
      color: #ff4444;
  }
  
  
  .sidebar-footer {
    padding: 16px;
    border-top: 1px solid #4d4d4f;
  }
  
  .user-info {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .user-email {
      font-size: 14px;
      color: white;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1;
  }
  
  .logout-button {
    background: none;
    border: none;
    color: #8e8ea0;
    cursor: pointer;
    font-size: 12px;
    padding: 0;
  }
  
  
  /* 主聊天区域 */
  .main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: white;
  }
  
  .chat-header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .chat-header h1 {
    font-size: 20px;
    display: flex;
    align-items: center;
  }
  
  .connection-status {
    font-size: 12px;
    font-weight: 500;
  }
  
  .connected { color: #155724; }
  .disconnected { color: #721c24; }
  
  
  .chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    background: var(--background-light);
  }
  
  .welcome-message {
    text-align: center;
    padding: 60px 20px;
    color: #666;
  }
  .welcome-logo {
    height: 64px;
    width: 64px;
    margin: 0 auto 16px;
  }
  
  .message {
    margin-bottom: 24px;
    display: flex;
    gap: 12px;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
  }
  
  .message-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
  }
  
  .user-message {
    flex-direction: row-reverse;
  }
  .user-message .message-avatar {
    background: var(--primary-color);
    color: white;
  }
  
  .bot-message .message-avatar {
    background: #10a37f;
  }
  .avatar-logo {
    height: 100%;
    width: 100%;
    border-radius: 50%;
  }
  .bot-message .message-avatar {
      background: transparent;
  }
  
  
  .message-content {
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.5;
    word-wrap: break-word;
    white-space: pre-wrap;
  }
  
  .user-message .message-content {
    background: var(--primary-color);
    color: white;
  }
  
  .bot-message .message-content {
    background: white;
    color: var(--text-dark);
    border: 1px solid var(--border-color);
  }
  .status-message {
      justify-content: center;
      color: #92400e;
      font-style: italic;
  }
  
  
  .input-container {
    padding: 24px;
    background: white;
    border-top: 1px solid var(--border-color);
  }
  
  .input-wrapper {
    max-width: 800px;
    margin: 0 auto;
    display: flex;
    gap: 12px;
  }
  
  #questionInput {
    flex: 1;
    padding: 12px 16px;
    border: 1px solid #d1d5db;
    border-radius: 24px;
    font-size: 16px;
  }
  
  #sendButton {
    width: 48px;
    height: 48px;
    background: var(--primary-color);
    color: white;
    border: none;
    border-radius: 50%;
    cursor: pointer;
  }
  
  #sendButton:disabled {
    background: #d1d5db;
    cursor: not-allowed;
  }
  
  .header-logo {
    height: 28px;
    width: 28px;
    margin-right: 12px;
  }
  
  /* Modal 样式 */
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
  }
  
  .modal-content {
    background: white;
    padding: 20px;
    border-radius: 8px;
    width: 90%;
    max-width: 600px;
    max-height: 80vh;
    display: flex;
    flex-direction: column;
  }
  
  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
    margin-bottom: 20px;
  }
  
  .modal-header h2 {
    font-size: 1.2rem;
  }
  
  .modal-close-btn {
    background: none;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
  }
  
  .modal-body {
    overflow-y: auto;
    flex: 1;
  }
  
  .prompt-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  
  .prompt-item {
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 6px;
    cursor: pointer;
    transition: background-color 0.2s;
  }
  
  .prompt-item:hover {
    background-color: #f0f0f0;
  }
  
  .prompt-item-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
  }
  .prompt-name {
    font-weight: bold;
  }
  .prompt-actions button {
    margin-left: 10px;
    background: none;
    border: none;
    cursor: pointer;
  }
  
  .prompt-content {
    font-size: 0.9rem;
    color: #555;
    white-space: pre-wrap;
  }
  
  .prompt-form {
    display: flex;
    flex-direction: column;
    gap: 15px;
  }
  .prompt-form input,
  .prompt-form textarea {
    width: 100%;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 1rem;
  }
  .prompt-form textarea {
    min-height: 200px;
    resize: vertical;
  }
  .prompt-form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
  }
  .prompt-form-actions button {
    padding: 10px 20px;
    border-radius: 5px;
    border: none;
    cursor: pointer;
  }
  .save-btn {
    background-color: var(--primary-color);
    color: white;
  }
  .cancel-btn {
    background-color: #ccc;
  }

  .btn-primary {
    background-color: var(--primary-color);
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    border: none;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 500;
    transition: background-color 0.2s;
  }

  .btn-primary:hover {
      background-color: var(--secondary-color);
  }

  .chat-item-content {
    flex-grow: 1;
    min-width: 0; /* 防止内容溢出 */
  }

  /* --- 新增样式: 角色标签 --- */
  .chat-prompt-tag {
      font-size: 11px;
      color: #a0a0a0;
      margin-top: 4px;
      background-color: #40414f;
      padding: 2px 6px;
      border-radius: 4px;
      align-self: flex-start; /* 让标签宽度自适应内容 */
      display: inline-block; /* 同样为了宽度自适应 */
  }

  .message.no-avatar {
    padding-left: 44px; /* 32px的头像宽度 + 12px的间距 */
  }

  .message-content {
    position: relative;
    padding-right: 30px; /* 为删除按钮留出空间 */
  }

  /* --- 新增样式：删除单条消息的按钮 --- */
  .delete-message-btn {
      position: absolute;
      top: 5px;
      right: 5px;
      background: rgba(0, 0, 0, 0.1);
      border: none;
      color: #666;
      cursor: pointer;
      padding: 2px 4px;
      border-radius: 4px;
      font-size: 12px;
      opacity: 0; /* 默认隐藏 */
      transition: opacity 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
  }

  /* 鼠标悬浮在消息上时，显示删除按钮 */
  .message:hover .delete-message-btn {
      opacity: 1;
  }

  .delete-message-btn:hover {
      background: #ff4444;
      color: white;
  }

  /* 用户消息的删除按钮样式微调 */
  .user-message .delete-message-btn {
      background: rgba(255, 255, 255, 0.2);
      color: rgba(255, 255, 255, 0.8);
  }

  .user-message .delete-message-btn:hover {
      background: #ff4444;
      color: white;
  }
```

## `src/vite-env.d.ts`

```typescript
/// <reference types="vite/client" />

```

## `tsconfig.app.json`

```json
{
  "compilerOptions": {
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.app.tsbuildinfo",
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "verbatimModuleSyntax": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "erasableSyntaxOnly": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedSideEffectImports": true
  },
  "include": ["src"]
}

```

## `tsconfig.json`

```json
{
  "files": [],
  "references": [
    { "path": "./tsconfig.app.json" },
    { "path": "./tsconfig.node.json" }
  ]
}

```

## `tsconfig.node.json`

```json
{
  "compilerOptions": {
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.node.tsbuildinfo",
    "target": "ES2023",
    "lib": ["ES2023"],
    "module": "ESNext",
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "verbatimModuleSyntax": true,
    "moduleDetection": "force",
    "noEmit": true,

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "erasableSyntaxOnly": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedSideEffectImports": true
  },
  "include": ["vite.config.ts"]
}

```

## `vite.config.ts`

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // 监听所有网络接口，方便手机等设备访问
    host: '0.0.0.0', 
    port: 5173, // 你可以指定一个端口
    proxy: {
      // 代理规则：所有/api和/ws的请求，都转发到后端服务器
      '/api': {
        target: 'http://localhost:28501', // 这是你的Python后端地址
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:28501', // WebSocket也需要代理
        ws: true,
      },
    },
  },
})
```

