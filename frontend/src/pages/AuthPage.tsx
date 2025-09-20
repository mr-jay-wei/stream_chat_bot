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