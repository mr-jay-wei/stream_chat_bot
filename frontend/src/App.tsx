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