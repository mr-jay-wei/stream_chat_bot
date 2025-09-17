// mobile/context/AuthContext.js
import React, { createContext, useState, useEffect } from 'react';
import * as SecureStore from 'expo-secure-store';
import apiClient from '../api'; // 导入我们配置好的axios实例

// 1. 创建上下文对象
export const AuthContext = createContext();

// 2. 创建一个 Provider 组件
export const AuthProvider = ({ children }) => {
  const [userToken, setUserToken] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [userInfo, setUserInfo] = useState(null);

  // 登录函数 【直接在这里实现】
  const login = async (email, password) => {
    try {
      // 使用原始的 apiClient 发送请求
      const response = await apiClient.post('/login', { email, password });
      const token = response.data.access_token;
      
      setUserToken(token);
      setUserInfo({ email: response.data.user_email });
      await SecureStore.setItemAsync('userToken', token);
      apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      return response.data;
    } catch (e) {
      console.error('Login error in AuthContext', e);
      throw e;
    }
  };

  // 注册函数 【直接在这里实现】
  const register = async (email, password) => {
    try {
      const response = await apiClient.post('/register', { email, password });
      // 注册成功不自动登录，让用户去登录页手动登录，流程更清晰
      return response.data;
    } catch (e) {
      console.error('Register error in AuthContext', e);
      throw e;
    }
  };

  // 登出函数
  const logout = async () => {
    setIsLoading(true);
    setUserToken(null);
    setUserInfo(null);
    await SecureStore.deleteItemAsync('userToken');
    delete apiClient.defaults.headers.common['Authorization'];
    setIsLoading(false);
  };

  // 检查用户是否已登录的函数
  const isLoggedIn = async () => {
    try {
      setIsLoading(true);
      const token = await SecureStore.getItemAsync('userToken');
      if (token) {
        setUserToken(token);
        apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        const meResponse = await apiClient.get('/me');
        setUserInfo(meResponse.data);
      }
    } catch (e) {
      // Token可能过期或无效，确保登出
      console.log("Token check failed, logging out.", e);
      await logout(); // 调用logout来清理状态
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    isLoggedIn();
  }, []);
  
  const authContextValue = {
    login,
    logout,
    register,
    userToken,
    userInfo,
    isLoading,
  };

  return (
    <AuthContext.Provider value={authContextValue}>
      {children}
    </AuthContext.Provider>
  );
};