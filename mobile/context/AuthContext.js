// mobile/context/AuthContext.js
import React, { createContext, useState, useEffect } from 'react';
import * as SecureStore from 'expo-secure-store';
import apiClient from '../api/index';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [userToken, setUserToken] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [userInfo, setUserInfo] = useState(null);

  // 登出函数 (保持不变)
  const logout = async () => {
    setUserToken(null);
    setUserInfo(null);
    delete apiClient.defaults.headers.common['Authorization'];
    await SecureStore.deleteItemAsync('userToken');
  };

  // 登录函数 (保持不变)
  const login = async (email, password) => {
    try {
      const response = await apiClient.post('/login', { email, password });
      const token = response.data.access_token;

      // 设置请求头是第一优先级的操作
      apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;

      // 然后再更新状态和持久化存储
      setUserToken(token);
      setUserInfo({ email: response.data.user_email });
      await SecureStore.setItemAsync('userToken', token);

      return response.data;
    } catch (e) {
      console.error('Login error in AuthContext', e);
      // 登录失败时，确保清理所有可能存在的旧状态
      await logout();
      throw e;
    }
  };

  // 注册函数 (保持不变)
  const register = async (email, password) => {
    try {
      const response = await apiClient.post('/register', { email, password });
      return response.data;
    } catch (e) {
      console.error('Register error in AuthContext', e);
      throw e;
    }
  };

  // 【核心重构】: isLogged In 函数
  useEffect(() => {
    const bootstrapAsync = async () => {
      let token;
      try {
        token = await SecureStore.getItemAsync('userToken');
        if (token) {
          // 先设置请求头，再去验证
          apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
          // 验证 token 有效性
          const meResponse = await apiClient.get('/me');
          setUserInfo(meResponse.data);
          setUserToken(token);
        }
      } catch (e) {
        // 如果token无效或任何步骤出错，都静默地清除
        console.log('Bootstrap failed, token invalid.', e);
        await SecureStore.deleteItemAsync('userToken');
        delete apiClient.defaults.headers.common['Authorization'];
      } finally {
        // 无论成功与否，最后都结束加载状态
        setIsLoading(false);
      }
    };
    bootstrapAsync();
  }, []);

  const authContextValue = {
    login,
    logout,
    register,
    userToken,
    userInfo,
    isLoading,
  };

  return <AuthContext.Provider value={authContextValue}>{children}</AuthContext.Provider>;
};
