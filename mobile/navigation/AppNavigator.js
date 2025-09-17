// mobile/navigation/AppNavigator.js

import React, { useContext } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { AuthContext } from '../context/AuthContext'; // 👈 导入 AuthContext
import { View, ActivityIndicator } from 'react-native';

import LoginScreen from '../screens/Auth/LoginScreen';
import RegisterScreen from '../screens/Auth/RegisterScreen';
import ChatScreen from '../screens/Main/ChatScreen';
import LoadingScreen from '../screens/LoadingScreen'; // 我们将创建一个加载屏

const Stack = createNativeStackNavigator();

export default function AppNavigator() {
  // 从 AuthContext 中获取状态
  const { userToken, isLoading } = useContext(AuthContext);

  // 如果正在加载（例如，正在检查本地存储的token），显示一个加载动画
  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{
        headerStyle: { backgroundColor: '#667eea' },
        headerTintColor: '#fff',
        headerTitleStyle: { fontWeight: 'bold' },
      }}>
        {/* 
          这里是核心逻辑：
          如果 userToken 存在 (用户已登录), 则显示应用主屏幕 (ChatScreen)
          如果 userToken 不存在 (用户未登录), 则显示认证相关的屏幕 (Login, Register)
        */}
        {userToken ? (
          <Stack.Screen name="Chat" component={ChatScreen} options={{ title: '蓬竹猫' }} />
        ) : (
          <>
            <Stack.Screen name="Login" component={LoginScreen} options={{ title: '登录' }} />
            <Stack.Screen name="Register" component={RegisterScreen} options={{ title: '注册' }} />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}