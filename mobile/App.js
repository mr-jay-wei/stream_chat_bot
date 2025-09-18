// mobile/App.js
import 'react-native-gesture-handler'; // 👈 【第1步】: 必须在顶部第一行导入
import { registerRootComponent } from 'expo';
import React from 'react';
import { GestureHandlerRootView } from 'react-native-gesture-handler'; // 👈 【第2步】: 导入“司令部”
import AppNavigator from './navigation/AppNavigator';
import { AuthProvider } from './context/AuthContext';

function App() {
  return (
    // 👇 【第3步】: 用“司令部”包裹所有内容
    <GestureHandlerRootView style={{ flex: 1 }}>
      <AuthProvider>
        <AppNavigator />
      </AuthProvider>
    </GestureHandlerRootView>
  );
}
registerRootComponent(App);
