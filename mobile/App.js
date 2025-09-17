// mobile/App.js

import { registerRootComponent } from 'expo';
import React from 'react';
import AppNavigator from './navigation/AppNavigator';
import { AuthProvider } from './context/AuthContext'; // 👈 导入 AuthProvider

function App() {
  return (
    // 用 AuthProvider 包裹我们的导航器
    <AuthProvider>
      <AppNavigator />
    </AuthProvider>
  );
}

registerRootComponent(App);