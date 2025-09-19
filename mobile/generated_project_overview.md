# 项目概览: mobile

本文档由`generate_project_overview.py`自动生成，包含了项目的结构树和所有可读文件的内容。

## 项目结构

```
mobile/
├── .cursor
│   └── mcp.json
├── .expo
│   ├── types
│   │   └── router.d.ts
│   ├── web
│   │   └── cache
│   │       └── production
│   │           └── images
│   │               └── favicon
│   │                   └── favicon-a4e030697a7571b3e95d31860e4da55d2f98e5e861e2b55e414f45a8556828ba-contain-transparent
│   ├── devices.json
│   └── README.md
├── api
│   ├── chat.js
│   ├── index.js
│   └── prompt.js
├── assets
│   └── images
├── components
│   ├── MessageItem.js
│   └── SwipeableRow.js
├── context
│   └── AuthContext.js
├── navigation
│   ├── AppNavigator.js
│   └── MainTabNavigator.js
├── screens
│   ├── Auth
│   │   ├── LoginScreen.js
│   │   └── RegisterScreen.js
│   ├── Main
│   │   ├── ChatScreen.js
│   │   ├── PromptEditScreen.js
│   │   ├── PromptListScreen.js
│   │   └── SessionListScreen.js
│   └── LoadingScreen.js
├── services
│   └── WebSocketClient.js
├── .env_example
├── .gitignore
├── .prettierrc.js
├── App.js
├── app.json
├── eslint.config.js
├── expo-env.d.ts
├── package.json
├── README.md
└── tsconfig.json
```

---

# 文件内容

## `.cursor/mcp.json`

```json
{
  "mcpServers": {
    "RadonAi": {
      "url": "http://127.0.0.1:63266/mcp",
      "type": "http",
      "headers": {
        "nonce": "7a6ba1e0-1f15-4d68-90df-3a71371f2694"
      }
    }
  }
}

```

## `.env_example`

```
# 这是前端应用的环境变量配置文件。
# 复制这个文件为 .env (cp .env_example .env)，然后填入你自己的值。
# 注意：所有暴露给客户端的环境变量，都必须以 EXPO_PUBLIC_ 开头。

# 你的后端API服务的访问地址。
# 在本地开发时，这通常是你电脑的局-域-网IP地址。
# 部署到生产环境时，这应该是一个公共的URL。
EXPO_PUBLIC_API_URL=http://YOUR_COMPUTER_LAN_IP:28501/api
```

## `.expo/devices.json`

```json
{
  "devices": []
}

```

## `.expo/README.md`

````text
> Why do I have a folder named ".expo" in my project?

The ".expo" folder is created when an Expo project is started using "expo start" command.

> What do the files contain?

- "devices.json": contains information about devices that have recently opened this project. This is used to populate the "Development sessions" list in your development builds.
- "settings.json": contains the server configuration that is used to serve the application manifest.

> Should I commit the ".expo" folder?

No, you should not share the ".expo" folder. It does not contain any information that is relevant for other developers working on the project, it is specific to your machine.
Upon project creation, the ".expo" folder is already added to your ".gitignore" file.

````

## `.expo/types/router.d.ts`

```typescript
/* eslint-disable */
import * as Router from 'expo-router';

export * from 'expo-router';

declare module 'expo-router' {
  export namespace ExpoRouter {
    export interface __routes<T extends string | object = string> {
      hrefInputParams: { pathname: Router.RelativePathString, params?: Router.UnknownInputParams } | { pathname: Router.ExternalPathString, params?: Router.UnknownInputParams };
      hrefOutputParams: { pathname: Router.RelativePathString, params?: Router.UnknownOutputParams } | { pathname: Router.ExternalPathString, params?: Router.UnknownOutputParams };
      href: Router.RelativePathString | Router.ExternalPathString | { pathname: Router.RelativePathString, params?: Router.UnknownInputParams } | { pathname: Router.ExternalPathString, params?: Router.UnknownInputParams };
    }
  }
}

```

## `.gitignore`

```
# Learn more https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files

# dependencies
node_modules/

# Expo
.expo/
dist/
web-build/
expo-env.d.ts

# Native
.kotlin/
*.orig.*
*.jks
*.p8
*.p12
*.key
*.mobileprovision

# Metro
.metro-health-check*

# debug
npm-debug.*
yarn-debug.*
yarn-error.*

# macOS
.DS_Store
*.pem

# local env files
.env*.local
.env

# typescript
*.tsbuildinfo

app-example

# generated native folders
/ios
/android

```

## `.prettierrc.js`

```javascript
module.exports = {
  arrowParens: 'avoid',
  bracketSameLine: true,
  singleQuote: true,
  trailingComma: 'all',
  printWidth: 100,
};

```

## `api/chat.js`

```javascript
// mobile/api/chat.js
import apiClient from './index';

/**
 * 获取指定会话的历史消息
 * @param {number} sessionId 会话ID
 * @returns {Promise<Array>} 消息列表
 */
export const getSessionMessages = async sessionId => {
  if (!sessionId) return [];
  try {
    const response = await apiClient.get(`/chat-sessions/${sessionId}/messages`);
    return response.data.messages || [];
  } catch (error) {
    console.error('Failed to fetch session messages:', error);
    throw error;
  }
};

/**
 * 获取当前用户的所有会话列表
 * @returns {Promise<Array>} 会话列表
 */
export const getUserSessions = async () => {
  try {
    const response = await apiClient.get(`/conversations`);
    return response.data.conversations || [];
  } catch (error) {
    console.error('Failed to fetch user sessions:', error);
    throw error;
  }
};

/**
 * 删除指定的会话
 * @param {number} sessionId 要删除的会话ID
 * @returns {Promise<object>} 后端返回的成功信息
 */
export const deleteSession = async sessionId => {
  try {
    const response = await apiClient.delete(`/chat-sessions/${sessionId}`);
    return response.data;
  } catch (error) {
    console.error(`Failed to delete session ${sessionId}:`, error);
    throw error;
  }
};

/**
 * 删除指定的消息
 * @param {number} messageId 要删除的消息ID
 * @returns {Promise<object>} 后端返回的成功信息
 */
export const deleteMessage = async messageId => {
  try {
    const response = await apiClient.delete(`/messages/${messageId}`);
    return response.data;
  } catch (error) {
    console.error(`Failed to delete message ${messageId}:`, error);
    throw error;
  }
};

```

## `api/index.js`

```javascript
// mobile/api/index.js
import axios from 'axios';

// 你的电脑在局域网中的IP地址。这是让手机能访问到电脑上运行的后端服务的关键。
// 你需要将 'YOUR_COMPUTER_IP' 替换成你自己的IP地址。
const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL;

if (!API_BASE_URL) {
  alert('错误：API URL 未在环境变量中配置！请检查 mobile/.env 文件。');
}

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;

```

## `api/prompt.js`

```javascript
// mobile/api/prompt.js
import apiClient from './index';

export const getPrompts = () => apiClient.get('/prompts');
export const createPrompt = (data) => apiClient.post('/prompts', data);
export const updatePrompt = (id, data) => apiClient.put(`/prompts/${id}`, data);
export const deletePrompt = (id) => apiClient.delete(`/prompts/${id}`);
```

## `App.js`

```javascript
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

```

## `app.json`

```json
{
  "expo": {
    "name": "mobile",
    "slug": "mobile",
    "version": "1.0.0",
    "orientation": "portrait",
    "icon": "./assets/images/icon.png",
    "scheme": "mobile",
    "userInterfaceStyle": "automatic",
    "newArchEnabled": true,
    "ios": {
      "supportsTablet": true
    },
    "android": {
      "adaptiveIcon": {
        "backgroundColor": "#E6F4FE",
        "foregroundImage": "./assets/images/android-icon-foreground.png",
        "backgroundImage": "./assets/images/android-icon-background.png",
        "monochromeImage": "./assets/images/android-icon-monochrome.png"
      },
      "edgeToEdgeEnabled": true,
      "predictiveBackGestureEnabled": false
    },
    "web": {
      "output": "static",
      "favicon": "./assets/images/favicon.png"
    },
    "plugins": [
      [
        "expo-splash-screen",
        {
          "image": "./assets/images/splash-icon.png",
          "imageWidth": 200,
          "resizeMode": "contain",
          "backgroundColor": "#ffffff",
          "dark": {
            "backgroundColor": "#000000"
          }
        }
      ],
      "expo-secure-store"
    ],
    "experiments": {
      "typedRoutes": true,
      "reactCompiler": true
    }
  }
}

```

## `components/MessageItem.js`

```javascript
// mobile/components/MessageItem.js
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';

const MessageItem = ({ item, onLongPress }) => {
  const isUser = item.role === 'user';

  return (
    <TouchableOpacity
      onLongPress={() => onLongPress(item.id)}
      style={[
        styles.messageContainer,
        isUser ? styles.userMessageContainer : styles.botMessageContainer,
      ]}>
      {item.role === 'assistant' && item.content === '' ? (
        <Text style={styles.typingIndicator}>...</Text>
      ) : (
        <Text style={[styles.messageText, isUser ? { color: 'white' } : { color: 'black' }]}>
          {item.content}
        </Text>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  messageContainer: { padding: 12, borderRadius: 18, marginVertical: 5, maxWidth: '80%' },
  userMessageContainer: { backgroundColor: '#667eea', alignSelf: 'flex-end' },
  botMessageContainer: {
    backgroundColor: 'white',
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: '#e0e0e0',
    minHeight: 40,
    justifyContent: 'center',
  },
  messageText: { fontSize: 16 },
  typingIndicator: { fontSize: 18, color: '#999', paddingHorizontal: 5 },
});

export default React.memo(MessageItem);

```

## `components/SwipeableRow.js`

```javascript
// mobile/components/SwipeableRow.js
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Animated } from 'react-native';
import { Swipeable } from 'react-native-gesture-handler';

const SwipeableRow = ({ item, onDelete, onNavigate }) => {
  // 定义右侧滑动出现的内容
  const renderRightActions = (progress, dragX) => {
    const trans = dragX.interpolate({
      inputRange: [-80, 0],
      outputRange: [0, 80],
      extrapolate: 'clamp',
    });
    return (
      <TouchableOpacity onPress={onDelete} style={styles.deleteButton}>
        <Animated.Text style={[styles.deleteButtonText, { transform: [{ translateX: trans }] }]}>
          删除
        </Animated.Text>
      </TouchableOpacity>
    );
  };

  return (
    <Swipeable renderRightActions={renderRightActions}>
      <TouchableOpacity style={styles.sessionItem} onPress={onNavigate}>
        <Text style={styles.sessionTitle} numberOfLines={1}>
          {item.title}
        </Text>
        <Text style={styles.sessionPreview} numberOfLines={1}>
          {item.preview}
        </Text>
      </TouchableOpacity>
    </Swipeable>
  );
};

const styles = StyleSheet.create({
  deleteButton: {
    backgroundColor: 'red',
    justifyContent: 'center',
    alignItems: 'flex-end',
    width: 80,
  },
  deleteButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
    padding: 20,
  },
  sessionItem: {
    backgroundColor: 'white',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  sessionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  sessionPreview: {
    fontSize: 14,
    color: '#666',
  },
});

export default SwipeableRow;

```

## `context/AuthContext.js`

```javascript
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

```

## `eslint.config.js`

```javascript
// https://docs.expo.dev/guides/using-eslint/
const { defineConfig } = require('eslint/config');
const expoConfig = require('eslint-config-expo/flat');

module.exports = defineConfig([
  expoConfig,
  {
    ignores: ['dist/*'],
  },
]);

```

## `expo-env.d.ts`

```typescript
/// <reference types="expo/types" />

// NOTE: This file should not be edited and should be in your git ignore
```

## `navigation/AppNavigator.js`

```javascript
// mobile/navigation/AppNavigator.js
import React, { useContext } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { AuthContext } from '../context/AuthContext';
import LoginScreen from '../screens/Auth/LoginScreen';
import RegisterScreen from '../screens/Auth/RegisterScreen';
import LoadingScreen from '../screens/LoadingScreen';
import MainTabNavigator from './MainTabNavigator'; 

const Stack = createNativeStackNavigator();

export default function AppNavigator() {
  const { userToken, isLoading } = useContext(AuthContext);

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <NavigationContainer>
      <Stack.Navigator>
        {userToken ? (
          // 登录后，加载整个Tab导航器，并隐藏它自己的头部
          <Stack.Screen 
            name="Main" 
            component={MainTabNavigator} 
            options={{ headerShown: false }}
          />
        ) : (
          // 未登录时，显示认证页面
          <>
            <Stack.Screen name="Login" component={LoginScreen} options={{ headerShown: false }} />
            <Stack.Screen name="Register" component={RegisterScreen} options={{ title: '注册' }} />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

## `navigation/MainTabNavigator.js`

```javascript
// mobile/navigation/MainTabNavigator.js
import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import SessionListScreen from '../screens/Main/SessionListScreen';
import ChatScreen from '../screens/Main/ChatScreen';
import PromptListScreen from '../screens/Main/PromptListScreen';
import PromptEditScreen from '../screens/Main/PromptEditScreen';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

function ChatStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="SessionList" component={SessionListScreen} />
      <Stack.Screen name="Chat" component={ChatScreen} />
    </Stack.Navigator>
  );
}

function PromptStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="PromptList" component={PromptListScreen} />
      <Stack.Screen name="PromptEdit" component={PromptEditScreen} />
    </Stack.Navigator>
  );
}

export default function MainTabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarActiveTintColor: '#667eea',
        tabBarInactiveTintColor: 'gray',
        headerShown: false,
      }}
    >
      <Tab.Screen 
        name="ChatStack" 
        component={ChatStack} 
        options={{ title: '对话' }}
      />
      <Tab.Screen 
        name="PromptStack" 
        component={PromptStack} 
        options={{ title: '我的角色' }}
      />
    </Tab.Navigator>
  );
}
```

## `package.json`

```json
{
  "name": "mobile",
  "main": "App.js",
  "version": "1.0.0",
  "scripts": {
    "start": "expo start",
    "reset-project": "node ./scripts/reset-project.js",
    "android": "expo start --android",
    "ios": "expo start --ios",
    "web": "expo start --web",
    "lint": "expo lint",
    "format": "prettier --write \"**/*.{js,jsx,ts,tsx,json}\""
  },
  "dependencies": {
    "@expo/vector-icons": "^15.0.2",
    "@react-navigation/bottom-tabs": "^7.4.7",
    "@react-navigation/elements": "^2.6.3",
    "@react-navigation/native": "^7.1.17",
    "@react-navigation/native-stack": "^7.3.26",
    "axios": "^1.12.2",
    "expo": "~54.0.8",
    "expo-constants": "~18.0.9",
    "expo-font": "~14.0.8",
    "expo-haptics": "~15.0.7",
    "expo-image": "~3.0.8",
    "expo-linking": "~8.0.8",
    "expo-router": "~6.0.6",
    "expo-secure-store": "~15.0.7",
    "expo-splash-screen": "~31.0.10",
    "expo-status-bar": "~3.0.8",
    "expo-symbols": "~1.0.7",
    "expo-system-ui": "~6.0.7",
    "expo-web-browser": "~15.0.7",
    "react": "19.1.0",
    "react-dom": "19.1.0",
    "react-native": "0.81.4",
    "react-native-gesture-handler": "~2.28.0",
    "react-native-reanimated": "~4.1.0",
    "react-native-safe-area-context": "~5.6.0",
    "react-native-screens": "~4.16.0",
    "react-native-url-polyfill": "^2.0.0",
    "react-native-web": "~0.21.0",
    "react-native-worklets": "0.5.1",
    "ws": "^8.18.3"
  },
  "devDependencies": {
    "@types/react": "~19.1.0",
    "eslint": "^9.25.0",
    "eslint-config-expo": "~10.0.0",
    "prettier": "^3.6.2",
    "typescript": "~5.9.2"
  },
  "private": true
}

```

## `README.md`

````text
\# 哈基米 - 移动端 App

欢迎使用蓬竹猫移动App！这是一个使用 **React Native (Expo)** 构建的、功能完备的跨平台AI聊天应用。

#\# ✨ 核心功能

- **完整的用户认证**: 支持注册、登录、登出和持久化登录状态。
- **多会话管理**: 在一个清晰的列表中查看、创建和删除你的所有对话。
- **实时流式聊天**: 与AI进行实时的、带有“打字机”效果的流式对话。
- **上下文记忆**: AI能够记住你在当前对话中提到的信息。
- **专业的用户体验**: 支持下拉刷新、滑动删除、长按删除等现代App交互。

#\# 🚀 启动指南

在开始之前，请确保你已经成功启动了[后端服务](../README.md)。

##\# 1. 安装依赖

在 `mobile/` 目录下，运行以下命令安装所有必需的JavaScript依赖包。

\`\`\`bash
npm install
\`\`\`
##\# 2. 配置环境变量
本项目需要一个.env文件来配置后端API的地址。
\`\`\`code
Bash

\# 在 mobile/ 目录下，复制模板文件
cp .env_example .env
\`\`\`
然后，打开新建的 .env 文件，修改 EXPO_PUBLIC_API_URL 的值为你电脑的局域网IP地址。例如：
\`\`\`code
Code
EXPO_PUBLIC_API_URL=http://192.168.1.10:28501/api
\`\`\`
提示: 你可以在Windows上通过运行 ipconfig 命令来查找你的IPv4地址。
##\# 3. 启动开发服务器
一切准备就绪后，运行以下命令来启动Expo开发服务器：
\`\`\`code
Bash
npx expo start
\`\`\`
##\# 4. 在手机上运行
确保你的手机和电脑连接在同一个Wi-Fi下。
在手机上从应用商店安装 Expo Go 应用。
打开Expo Go，扫描终端上显示的二维码。
App将会在你的手机上启动，现在你可以开始使用了！
##\# 🛠️ 常用脚本
- npm start: 启动开发服务器。
- npm run format: 使用Prettier格式化所有代码。
- npm run lint: 检查代码风格问题。
````

## `screens/Auth/LoginScreen.js`

```javascript
// mobile/screens/Auth/LoginScreen.js

import React, { useState, useContext } from 'react'; // 👈 导入 useContext
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { AuthContext } from '../../context/AuthContext'; // 👈 导入 AuthContext

export default function LoginScreen({ navigation }) {
  const { login } = useContext(AuthContext); // 👈 从 Context 获取 login 函数
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('错误', '请输入邮箱和密码');
      return;
    }
    setLoading(true);
    try {
      await login(email, password); // 👈 直接调用 context 的 login
      // 登录成功后，AppNavigator会自动因为userToken状态变化而重新渲染，无需手动跳转
    } catch (error) {
      Alert.alert('登录失败', error.response?.data?.detail || '邮箱或密码错误，请重试');
    } finally {
      setLoading(false);
    }
  };

  const navigateToRegister = () => {
    navigation.navigate('Register');
  };

  // 3. 修改 JSX，添加加载指示器
  return (
    <View style={styles.container}>
      <Text style={styles.title}>欢迎回来！</Text>

      {/* ... TextInput部分保持不变 ... */}
      <TextInput
        style={styles.input}
        placeholder="请输入邮箱"
        value={email}
        onChangeText={setEmail}
        keyboardType="email-address"
        autoCapitalize="none"
      />
      <TextInput
        style={styles.input}
        placeholder="请输入密码"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
      />

      <TouchableOpacity style={styles.button} onPress={handleLogin} disabled={loading}>
        {loading ? (
          <ActivityIndicator color="white" /> // 👈 如果正在加载，显示一个旋转的菊花
        ) : (
          <Text style={styles.buttonText}>登录</Text> // 👈 否则显示文字
        )}
      </TouchableOpacity>

      <TouchableOpacity onPress={navigateToRegister} disabled={loading}>
        <Text style={styles.linkText}>还没有账户？去注册</Text>
      </TouchableOpacity>
    </View>
  );
}

// 定义组件的样式
const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 40,
  },
  input: {
    width: '100%',
    height: 50,
    backgroundColor: 'white',
    borderRadius: 8,
    paddingHorizontal: 15,
    fontSize: 16,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  button: {
    width: '100%',
    height: 50,
    backgroundColor: '#667eea',
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  linkText: {
    color: '#667eea',
    fontSize: 16,
    marginTop: 20,
  },
});

```

## `screens/Auth/RegisterScreen.js`

```javascript
// mobile/screens/Auth/RegisterScreen.js
import React, { useState, useContext } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { AuthContext } from '../../context/AuthContext';

export default function RegisterScreen({ navigation }) {
  const { register } = useContext(AuthContext);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleRegister = async () => {
    if (!email || !password || !confirmPassword) {
      Alert.alert('错误', '请填写所有字段');
      return;
    }
    if (password !== confirmPassword) {
      Alert.alert('错误', '两次输入的密码不一致');
      return;
    }

    setLoading(true);
    try {
      // 调用Context中的register函数，现在它只负责注册
      await register(email, password);

      // 注册成功后，弹窗提示用户，并提供按钮返回登录页
      Alert.alert(
        '注册成功',
        '您的账户已创建，请返回登录页面进行登录。',
        [{ text: '好的', onPress: () => navigation.goBack() }], // 点击按钮后执行 navigation.goBack()
      );
    } catch (error) {
      // 从后端获取更具体的错误信息并显示
      Alert.alert('注册失败', error.response?.data?.detail || '该邮箱可能已被注册，请重试');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>创建新账户</Text>

      <TextInput
        style={styles.input}
        placeholder="请输入邮箱"
        value={email}
        onChangeText={setEmail}
        keyboardType="email-address"
        autoCapitalize="none"
        editable={!loading}
      />

      <TextInput
        style={styles.input}
        placeholder="请输入密码"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        editable={!loading}
      />

      <TextInput
        style={styles.input}
        placeholder="请确认密码"
        value={confirmPassword}
        onChangeText={setConfirmPassword}
        secureTextEntry
        editable={!loading}
      />

      <TouchableOpacity style={styles.button} onPress={handleRegister} disabled={loading}>
        {loading ? (
          <ActivityIndicator color="white" />
        ) : (
          <Text style={styles.buttonText}>注册</Text>
        )}
      </TouchableOpacity>
    </View>
  );
}

// 样式部分保持不变
const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 40,
  },
  input: {
    width: '100%',
    height: 50,
    backgroundColor: 'white',
    borderRadius: 8,
    paddingHorizontal: 15,
    fontSize: 16,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  button: {
    width: '100%',
    height: 50,
    backgroundColor: '#667eea',
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
});

```

## `screens/LoadingScreen.js`

```javascript
import React from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';

export default function LoadingScreen() {
  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#667eea" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

```

## `screens/Main/ChatScreen.js`

```javascript
// mobile/screens/Main/ChatScreen.js
import React, { useLayoutEffect, useContext, useState, useEffect, useRef, useCallback } from 'react';
import { 
  View, Text, StyleSheet, TextInput, TouchableOpacity, 
  FlatList, KeyboardAvoidingView, Platform, 
  TouchableWithoutFeedback, Keyboard, ActivityIndicator, Alert
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { AuthContext } from '../../context/AuthContext';
import { webSocketClient } from '../../services/WebSocketClient';
import { getSessionMessages, deleteMessage } from '../../api/chat';
import MessageItem from '../../components/MessageItem';

export default function ChatScreen({ navigation, route }) {
  const { logout, userToken } = useContext(AuthContext);
  const { sessionId, promptId } = route.params;

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [currentSessionId, setCurrentSessionId] = useState(sessionId);
  const [isLoadingHistory, setIsLoadingHistory] = useState(!!sessionId);
  
  const currentBotMessageId = useRef(null);
  const flatListRef = useRef(null);

  useEffect(() => {
    if (userToken && webSocketClient) {
      if (!webSocketClient.ws || webSocketClient.ws.readyState === WebSocket.CLOSED) {
        webSocketClient.connect(userToken);
      }
    }
    return () => {
      if (webSocketClient && webSocketClient.ws) {
        webSocketClient.close();
      }
    };
  }, [userToken]);

  useEffect(() => {
    const loadHistory = async (sid) => {
      if (!sid) return;
      setIsLoadingHistory(true);
      try {
        const historyMessages = await getSessionMessages(sid);
        const formattedMessages = historyMessages.map(msg => ({
          id: msg.id.toString(),
          role: msg.role,
          content: msg.content,
        }));
        setMessages(formattedMessages);
      } catch (error) {
        Alert.alert("加载失败", "无法加载历史消息。");
      } finally {
        setIsLoadingHistory(false);
      }
    };

    if (sessionId) {
      loadHistory(sessionId);
    }
    
    const handleWebSocketMessage = (message) => {
      if (message.type === 'processing' && message.data?.session_id && currentSessionId === null) {
        setCurrentSessionId(message.data.session_id);
      }
    
      setMessages(prevMessages => {
        let newMessages = [...prevMessages];
    
        switch (message.type) {
          case 'generation_start': {
            const botPlaceholder = { id: `bot-${Date.now()}`, role: 'assistant', content: '' };
            currentBotMessageId.current = botPlaceholder.id;
            newMessages.push(botPlaceholder);
            break;
          }
          case 'generation_chunk': {
            const chunk = message.data?.chunk || '';
            newMessages = newMessages.map(m => m.id === currentBotMessageId.current ? { ...m, content: m.content + chunk } : m);
            break;
          }
          case 'complete': {
            // 把占位符的 id 更新为后端返回的实际 id（如果有）
            const aiId = message.data?.ai_message_id ? String(message.data.ai_message_id) : null;
            newMessages = newMessages.map(m => m.id === currentBotMessageId.current ? (aiId ? { ...m, id: aiId } : m) : m);
            currentBotMessageId.current = null;
            break;
          }
          case 'error': {
            Alert.alert('AI 错误', message.data?.error || '发生未知错误');
            newMessages = newMessages.filter(m => m.id !== currentBotMessageId.current);
            currentBotMessageId.current = null;
            break;
          }
          default:
            break;
        }
    
        return newMessages;
      });
    };
    
    if(webSocketClient) webSocketClient.on('message', handleWebSocketMessage);
    
    return () => {
      if(webSocketClient) webSocketClient.removeListener('message', handleWebSocketMessage);
    };
  }, [sessionId, currentSessionId]);

  useLayoutEffect(() => {
    navigation.setOptions({
        title: sessionId ? '继续对话' : '新对话'
    });
  }, [navigation, sessionId]);

  useEffect(() => {
    if (flatListRef.current && messages.length > 0) {
      flatListRef.current.scrollToEnd({ animated: true });
    }
  }, [messages]);

  const handleSend = () => {
    if (input.trim().length === 0) return;
    const userMessage = { id: `user-${Date.now()}`, role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    
    const payload = {
      type: 'question',
      content: input,
      session_id: currentSessionId,
    };
    if (currentSessionId === null) {
      payload.prompt_id = promptId;
    }
    
    if (webSocketClient && webSocketClient.ws && webSocketClient.ws.readyState === WebSocket.OPEN) {
      webSocketClient.sendMessage(payload);
    } else {
      Alert.alert("连接错误", "无法发送消息，请检查网络连接。");
    }
    setInput('');
    Keyboard.dismiss();
  };
  
  const handleLongPressMessage = (messageId) => {
    if (String(messageId).startsWith('bot-') || String(messageId).startsWith('user-')) return;
    Alert.alert("确认删除", "要删除这条消息吗？", [
      { text: "取消" },
      { text: "删除", onPress: async () => {
          try {
            await deleteMessage(messageId);
            setMessages(prev => prev.filter(m => m.id !== messageId.toString()));
          } catch (error) {
            Alert.alert("删除失败");
          }
        }, style: "destructive"
      }
    ]);
  };

  const renderMessage = useCallback(({ item }) => (
    <MessageItem item={item} onLongPress={handleLongPressMessage} />
  ), []);

  return (
    <SafeAreaView style={styles.container} edges={['bottom', 'left', 'right']}>
      <KeyboardAvoidingView style={{flex: 1}} behavior={Platform.OS === "ios" ? "padding" : "height"} keyboardVerticalOffset={Platform.OS === "ios" ? 90 : 0}>
        <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
          <View style={{ flex: 1 }}>
            {isLoadingHistory ? (
              <View style={styles.centered}><ActivityIndicator size="large" color="#667eea" /></View>
            ) : (
              <FlatList
                ref={flatListRef} data={messages} renderItem={renderMessage}
                keyExtractor={(item) => item.id} style={styles.messageList}
                ListEmptyComponent={<View style={styles.centered}><Text style={styles.emptyText}>开始你的对话吧！</Text></View>}
                initialNumToRender={15} maxToRenderPerBatch={10} windowSize={21}
              />
            )}
          </View>
        </TouchableWithoutFeedback>
        <View style={styles.inputContainer}>
          <TextInput style={styles.input} value={input} onChangeText={setInput} placeholder="请输入您的问题..." />
          <TouchableOpacity style={styles.sendButton} onPress={handleSend}>
            <Text style={styles.sendButtonText}>发送</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  messageList: { flex: 1, paddingHorizontal: 10, paddingTop: 10 },
  emptyText: { fontSize: 18, color: '#aaa' },
  inputContainer: { flexDirection: 'row', padding: 10, borderTopWidth: 1, borderTopColor: '#ddd', backgroundColor: 'white' },
  input: { flex: 1, height: 40, borderWidth: 1, borderColor: '#ddd', borderRadius: 20, paddingHorizontal: 15, backgroundColor: '#f0f0f0' },
  sendButton: { marginLeft: 10, justifyContent: 'center', alignItems: 'center', backgroundColor: '#667eea', borderRadius: 20, paddingHorizontal: 15 },
  sendButtonText: { color: 'white', fontWeight: 'bold' },
});
```

## `screens/Main/PromptEditScreen.js`

```javascript
// mobile/screens/Main/PromptEditScreen.js
import React, { useState, useLayoutEffect } from 'react';
import { ScrollView, TextInput, StyleSheet, Button, Alert, KeyboardAvoidingView, Platform, TouchableOpacity, Text } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { createPrompt, updatePrompt } from '../../api/prompt';


export default function PromptEditScreen({ route, navigation }) {
  const { prompt } = route.params;
  const isEditing = !!prompt;

  const [name, setName] = useState(prompt?.name || '');
  const [content, setContent] = useState(prompt?.content || '');
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    if (!name.trim() || !content.trim()) {
      Alert.alert('提示', '角色名称和设定不能为空');
      return;
    }
    setIsSaving(true);
    try {
      if (isEditing) {
        await updatePrompt(prompt.id, { name, content });
      } else {
        await createPrompt({ name, content });
      }
      navigation.goBack();
    } catch (error) {
      Alert.alert('错误', '保存失败，请稍后重试');
    } finally {
      setIsSaving(false);
    }
  };
  
  useLayoutEffect(() => {
    navigation.setOptions({
      title: isEditing ? '编辑角色' : '新建角色',
      headerStyle: { backgroundColor: '#667eea' },
      headerTintColor: '#fff',
      headerRight: () => (
        <TouchableOpacity
          onPress={handleSave}
          disabled={isSaving}
          style={{ marginRight: 10, opacity: isSaving ? 0.6 : 1 }}
        >
          <Text style={{ color: '#fff', fontSize: 16 }}>
            {isSaving ? '保存中...' : '保存'}
          </Text>
        </TouchableOpacity>
      ),
    });
  }, [navigation, isEditing, name, content, isSaving]);

  return (
    <SafeAreaView style={styles.container} edges={['left', 'right', 'bottom']}>
      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={styles.scrollContainer}>
          <TextInput
            style={styles.input}
            placeholder="角色名称 (例如：语言教练)"
            value={name}
            onChangeText={setName}
          />
          <TextInput
            style={[styles.input, styles.textArea]}
            placeholder="角色设定 (例如：你是一位严格的雅思口语考官...)"
            value={content}
            onChangeText={setContent}
            multiline
          />
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  scrollContainer: { padding: 15 },
  input: { backgroundColor: 'white', borderWidth: 1, borderColor: '#ccc', borderRadius: 8, padding: 15, fontSize: 16, marginBottom: 20 },
  textArea: { height: 300, textAlignVertical: 'top' },
});
```

## `screens/Main/PromptListScreen.js`

```javascript
// mobile/screens/Main/PromptListScreen.js
import React, { useState, useCallback, useLayoutEffect, useContext } from 'react';
import { 
  View, Text, StyleSheet, FlatList, TouchableOpacity, 
  Alert, ActivityIndicator, Button 
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { getPrompts, deletePrompt } from '../../api/prompt';
import { AuthContext } from '../../context/AuthContext';

export default function PromptListScreen({ navigation }) {
  const { userToken } = useContext(AuthContext);
  const [prompts, setPrompts] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchPrompts = useCallback(async () => {
    if (!userToken) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await getPrompts();
      setPrompts(response.data);
    } catch (err) {
      setError('无法加载角色列表');
    } finally {
      setIsLoading(false);
    }
  }, [userToken]);

  useFocusEffect(useCallback(() => { fetchPrompts(); }, [fetchPrompts]));
  
  useLayoutEffect(() => {
    navigation.setOptions({
      title: '我的角色',
      headerStyle: { backgroundColor: '#667eea' },
      headerTintColor: '#fff',
      headerRight: () => (
        <TouchableOpacity style={styles.newButton} onPress={() => navigation.navigate('PromptEdit', { prompt: null })}>
          <Text style={styles.newButtonText}>+ 新建</Text>
        </TouchableOpacity>
      ),
    });
  }, [navigation]);

  const handleDelete = (id) => {
    Alert.alert("确认删除", "确定要删除这个角色吗？", [
      { text: "取消" },
      { text: "删除", onPress: async () => {
          try {
            await deletePrompt(id);
            setPrompts(prev => prev.filter(p => p.id !== id));
          } catch (error) {
            Alert.alert('错误', '删除失败');
          }
        }, style: "destructive" 
      }
    ]);
  };

  const renderContent = () => {
    if (isLoading && prompts.length === 0) {
      return <View style={styles.centered}><ActivityIndicator size="large" color="#667eea" /></View>;
    }
    if (error) {
      return (
        <View style={styles.centered}>
          <Text style={styles.errorText}>{error}</Text>
          <Button title="点我重试" onPress={fetchPrompts} color="#667eea" />
        </View>
      );
    }
    return (
      <FlatList
        data={prompts}
        keyExtractor={(item) => item.id.toString()}
        onRefresh={fetchPrompts}
        refreshing={isLoading}
        renderItem={({ item }) => (
          <View style={styles.promptItem}>
            <TouchableOpacity style={styles.promptContent} onPress={() => navigation.navigate('PromptEdit', { prompt: item })}>
              <Text style={styles.promptName}>{item.name}</Text>
              <Text numberOfLines={2} style={styles.promptPreview}>{item.content}</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => handleDelete(item.id)} style={styles.deleteButton}>
              <Text style={styles.deleteText}>删除</Text>
            </TouchableOpacity>
          </View>
        )}
        ListEmptyComponent={
          <View style={styles.centered}>
            <Text style={styles.emptyText}>还没有自定义角色，{"\n"}点击右上角“+ 新建”来创建一个吧！</Text>
          </View>
        }
      />
    );
  };
  
  return (
    <SafeAreaView style={styles.container} edges={['left', 'right', 'bottom']}>
      {renderContent()}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  promptItem: { flexDirection: 'row', borderBottomWidth: 1, borderColor: '#eee', alignItems: 'center', backgroundColor: 'white' },
  promptContent: { flex: 1, paddingVertical: 15, paddingLeft: 20, paddingRight: 10 },
  promptName: { fontWeight: 'bold', fontSize: 16, marginBottom: 5 },
  promptPreview: { color: '#666' },
  deleteButton: { padding: 20 },
  deleteText: { color: 'red', fontSize: 16 },
  newButton: { marginRight: 10, paddingVertical: 5, paddingHorizontal: 10, borderRadius: 5 },
  newButtonText: { color: 'white', fontSize: 16, fontWeight: '600' },
  emptyText: { textAlign: 'center', lineHeight: 24, fontSize: 16, color: '#999' },
  errorText: { textAlign: 'center', fontSize: 16, color: 'red', marginBottom: 20, lineHeight: 24 },
});
```

## `screens/Main/SessionListScreen.js`

```javascript
// mobile/screens/Main/SessionListScreen.js
import React, { useState, useLayoutEffect, useCallback, useContext } from 'react';
import { 
  View, Text, StyleSheet, FlatList, TouchableOpacity, 
  ActivityIndicator, Alert, ScrollView, Button 
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { AuthContext } from '../../context/AuthContext';
import { getUserSessions, deleteSession } from '../../api/chat';
import { getPrompts } from '../../api/prompt';
import SwipeableRow from '../../components/SwipeableRow';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function SessionListScreen({ navigation }) {
  const { logout, userToken } = useContext(AuthContext);
  const [sessions, setSessions] = useState([]);
  const [prompts, setPrompts] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    if (!userToken) return;
    setIsLoading(true);
    setError(null);
    try {
      const [sessionsData, promptsResponse] = await Promise.all([ getUserSessions(), getPrompts() ]);
      setSessions(sessionsData);
      setPrompts(promptsResponse.data);
    } catch (err) {
      setError("无法加载数据，请检查网络并重试。");
    } finally {
      setIsLoading(false);
    }
  }, [userToken]);

  useFocusEffect(useCallback(() => { fetchData(); }, [fetchData]));

  useLayoutEffect(() => {
    navigation.setOptions({
      title: '我的对话',
      headerStyle: { backgroundColor: '#667eea' },
      headerTintColor: '#fff',
      headerRight: () => (
        <TouchableOpacity onPress={logout} style={{ paddingHorizontal: 10 }}>
          <Text style={{ color: 'white', fontSize: 16 }}>登出</Text>
        </TouchableOpacity>
      )
    });
  }, [navigation, logout]);

  const handleDelete = (sessionId) => {
    Alert.alert("确认删除", "此操作不可撤销。", [
      { text: "取消" },
      { text: "删除", onPress: async () => {
          try {
            await deleteSession(sessionId);
            setSessions(prev => prev.filter(s => s.id !== sessionId));
          } catch (error) {
            Alert.alert('错误', '删除失败');
          }
        }, style: "destructive"
      }
    ]);
  };

  const renderItem = ({ item }) => (
    <SwipeableRow
      item={item}
      onDelete={() => handleDelete(item.id)}
      onNavigate={() => navigation.navigate('Chat', { sessionId: item.id, promptId: null })}
    />
  );

  const renderHeader = () => (
    <View style={styles.headerContainer}>
      <Text style={styles.headerTitle}>选择一个角色开始新对话</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.promptScroll}>
        <TouchableOpacity style={styles.promptChip} onPress={() => navigation.navigate('Chat', { sessionId: null, promptId: null })}>
          <Text style={styles.promptChipText}>默认助手</Text>
        </TouchableOpacity>
        {prompts.map(prompt => (
          <TouchableOpacity key={prompt.id} style={styles.promptChip} onPress={() => navigation.navigate('Chat', { sessionId: null, promptId: prompt.id })}>
            <Text style={styles.promptChipText}>{prompt.name}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );

  const renderContent = () => {
    if (isLoading && sessions.length === 0 && prompts.length === 0) {
      return <View style={styles.centered}><ActivityIndicator size="large" color="#667eea" /></View>;
    }
    if (error) {
      return (
        <View style={styles.centered}>
          <Text style={styles.errorText}>{error}</Text>
          <Button title="点我重试" onPress={fetchData} color="#667eea" />
        </View>
      );
    }
    return (
      <FlatList
        data={sessions}
        renderItem={renderItem}
        keyExtractor={item => item.id.toString()}
        ListHeaderComponent={renderHeader}
        ListEmptyComponent={<View style={styles.centered}><Text style={styles.emptyText}>没有历史对话</Text></View>}
        onRefresh={fetchData}
        refreshing={isLoading}
      />
    );
  };
  
  return (
    <SafeAreaView style={styles.container} edges={['left', 'right', 'bottom']}>
      {renderContent()}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  headerContainer: { padding: 15, borderBottomWidth: 1, borderColor: '#eee', backgroundColor: 'white' },
  headerTitle: { fontSize: 16, fontWeight: 'bold', marginBottom: 15, color: '#333' },
  promptScroll: { paddingBottom: 5 },
  promptChip: { backgroundColor: '#e9e9f7', paddingHorizontal: 15, paddingVertical: 8, borderRadius: 20, marginRight: 10, justifyContent: 'center', alignItems: 'center' },
  promptChipText: { color: '#43419a', fontWeight: '500' },
  emptyText: { textAlign: 'center', fontSize: 16, color: '#999' },
  errorText: { textAlign: 'center', fontSize: 16, color: 'red', marginBottom: 20 },
});
```

## `services/WebSocketClient.js`

```javascript
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

```

## `tsconfig.json`

```json
{
  "extends": "expo/tsconfig.base",
  "compilerOptions": {
    "strict": true,
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["**/*.ts", "**/*.tsx", ".expo/types/**/*.ts", "expo-env.d.ts"]
}

```

