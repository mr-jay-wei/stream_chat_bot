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
│   └── index.js
├── assets
│   └── images
├── components
├── context
│   └── AuthContext.js
├── navigation
│   └── AppNavigator.js
├── screens
│   ├── Auth
│   │   ├── LoginScreen.js
│   │   └── RegisterScreen.js
│   ├── Main
│   │   ├── ChatScreen.js
│   │   └── SessionListScreen.js
│   └── LoadingScreen.js
├── services
│   └── WebSocketClient.js
├── .gitignore
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

# typescript
*.tsbuildinfo

app-example

# generated native folders
/ios
/android

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
export const getSessionMessages = async (sessionId) => {
  if (!sessionId) return [];
  try {
    const response = await apiClient.get(`/chat-sessions/${sessionId}/messages`);
    return response.data.messages || []; // 确保即使没有消息也返回一个空数组
  } catch (error) {
    console.error('Failed to fetch session messages:', error);
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
const YOUR_COMPUTER_IP = '192.168.31.134'; // 例如: '172.20.10.2'

const API_BASE_URL = `http://${YOUR_COMPUTER_IP}:28501/api`;

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;
```

## `App.js`

```javascript
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

## `context/AuthContext.js`

```javascript
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
    "lint": "expo lint"
  },
  "dependencies": {
    "@expo/vector-icons": "^15.0.2",
    "@react-navigation/bottom-tabs": "^7.4.0",
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
    "typescript": "~5.9.2"
  },
  "private": true
}

```

## `README.md`

````text
\# Welcome to your Expo app 👋

This is an [Expo](https://expo.dev) project created with [`create-expo-app`](https://www.npmjs.com/package/create-expo-app).

#\# Get started

1. Install dependencies

   \`\`\`bash
   npm install
   \`\`\`

2. Start the app

   \`\`\`bash
   npx expo start
   \`\`\`

In the output, you'll find options to open the app in a

- [development build](https://docs.expo.dev/develop/development-builds/introduction/)
- [Android emulator](https://docs.expo.dev/workflow/android-studio-emulator/)
- [iOS simulator](https://docs.expo.dev/workflow/ios-simulator/)
- [Expo Go](https://expo.dev/go), a limited sandbox for trying out app development with Expo

You can start developing by editing the files inside the **app** directory. This project uses [file-based routing](https://docs.expo.dev/router/introduction).

#\# Get a fresh project

When you're ready, run:

\`\`\`bash
npm run reset-project
\`\`\`

This command will move the starter code to the **app-example** directory and create a blank **app** directory where you can start developing.

#\# Learn more

To learn more about developing your project with Expo, look at the following resources:

- [Expo documentation](https://docs.expo.dev/): Learn fundamentals, or go into advanced topics with our [guides](https://docs.expo.dev/guides).
- [Learn Expo tutorial](https://docs.expo.dev/tutorial/introduction/): Follow a step-by-step tutorial where you'll create a project that runs on Android, iOS, and the web.

#\# Join the community

Join our community of developers creating universal apps.

- [Expo on GitHub](https://github.com/expo/expo): View our open source platform and contribute.
- [Discord community](https://chat.expo.dev): Chat with Expo users and ask questions.

````

## `screens/Auth/LoginScreen.js`

```javascript
// mobile/screens/Auth/LoginScreen.js

import React, { useState, useContext } from 'react'; // 👈 导入 useContext
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert, ActivityIndicator } from 'react-native';
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
      <TextInput style={styles.input} placeholder="请输入邮箱" value={email} onChangeText={setEmail} keyboardType="email-address" autoCapitalize="none" />
      <TextInput style={styles.input} placeholder="请输入密码" value={password} onChangeText={setPassword} secureTextEntry />

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
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert, ActivityIndicator } from 'react-native';
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
        [{ text: '好的', onPress: () => navigation.goBack() }] // 点击按钮后执行 navigation.goBack()
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
import { View, Text, StyleSheet, TextInput, TouchableOpacity, FlatList, KeyboardAvoidingView, Platform, TouchableWithoutFeedback, Keyboard } from 'react-native';
import { AuthContext } from '../../context/AuthContext';
import { webSocketClient } from '../../services/WebSocketClient';
import { getSessionMessages } from '../../api/chat';
import MessageItem from '../../components/MessageItem';

export default function ChatScreen({ navigation }) {
  const { logout, userToken } = useContext(AuthContext);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [currentSessionId, setCurrentSessionId] = useState(null);
  
  const currentBotMessageId = useRef(null);
  const flatListRef = useRef(null);

  // Effect for WebSocket connection management
  useEffect(() => {
    if (userToken && webSocketClient) {
      if (!webSocketClient.ws || webSocketClient.ws.readyState === WebSocket.CLOSED) {
        webSocketClient.connect(userToken);
      }
    }
    return () => {
      console.log("ChatScreen unmounting, closing WebSocket.");
      if (webSocketClient && webSocketClient.ws) {
        webSocketClient.close();
      }
    };
  }, [userToken]);

  // Effect for handling WebSocket messages
  useEffect(() => {
    const handleWebSocketMessage = (message) => {
      if (message.type === 'processing' && message.data.session_id && currentSessionId === null) {
        setCurrentSessionId(message.data.session_id);
      }

      setMessages(prevMessages => {
        let newMessages = [...prevMessages];
        switch (message.type) {
          case 'generation_start':
            const botPlaceholder = { id: `bot-${Date.now()}`, role: 'assistant', content: '' };
            currentBotMessageId.current = botPlaceholder.id;
            newMessages.push(botPlaceholder);
            break;
          case 'generation_chunk':
            newMessages = newMessages.map(msg =>
              msg.id === currentBotMessageId.current ? { ...msg, content: msg.content + message.data.chunk } : msg
            );
            break;
          case 'complete':
            const finalBotMessageId = message.data.ai_message_id;
            const finalUserMessageId = message.data.user_message_id;
            newMessages = newMessages.map(msg => {
              if (msg.id === currentBotMessageId.current && finalBotMessageId) {
                return { ...msg, id: finalBotMessageId.toString() };
              }
              const lastUserMsg = newMessages.filter(m => m.role === 'user' && String(m.id).startsWith('user-')).pop();
              if (lastUserMsg && msg.id === lastUserMsg.id && finalUserMessageId) {
                 return { ...msg, id: finalUserMessageId.toString() };
              }
              return msg;
            });
            currentBotMessageId.current = null;
            break;
        }
        return newMessages;
      });
    };
    
    if(webSocketClient) {
      webSocketClient.on('message', handleWebSocketMessage);
    }

    return () => {
      console.log("Removing WebSocket message listener.");
      if(webSocketClient) {
        webSocketClient.removeListener('message', handleWebSocketMessage);
      }
    };
  }, [currentSessionId]);

  // Effect for setting navigation options
  useLayoutEffect(() => {
    navigation.setOptions({
      headerRight: () => (
        <TouchableOpacity onPress={logout} style={{ marginRight: 10 }}>
          <Text style={{ color: 'white', fontSize: 16 }}>登出</Text>
        </TouchableOpacity>
      ),
    });
  }, [navigation, logout]);

  // Effect for auto-scrolling
  useEffect(() => {
    if (flatListRef.current && messages.length > 0) {
      flatListRef.current.scrollToEnd({ animated: false });
    }
  }, [messages]);

  const handleSend = () => {
    if (input.trim().length === 0) return;
    const userMessage = { id: `user-${Date.now()}`, role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    
    if (webSocketClient && webSocketClient.ws && webSocketClient.ws.readyState === WebSocket.OPEN) {
      webSocketClient.sendMessage({ type: 'question', content: input, session_id: currentSessionId });
    } else {
      console.warn("WebSocket not ready, message not sent.");
    }
    setInput('');
    Keyboard.dismiss();
  };

  const renderMessage = useCallback(({ item }) => <MessageItem item={item} />, []);

  return (
    <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === "ios" ? "padding" : "height"} keyboardVerticalOffset={90}>
      <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
        <FlatList
          ref={flatListRef}
          data={messages}
          renderItem={renderMessage}
          keyExtractor={(item) => item.id}
          style={styles.messageList}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Text style={styles.emptyText}>开始你的对话吧！</Text>
            </View>
          }
          initialNumToRender={15}
          maxToRenderPerBatch={10}
          windowSize={21}
        />
      </TouchableWithoutFeedback>
      <View style={styles.inputContainer}>
        <TextInput style={styles.input} value={input} onChangeText={setInput} placeholder="请输入您的问题..." />
        <TouchableOpacity style={styles.sendButton} onPress={handleSend}>
          <Text style={styles.sendButtonText}>发送</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  messageList: { flex: 1, paddingHorizontal: 10, paddingTop: 10 },
  emptyContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', paddingTop: '50%' },
  emptyText: { fontSize: 18, color: '#aaa' },
  inputContainer: { flexDirection: 'row', padding: 10, borderTopWidth: 1, borderTopColor: '#ddd', backgroundColor: 'white' },
  input: { flex: 1, height: 40, borderWidth: 1, borderColor: '#ddd', borderRadius: 20, paddingHorizontal: 15, backgroundColor: '#f5f5f5' },
  sendButton: { marginLeft: 10, justifyContent: 'center', alignItems: 'center', backgroundColor: '#667eea', borderRadius: 20, paddingHorizontal: 15 },
  sendButtonText: { color: 'white', fontWeight: 'bold' },
});
```

## `screens/Main/SessionListScreen.js`

[文件为空]

## `services/WebSocketClient.js`

```javascript
// mobile/services/WebSocketClient.js
import 'react-native-url-polyfill/auto';
import apiClient from '../api/index';

const WEBSOCKET_URL = apiClient.defaults.baseURL.replace('http', 'ws').replace('/api', '/ws');

class WebSocketClient {
  // ... 内部代码完全不变 ...
  constructor() { this.ws = null; this.listeners = {}; }
  on(event, callback) { if (!this.listeners[event]) { this.listeners[event] = []; } this.listeners[event].push(callback); }
  removeListener(event, callback) { if (this.listeners[event]) { this.listeners[event] = this.listeners[event].filter(l => l !== callback); } }
  emit(event, data) { if (this.listeners[event]) { this.listeners[event].forEach(callback => callback(data)); } }
  connect(token) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
    this.ws = new WebSocket(WEBSOCKET_URL);
    this.ws.onopen = () => { this.emit('open'); this.sendMessage({ type: 'auth', token }); };
    this.ws.onmessage = (event) => { try { this.emit('message', JSON.parse(event.data)); } catch (e) { console.error('Parse error', e); } };
    this.ws.onerror = (error) => { this.emit('error', error); };
    this.ws.onclose = (event) => { this.emit('close'); this.ws = null; };
  }
  sendMessage(message) { if (this.ws && this.ws.readyState === WebSocket.OPEN) { this.ws.send(JSON.stringify(message)); } else { console.error('WS not connected.'); } }
  close() { if (this.ws) { this.ws.close(); } }
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
      "@/*": [
        "./*"
      ]
    }
  },
  "include": [
    "**/*.ts",
    "**/*.tsx",
    ".expo/types/**/*.ts",
    "expo-env.d.ts"
  ]
}

```

