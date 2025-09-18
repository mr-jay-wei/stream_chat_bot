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
