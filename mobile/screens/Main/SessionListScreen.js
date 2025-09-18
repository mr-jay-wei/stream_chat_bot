// mobile/screens/Main/SessionListScreen.js
import React, { useState, useLayoutEffect, useCallback, useContext } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, ActivityIndicator, Alert, Button } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { AuthContext } from '../../context/AuthContext';
import { getUserSessions, deleteSession } from '../../api/chat';
import SwipeableRow from '../../components/SwipeableRow';

export default function SessionListScreen({ navigation }) {
  const { logout } = useContext(AuthContext);
  const [sessions, setSessions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null); // 👈 新增：错误状态

  // 将数据获取逻辑封装成一个可复用的函数
  const fetchSessions = useCallback(async () => {
    setIsLoading(true);
    setError(null); // 每次获取前重置错误状态
    try {
      const userSessions = await getUserSessions();
      setSessions(userSessions);
    } catch (err) {
      console.error("无法加载会话列表", err);
      setError("无法加载会话列表，请检查您的网络连接。"); // 👈 设置错误信息
    } finally {
      setIsLoading(false);
    }
  }, []);

  // 每次进入页面时，调用fetchSessions
  useFocusEffect(
    useCallback(() => {
      fetchSessions();
    }, [fetchSessions])
  );

  // 设置导航栏按钮
  useLayoutEffect(() => {
    navigation.setOptions({
      title: '我的对话',
      headerRight: () => (
        <TouchableOpacity onPress={logout} style={{ paddingHorizontal: 10 }}>
          <Text style={{ color: 'white', fontSize: 16 }}>登出</Text>
        </TouchableOpacity>
      ),
      headerLeft: () => (
        <TouchableOpacity
          onPress={() => navigation.navigate('Chat', { sessionId: null })}
          style={{ paddingHorizontal: 15 }}>
          <Text style={{ color: 'white', fontSize: 24, fontWeight: 'bold' }}>+</Text>
        </TouchableOpacity>
      ),
    });
  }, [navigation, logout]);

  // 处理删除会话的逻辑
  const handleDelete = sessionId => {
    Alert.alert('确认删除', '确定要删除这个对话吗？所有消息都将被永久删除。', [
      { text: '取消', style: 'cancel' },
      {
        text: '删除',
        onPress: async () => {
          try {
            await deleteSession(sessionId);
            setSessions(prevSessions => prevSessions.filter(s => s.id !== sessionId));
          } catch (error) {
            Alert.alert('删除失败', '无法删除该对话，请稍后重试。');
          }
        },
        style: 'destructive',
      },
    ]);
  };

  const renderItem = ({ item }) => (
    <SwipeableRow
      item={item}
      onDelete={() => handleDelete(item.id)}
      onNavigate={() => navigation.navigate('Chat', { sessionId: item.id })}
    />
  );

  // 根据不同状态，渲染不同的UI
  const renderContent = () => {
    if (isLoading) {
      return <View style={styles.centered}><ActivityIndicator size="large" color="#667eea" /></View>;
    }

    if (error) {
      return (
        <View style={styles.centered}>
          <Text style={styles.errorText}>{error}</Text>
          <Button title="点我重试" onPress={fetchSessions} color="#667eea" />
        </View>
      );
    }

    return (
      <FlatList
        data={sessions}
        renderItem={renderItem}
        keyExtractor={item => item.id.toString()}
        onRefresh={fetchSessions} // 👈 新增：下拉刷新功能
        refreshing={isLoading}    // 👈 新增：控制下拉刷新的加载动画
        ListEmptyComponent={
          <View style={styles.centered}>
            <Text style={styles.emptyText}>没有历史会话{"\n"}点击左上角 '+' 开始新聊天</Text>
          </View>
        }
      />
    );
  };

  return <View style={styles.container}>{renderContent()}</View>;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  emptyText: { textAlign: 'center', fontSize: 16, color: '#999', lineHeight: 24 },
  errorText: { textAlign: 'center', fontSize: 16, color: 'red', marginBottom: 20 },
});