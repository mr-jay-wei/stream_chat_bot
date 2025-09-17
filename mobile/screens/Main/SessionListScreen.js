// mobile/screens/Main/SessionListScreen.js
import React, { useState, useLayoutEffect, useCallback, useContext } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, ActivityIndicator, Alert } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { AuthContext } from '../../context/AuthContext';
import { getUserSessions, deleteSession } from '../../api/chat';
import SwipeableRow from '../../components/SwipeableRow';

export default function SessionListScreen({ navigation }) {
  const { logout } = useContext(AuthContext);
  const [sessions, setSessions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  // 每次进入页面时刷新列表
  useFocusEffect(
    useCallback(() => {
      const fetchSessions = async () => {
        setIsLoading(true);
        try {
          const userSessions = await getUserSessions();
          setSessions(userSessions);
        } catch (error) {
          console.error("无法加载会话列表", error);
        } finally {
          setIsLoading(false);
        }
      };
      fetchSessions();
    }, [])
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
        <TouchableOpacity onPress={() => navigation.navigate('Chat', { sessionId: null })} style={{ paddingHorizontal: 15 }}>
          <Text style={{ color: 'white', fontSize: 24, fontWeight: 'bold' }}>+</Text>
        </TouchableOpacity>
      )
    });
  }, [navigation, logout]);

  // 处理删除会话的逻辑
  const handleDelete = (sessionId) => {
    Alert.alert(
      "确认删除",
      "确定要删除这个对话吗？所有消息都将被永久删除。",
      [
        { text: "取消", style: "cancel" },
        { 
          text: "删除", 
          onPress: async () => {
            try {
              await deleteSession(sessionId);
              setSessions(prevSessions => prevSessions.filter(s => s.id !== sessionId));
            } catch (error) {
              Alert.alert("删除失败", "无法删除该对话，请稍后重试。");
            }
          },
          style: "destructive" 
        }
      ]
    );
  };

  const renderItem = ({ item }) => (
    <SwipeableRow
      item={item}
      onDelete={() => handleDelete(item.id)}
      onNavigate={() => navigation.navigate('Chat', { sessionId: item.id })}
    />
  );

  if (isLoading) {
    return <View style={styles.centered}><ActivityIndicator size="large" color="#667eea" /></View>;
  }

  return (
    <FlatList
      data={sessions}
      renderItem={renderItem}
      keyExtractor={item => item.id.toString()}
      style={styles.container}
      ListEmptyComponent={<View style={styles.centered}><Text style={styles.emptyText}>没有历史会话{"\n"}点击左上角 '+' 开始新聊天</Text></View>}
    />
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  emptyText: { textAlign: 'center', fontSize: 16, color: '#999', lineHeight: 24 },
});