// mobile/screens/Main/SessionListScreen.js
import React, { useState, useLayoutEffect, useCallback, useContext } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { AuthContext } from '../../context/AuthContext';
import { getUserSessions } from '../../api/chat';

export default function SessionListScreen({ navigation }) {
  const { logout } = useContext(AuthContext);
  const [sessions, setSessions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

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

  const renderItem = ({ item }) => (
    <TouchableOpacity style={styles.sessionItem} onPress={() => navigation.navigate('Chat', { sessionId: item.id })}>
      <Text style={styles.sessionTitle} numberOfLines={1}>{item.title}</Text>
      <Text style={styles.sessionPreview} numberOfLines={1}>{item.preview}</Text>
    </TouchableOpacity>
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
  emptyText: { textAlign: 'center', fontSize: 16, color: '#999' },
  sessionItem: { backgroundColor: 'white', padding: 20, borderBottomWidth: 1, borderBottomColor: '#eee' },
  sessionTitle: { fontSize: 16, fontWeight: 'bold', marginBottom: 5 },
  sessionPreview: { fontSize: 14, color: '#666' },
});