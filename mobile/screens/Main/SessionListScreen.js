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