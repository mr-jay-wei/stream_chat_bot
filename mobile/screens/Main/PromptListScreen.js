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