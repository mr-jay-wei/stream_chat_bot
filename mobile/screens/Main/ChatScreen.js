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