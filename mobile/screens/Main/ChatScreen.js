// mobile/screens/Main/ChatScreen.js
import React, { useLayoutEffect, useContext, useState, useEffect, useRef, useCallback } from 'react';
import { 
  View, Text, StyleSheet, TextInput, TouchableOpacity, 
  FlatList, KeyboardAvoidingView, Platform, 
  TouchableWithoutFeedback, Keyboard, ActivityIndicator, Alert
} from 'react-native';
import { AuthContext } from '../../context/AuthContext';
import { webSocketClient } from '../../services/WebSocketClient';
import { getSessionMessages, deleteMessage } from '../../api/chat';
import MessageItem from '../../components/MessageItem';

export default function ChatScreen({ navigation, route }) {
  const { logout, userToken } = useContext(AuthContext);
  const initialSessionId = route.params?.sessionId;

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [currentSessionId, setCurrentSessionId] = useState(initialSessionId);
  const [isLoadingHistory, setIsLoadingHistory] = useState(!!initialSessionId);
  
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

  // Effect for handling WebSocket messages and loading history
  useEffect(() => {
    const loadHistory = async (sessionId) => {
      if (!sessionId) return;
      setIsLoadingHistory(true);
      try {
        const historyMessages = await getSessionMessages(sessionId);
        const formattedMessages = historyMessages.map(msg => ({
          id: msg.id.toString(),
          role: msg.role,
          content: msg.content,
        }));
        setMessages(formattedMessages);
      } catch (error) {
        console.error("加载历史消息失败", error);
      } finally {
        setIsLoadingHistory(false);
      }
    };

    if (initialSessionId) {
      loadHistory(initialSessionId);
    }
    
    const handleWebSocketMessage = (message) => {
      if (message.type === 'processing' && message.data.session_id && currentSessionId === null) {
        console.log(`Session ID updated from null to: ${message.data.session_id}`);
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
  }, [initialSessionId, currentSessionId]);

  // Effect for setting navigation options dynamically
  useLayoutEffect(() => {
    navigation.setOptions({
        title: initialSessionId ? '继续对话' : '新对话'
    });
  }, [navigation, initialSessionId]);

  // Effect for auto-scrolling
  useEffect(() => {
    if (flatListRef.current && messages.length > 0) {
      flatListRef.current.scrollToEnd({ animated: true });
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
  
  const handleLongPressMessage = (messageId) => {
    // 确保messageId是有效的，避免对临时消息进行操作
    if (String(messageId).startsWith('bot-') || String(messageId).startsWith('user-')) {
        return;
    }
    Alert.alert(
      "确认删除",
      "要删除这条消息吗？",
      [
        { text: "取消", style: "cancel" },
        {
          text: "删除",
          onPress: async () => {
            try {
              await deleteMessage(messageId);
              setMessages(prevMessages => prevMessages.filter(m => m.id !== messageId.toString()));
            } catch (error) {
              Alert.alert("删除失败", "无法删除该消息，请稍后重试。");
            }
          },
          style: "destructive"
        }
      ]
    );
  };

  const renderMessage = useCallback(({ item }) => (
    <MessageItem item={item} onLongPress={handleLongPressMessage} />
  ), []);

  return (
    <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === "ios" ? "padding" : "height"} keyboardVerticalOffset={90}>
      <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
        <View style={{ flex: 1 }}>
          {isLoadingHistory ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#667eea" />
            </View>
          ) : (
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
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  messageList: { flex: 1, paddingHorizontal: 10, paddingTop: 10 },
  emptyContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', paddingTop: '50%' },
  emptyText: { fontSize: 18, color: '#aaa' },
  inputContainer: { flexDirection: 'row', padding: 10, borderTopWidth: 1, borderTopColor: '#ddd', backgroundColor: 'white' },
  input: { flex: 1, height: 40, borderWidth: 1, borderColor: '#ddd', borderRadius: 20, paddingHorizontal: 15, backgroundColor: '#f5f5f5' },
  sendButton: { marginLeft: 10, justifyContent: 'center', alignItems: 'center', backgroundColor: '#667eea', borderRadius: 20, paddingHorizontal: 15 },
  sendButtonText: { color: 'white', fontWeight: 'bold' },
});