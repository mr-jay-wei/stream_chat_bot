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
        isUser ? styles.userMessageContainer : styles.botMessageContainer
      ]}
    >
      {item.role === 'assistant' && item.content === '' ? (
        <Text style={styles.typingIndicator}>...</Text>
      ) : (
        <Text style={[
          styles.messageText, 
          isUser ? { color: 'white' } : { color: 'black' }
        ]}>
          {item.content}
        </Text>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  messageContainer: { padding: 12, borderRadius: 18, marginVertical: 5, maxWidth: '80%' },
  userMessageContainer: { backgroundColor: '#667eea', alignSelf: 'flex-end' },
  botMessageContainer: { backgroundColor: 'white', alignSelf: 'flex-start', borderWidth: 1, borderColor: '#e0e0e0', minHeight: 40, justifyContent: 'center' },
  messageText: { fontSize: 16 },
  typingIndicator: { fontSize: 18, color: '#999', paddingHorizontal: 5 },
});

export default React.memo(MessageItem);