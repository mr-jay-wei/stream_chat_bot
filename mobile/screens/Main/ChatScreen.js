// mobile/screens/Main/ChatScreen.js
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

export default function ChatScreen() {
    return (
      <View style={styles.container}>
        <Text>这是聊天主页面</Text>
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