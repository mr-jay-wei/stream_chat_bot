// mobile/screens/Main/PromptEditScreen.js
import React, { useState, useLayoutEffect } from 'react';
import { ScrollView, TextInput, StyleSheet, Button, Alert, KeyboardAvoidingView, Platform, TouchableOpacity, Text } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { createPrompt, updatePrompt } from '../../api/prompt';


export default function PromptEditScreen({ route, navigation }) {
  const { prompt } = route.params;
  const isEditing = !!prompt;

  const [name, setName] = useState(prompt?.name || '');
  const [content, setContent] = useState(prompt?.content || '');
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    if (!name.trim() || !content.trim()) {
      Alert.alert('提示', '角色名称和设定不能为空');
      return;
    }
    setIsSaving(true);
    try {
      if (isEditing) {
        await updatePrompt(prompt.id, { name, content });
      } else {
        await createPrompt({ name, content });
      }
      navigation.goBack();
    } catch (error) {
      Alert.alert('错误', '保存失败，请稍后重试');
    } finally {
      setIsSaving(false);
    }
  };
  
  useLayoutEffect(() => {
    navigation.setOptions({
      title: isEditing ? '编辑角色' : '新建角色',
      headerStyle: { backgroundColor: '#667eea' },
      headerTintColor: '#fff',
      headerRight: () => (
        <TouchableOpacity
          onPress={handleSave}
          disabled={isSaving}
          style={{ marginRight: 10, opacity: isSaving ? 0.6 : 1 }}
        >
          <Text style={{ color: '#fff', fontSize: 16 }}>
            {isSaving ? '保存中...' : '保存'}
          </Text>
        </TouchableOpacity>
      ),
    });
  }, [navigation, isEditing, name, content, isSaving]);

  return (
    <SafeAreaView style={styles.container} edges={['left', 'right', 'bottom']}>
      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={styles.scrollContainer}>
          <TextInput
            style={styles.input}
            placeholder="角色名称 (例如：语言教练)"
            value={name}
            onChangeText={setName}
          />
          <TextInput
            style={[styles.input, styles.textArea]}
            placeholder="角色设定 (例如：你是一位严格的雅思口语考官...)"
            value={content}
            onChangeText={setContent}
            multiline
          />
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  scrollContainer: { padding: 15 },
  input: { backgroundColor: 'white', borderWidth: 1, borderColor: '#ccc', borderRadius: 8, padding: 15, fontSize: 16, marginBottom: 20 },
  textArea: { height: 300, textAlignVertical: 'top' },
});