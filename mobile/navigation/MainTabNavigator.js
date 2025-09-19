// mobile/navigation/MainTabNavigator.js
import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import SessionListScreen from '../screens/Main/SessionListScreen';
import ChatScreen from '../screens/Main/ChatScreen';
import PromptListScreen from '../screens/Main/PromptListScreen';
import PromptEditScreen from '../screens/Main/PromptEditScreen';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

function ChatStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="SessionList" component={SessionListScreen} />
      <Stack.Screen name="Chat" component={ChatScreen} />
    </Stack.Navigator>
  );
}

function PromptStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="PromptList" component={PromptListScreen} />
      <Stack.Screen name="PromptEdit" component={PromptEditScreen} />
    </Stack.Navigator>
  );
}

export default function MainTabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarActiveTintColor: '#667eea',
        tabBarInactiveTintColor: 'gray',
        headerShown: false,
      }}
    >
      <Tab.Screen 
        name="ChatStack" 
        component={ChatStack} 
        options={{ title: '对话' }}
      />
      <Tab.Screen 
        name="PromptStack" 
        component={PromptStack} 
        options={{ title: '我的角色' }}
      />
    </Tab.Navigator>
  );
}