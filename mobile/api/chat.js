// mobile/api/chat.js
import apiClient from './index';

export const getSessionMessages = async (sessionId) => {
  if (!sessionId) return [];
  try {
    const response = await apiClient.get(`/chat-sessions/${sessionId}/messages`);
    return response.data.messages || [];
  } catch (error) {
    console.error('Failed to fetch session messages:', error);
    throw error;
  }
};

export const getUserSessions = async () => {
  try {
    const response = await apiClient.get(`/conversations`);
    return response.data.conversations || [];
  } catch (error) {
    console.error('Failed to fetch user sessions:', error);
    throw error;
  }
};