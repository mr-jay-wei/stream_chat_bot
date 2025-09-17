// mobile/api/chat.js
import apiClient from './index';

/**
 * 获取指定会话的历史消息
 * @param {number} sessionId 会话ID
 * @returns {Promise<Array>} 消息列表
 */
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

/**
 * 获取当前用户的所有会话列表
 * @returns {Promise<Array>} 会话列表
 */
export const getUserSessions = async () => {
  try {
    const response = await apiClient.get(`/conversations`);
    return response.data.conversations || [];
  } catch (error) {
    console.error('Failed to fetch user sessions:', error);
    throw error;
  }
};

/**
 * 删除指定的会话
 * @param {number} sessionId 要删除的会话ID
 * @returns {Promise<object>} 后端返回的成功信息
 */
export const deleteSession = async (sessionId) => {
  try {
    const response = await apiClient.delete(`/chat-sessions/${sessionId}`);
    return response.data;
  } catch (error) {
    console.error(`Failed to delete session ${sessionId}:`, error);
    throw error;
  }
};

/**
 * 删除指定的消息
 * @param {number} messageId 要删除的消息ID
 * @returns {Promise<object>} 后端返回的成功信息
 */
export const deleteMessage = async (messageId) => {
  try {
    const response = await apiClient.delete(`/messages/${messageId}`);
    return response.data;
  } catch (error)
 {
    console.error(`Failed to delete message ${messageId}:`, error);
    throw error;
  }
};