// frontend/src/api/chat.ts

import apiClient from './apiClient';

// 删除单条消息的函数
export const deleteMessage = (messageId: number) => {
  return apiClient.delete(`/messages/${messageId}`);
};