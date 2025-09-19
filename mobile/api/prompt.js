// mobile/api/prompt.js
import apiClient from './index';

export const getPrompts = () => apiClient.get('/prompts');
export const createPrompt = (data) => apiClient.post('/prompts', data);
export const updatePrompt = (id, data) => apiClient.put(`/prompts/${id}`, data);
export const deletePrompt = (id) => apiClient.delete(`/prompts/${id}`);