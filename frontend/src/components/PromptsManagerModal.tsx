// frontend/src/components/PromptsManagerModal.tsx

import React, { useState, useEffect } from 'react';
import Modal from './Modal';
import apiClient from '../api/apiClient';

// (类型定义部分保持不变)
export interface Prompt {
    id: number;
    user_id: number;
    name: string;
    content: string;
}

const PromptsManagerModal: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [editingPrompt, setEditingPrompt] = useState<Partial<Prompt> | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchPrompts = async () => {
    setIsLoading(true);
    try {
        const response = await apiClient.get<Prompt[]>('/prompts');
        setPrompts(response.data);
    } catch (error) {
        console.error("Failed to fetch prompts", error);
        alert("加载角色列表失败");
    } finally {
        setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPrompts();
  }, []);

  const handleSave = async () => {
    if (!editingPrompt || !editingPrompt.name?.trim() || !editingPrompt.content?.trim()) {
      alert('角色名称和设定不能为空');
      return;
    }
    try {
      if (editingPrompt.id) {
        await apiClient.put(`/prompts/${editingPrompt.id}`, { name: editingPrompt.name, content: editingPrompt.content });
      } else {
        await apiClient.post('/prompts', { name: editingPrompt.name, content: editingPrompt.content });
      }
      setEditingPrompt(null);
      fetchPrompts();
    } catch (error) {
      alert('保存失败');
    }
  };

  const handleDelete = async (id: number) => {
    if (window.confirm('确定要删除这个角色吗? 这将永久移除它。')) {
      try {
        await apiClient.delete(`/prompts/${id}`);
        fetchPrompts();
      } catch (error) {
        alert('删除失败');
      }
    }
  };

  return (
    <Modal title="管理我的角色" onClose={onClose}>
      {editingPrompt ? (
        <div className="prompt-form">
          <input
            type="text"
            placeholder="角色名称"
            value={editingPrompt.name || ''}
            onChange={(e) => setEditingPrompt({ ...editingPrompt, name: e.target.value })}
          />
          <textarea
            placeholder="角色设定 (例如：你是一位严格的雅思口语考官...)"
            value={editingPrompt.content || ''}
            onChange={(e) => setEditingPrompt({ ...editingPrompt, content: e.target.value })}
          />
          <div className="prompt-form-actions">
            <button className="cancel-btn" onClick={() => setEditingPrompt(null)}>取消</button>
            <button className="save-btn" onClick={handleSave}>保存</button>
          </div>
        </div>
      ) : (
        <>
          {/* --- 关键改动：使用了新的 btn-primary 样式 --- */}
          <button className="btn-primary" onClick={() => setEditingPrompt({ name: '', content: '' })}>+ 新建角色</button>
          {isLoading ? (
              <p style={{textAlign: 'center', margin: '20px'}}>正在加载...</p>
          ) : (
            <div className="prompt-list" style={{marginTop: '20px'}}>
                {prompts.length === 0 ? (
                    <p style={{textAlign: 'center', color: '#666'}}>你还没有创建任何角色。</p>
                ) : (
                    prompts.map(prompt => (
                    <div key={prompt.id} className="prompt-item">
                        <div className="prompt-item-header">
                            <div className="prompt-name">{prompt.name}</div>
                            <div className="prompt-actions">
                                <button title="编辑" onClick={() => setEditingPrompt(prompt)}>✏️</button>
                                <button title="删除" onClick={() => handleDelete(prompt.id)}>🗑️</button>
                            </div>
                        </div>
                        <div className="prompt-content">{prompt.content}</div>
                    </div>
                    ))
                )}
            </div>
          )}
        </>
      )}
    </Modal>
  );
};

export default PromptsManagerModal;