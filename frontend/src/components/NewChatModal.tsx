import React from 'react';
import Modal from './Modal';

// --- 内联类型定义 ---
export interface Prompt {
    id: number;
    user_id: number;
    name: string;
    content: string;
}

interface NewChatModalProps {
  prompts: Prompt[];
  onClose: () => void;
  onSelectPrompt: (promptId: number | null) => void;
}

const NewChatModal: React.FC<NewChatModalProps> = ({ prompts, onClose, onSelectPrompt }) => {
  return (
    <Modal title="选择一个角色开始新对话" onClose={onClose}>
      <div className="prompt-list">
        <div className="prompt-item" onClick={() => onSelectPrompt(null)}>
          <div className="prompt-name">哈基米</div>
          <div className="prompt-content">使用系统默认的哈基米助手。</div>
        </div>
        {prompts.map(prompt => (
          <div key={prompt.id} className="prompt-item" onClick={() => onSelectPrompt(prompt.id)}>
            <div className="prompt-name">{prompt.name}</div>
            <div className="prompt-content">{prompt.content.substring(0, 100)}...</div>
          </div>
        ))}
      </div>
    </Modal>
  );
};

export default NewChatModal;