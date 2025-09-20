// frontend/src/components/MessageItem.tsx

import React from 'react';

// --- 类型定义 ---
export interface Message {
  id: number;
  chat_session_id: number;
  role: 'user' | 'assistant';
  content: string;
}

interface MessageItemProps {
  message: Message;
  showAvatar: boolean;
  onDelete: (messageId: number) => void;
}

const MessageItem: React.FC<MessageItemProps> = ({ message, showAvatar, onDelete }) => {
  const messageClass = `message ${message.role}-message ${showAvatar ? '' : 'no-avatar'}`;

  // 只有当消息ID是数字时（意味着它已经保存在数据库），才显示删除按钮
  const canBeDeleted = typeof message.id === 'number' && message.id > 0;

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // 防止触发其他点击事件
    if (window.confirm('确定要删除这条消息吗？')) {
      onDelete(message.id);
    }
  };

  return (
    <div className={messageClass}>
      <div className="message-avatar">
        {showAvatar && (
          message.role === 'user' ? '👤' : <img src="/images/my-logo.png" alt="Bot" className="avatar-logo" />
        )}
      </div>
      <div className="message-content">
        {message.content}
        {canBeDeleted && (
          <button className="delete-message-btn" title="删除消息" onClick={handleDeleteClick}>
            🗑️
          </button>
        )}
      </div>
    </div>
  );
};

export default MessageItem;