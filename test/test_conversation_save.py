#!/usr/bin/env python3
# 测试对话保存功能

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import Conversation
from datetime import datetime

def test_save_conversation():
    """测试保存对话到数据库"""
    db = SessionLocal()
    try:
        # 创建测试对话
        conversation = Conversation(
            user_id=7,  # xiaobai@123.com的ID
            question="测试问题 - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            answer="测试回答 - 这是一个测试回答"
        )
        
        print(f"准备保存对话: {conversation.question}")
        
        # 添加到数据库
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
        print(f"✅ 对话保存成功! ID: {conversation.id}")
        print(f"   用户ID: {conversation.user_id}")
        print(f"   问题: {conversation.question}")
        print(f"   回答: {conversation.answer}")
        print(f"   创建时间: {conversation.created_at}")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存对话失败: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("🧪 测试对话保存功能...")
    success = test_save_conversation()
    
    if success:
        print("\n🎉 测试成功！对话保存功能正常工作。")
    else:
        print("\n💥 测试失败！对话保存功能有问题。")