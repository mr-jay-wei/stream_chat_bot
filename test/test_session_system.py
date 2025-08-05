#!/usr/bin/env python3
# 测试会话管理系统

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.session_manager import session_manager
from app.models import User, ChatSession, Message
from datetime import datetime

def test_session_system():
    """测试完整的会话管理系统"""
    db = SessionLocal()
    try:
        print("🧪 测试会话管理系统...")
        print("=" * 50)
        
        # 1. 获取测试用户
        user = db.query(User).filter(User.email == "xiaobai@123.com").first()
        if not user:
            print("❌ 未找到测试用户 xiaobai@123.com")
            return False
        
        print(f"✅ 找到测试用户: {user.email} (ID: {user.id})")
        
        # 2. 创建新会话
        print("\n📝 测试创建新会话...")
        session_id = session_manager.create_new_session(db, user.id, "测试会话 - Python学习")
        if session_id:
            print(f"✅ 创建新会话成功: ID {session_id}")
        else:
            print("❌ 创建新会话失败")
            return False
        
        # 3. 添加消息到会话
        print("\n💬 测试添加消息...")
        
        # 添加用户消息
        msg1_id = session_manager.add_message_to_session(db, session_id, "user", "你好，我想学习Python")
        if msg1_id:
            print(f"✅ 添加用户消息成功: ID {msg1_id}")
        else:
            print("❌ 添加用户消息失败")
            return False
        
        # 添加AI回复
        msg2_id = session_manager.add_message_to_session(db, session_id, "assistant", "你好！我很乐意帮助你学习Python。Python是一门非常适合初学者的编程语言...")
        if msg2_id:
            print(f"✅ 添加AI回复成功: ID {msg2_id}")
        else:
            print("❌ 添加AI回复失败")
            return False
        
        # 4. 获取会话消息
        print("\n📖 测试获取会话消息...")
        messages = session_manager.get_session_messages(db, session_id, user.id)
        print(f"✅ 获取到 {len(messages)} 条消息:")
        for msg in messages:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            content_preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            print(f"   {role_icon} {msg['role']}: {content_preview}")
        
        # 5. 获取用户会话列表
        print("\n📋 测试获取用户会话列表...")
        sessions = session_manager.get_user_sessions(db, user.id)
        print(f"✅ 用户共有 {len(sessions)} 个会话:")
        for session in sessions[:3]:  # 只显示前3个
            print(f"   📁 {session['title']} ({session['message_count']}条消息)")
        
        # 6. 测试会话上下文
        print("\n🧠 测试获取会话上下文...")
        context = session_manager.get_session_context_for_ai(db, session_id, user.id)
        print(f"✅ 获取到 {len(context)} 条上下文消息")
        
        # 7. 测试删除会话
        print(f"\n🗑️ 测试删除会话 {session_id}...")
        delete_success = session_manager.delete_session(db, user.id, session_id)
        if delete_success:
            print("✅ 删除会话成功")
        else:
            print("❌ 删除会话失败")
            return False
        
        print("\n🎉 所有测试通过！会话管理系统工作正常。")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    success = test_session_system()
    if success:
        print("\n✅ 会话管理系统测试完成，所有功能正常！")
    else:
        print("\n❌ 会话管理系统测试失败，需要进一步调试。")