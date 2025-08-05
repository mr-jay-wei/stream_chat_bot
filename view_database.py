#!/usr/bin/env python3
# 数据库查看工具 - 适合数据库小白使用

from app.database import SessionLocal
from app.models import User, ChatSession, Message, Conversation
from sqlalchemy import text
import pandas as pd
from datetime import datetime

def show_table_info():
    """显示所有表的基本信息"""
    db = SessionLocal()
    try:
        print("🗄️ 数据库表信息")
        print("=" * 60)
        
        # 获取所有表名
        result = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        
        tables = [row[0] for row in result.fetchall()]
        
        for table in tables:
            # 获取表的记录数
            count_result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = count_result.scalar()
            
            print(f"📋 表名: {table}")
            print(f"   记录数: {count}")
            
            # 获取表结构
            structure_result = db.execute(text(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
            """))
            
            print("   字段:")
            for col_name, data_type, nullable in structure_result.fetchall():
                null_info = "可空" if nullable == "YES" else "非空"
                print(f"     - {col_name}: {data_type} ({null_info})")
            print()
            
    except Exception as e:
        print(f"❌ 获取表信息失败: {e}")
    finally:
        db.close()

def show_users():
    """显示用户表内容"""
    db = SessionLocal()
    try:
        print("👥 用户表 (users)")
        print("=" * 60)
        
        users = db.query(User).all()
        
        if not users:
            print("📭 用户表为空")
            return
        
        print(f"📊 总用户数: {len(users)}")
        print()
        
        for user in users:
            print(f"🆔 ID: {user.id}")
            print(f"📧 邮箱: {user.email}")
            print(f"✅ 激活状态: {'是' if user.is_active else '否'}")
            print(f"📅 注册时间: {user.created_at}")
            print(f"🔄 更新时间: {user.updated_at}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 查看用户表失败: {e}")
    finally:
        db.close()

def show_chat_sessions():
    """显示会话表内容"""
    db = SessionLocal()
    try:
        print("💬 会话表 (chat_sessions)")
        print("=" * 60)
        
        sessions = db.query(ChatSession).all()
        
        if not sessions:
            print("📭 会话表为空")
            return
        
        print(f"📊 总会话数: {len(sessions)}")
        print()
        
        for session in sessions:
            # 获取会话的消息数量
            message_count = db.query(Message).filter(Message.chat_session_id == session.id).count()
            
            print(f"🆔 会话ID: {session.id}")
            print(f"👤 用户ID: {session.user_id}")
            print(f"📝 标题: {session.title}")
            print(f"💬 消息数: {message_count}")
            print(f"📅 创建时间: {session.created_at}")
            print(f"🔄 更新时间: {session.updated_at}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 查看会话表失败: {e}")
    finally:
        db.close()

def show_messages(session_id=None, limit=10):
    """显示消息表内容"""
    db = SessionLocal()
    try:
        print("💭 消息表 (messages)")
        print("=" * 60)
        
        query = db.query(Message)
        if session_id:
            query = query.filter(Message.chat_session_id == session_id)
            print(f"🔍 筛选条件: 会话ID = {session_id}")
        
        messages = query.order_by(Message.created_at.desc()).limit(limit).all()
        
        if not messages:
            print("📭 消息表为空")
            return
        
        print(f"📊 显示最新 {len(messages)} 条消息:")
        print()
        
        for msg in messages:
            role_icon = "👤" if msg.role == "user" else "🤖"
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            
            print(f"🆔 消息ID: {msg.id}")
            print(f"💬 会话ID: {msg.chat_session_id}")
            print(f"{role_icon} 角色: {msg.role}")
            print(f"📝 内容: {content_preview}")
            print(f"📅 时间: {msg.created_at}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 查看消息表失败: {e}")
    finally:
        db.close()

def show_conversations(limit=10):
    """显示旧对话表内容"""
    db = SessionLocal()
    try:
        print("📜 旧对话表 (conversations)")
        print("=" * 60)
        
        conversations = db.query(Conversation).order_by(Conversation.created_at.desc()).limit(limit).all()
        
        if not conversations:
            print("📭 对话表为空")
            return
        
        print(f"📊 显示最新 {len(conversations)} 条对话:")
        print()
        
        for conv in conversations:
            question_preview = conv.question[:50] + "..." if len(conv.question) > 50 else conv.question
            answer_preview = conv.answer[:50] + "..." if len(conv.answer) > 50 else conv.answer
            
            print(f"🆔 对话ID: {conv.id}")
            print(f"👤 用户ID: {conv.user_id}")
            print(f"❓ 问题: {question_preview}")
            print(f"💡 回答: {answer_preview}")
            print(f"📅 时间: {conv.created_at}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 查看对话表失败: {e}")
    finally:
        db.close()

def show_user_activity(user_id):
    """显示特定用户的活动"""
    db = SessionLocal()
    try:
        print(f"👤 用户 {user_id} 的活动详情")
        print("=" * 60)
        
        # 获取用户信息
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            print(f"❌ 未找到用户ID {user_id}")
            return
        
        print(f"📧 用户邮箱: {user.email}")
        print(f"📅 注册时间: {user.created_at}")
        print()
        
        # 获取用户的会话
        sessions = db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
        print(f"💬 总会话数: {len(sessions)}")
        
        for session in sessions:
            message_count = db.query(Message).filter(Message.chat_session_id == session.id).count()
            print(f"  📁 {session.title} ({message_count}条消息)")
        
        # 获取用户的旧对话
        old_conversations = db.query(Conversation).filter(Conversation.user_id == user_id).count()
        print(f"📜 旧对话数: {old_conversations}")
        
    except Exception as e:
        print(f"❌ 查看用户活动失败: {e}")
    finally:
        db.close()

def main_menu():
    """主菜单"""
    while True:
        print("\n🗄️ 数据库查看工具")
        print("=" * 40)
        print("1. 📋 查看所有表信息")
        print("2. 👥 查看用户表")
        print("3. 💬 查看会话表")
        print("4. 💭 查看消息表")
        print("5. 📜 查看旧对话表")
        print("6. 👤 查看特定用户活动")
        print("7. 🔍 自定义查询")
        print("0. 🚪 退出")
        print("-" * 40)
        
        choice = input("请选择操作 (0-7): ").strip()
        
        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            show_table_info()
        elif choice == "2":
            show_users()
        elif choice == "3":
            show_chat_sessions()
        elif choice == "4":
            limit = input("显示多少条消息？(默认10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            show_messages(limit=limit)
        elif choice == "5":
            limit = input("显示多少条对话？(默认10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            show_conversations(limit=limit)
        elif choice == "6":
            user_id = input("请输入用户ID: ").strip()
            if user_id.isdigit():
                show_user_activity(int(user_id))
            else:
                print("❌ 请输入有效的用户ID")
        elif choice == "7":
            custom_query()
        else:
            print("❌ 无效选择，请重试")
        
        input("\n按回车键继续...")

def custom_query():
    """自定义SQL查询"""
    db = SessionLocal()
    try:
        print("\n🔍 自定义查询")
        print("=" * 40)
        print("⚠️  注意：只支持SELECT查询，确保查询安全")
        print("示例查询:")
        print("  SELECT * FROM users LIMIT 5;")
        print("  SELECT COUNT(*) FROM messages;")
        print()
        
        query = input("请输入SQL查询: ").strip()
        
        if not query.upper().startswith("SELECT"):
            print("❌ 只支持SELECT查询")
            return
        
        result = db.execute(text(query))
        rows = result.fetchall()
        
        if not rows:
            print("📭 查询结果为空")
            return
        
        # 获取列名
        columns = result.keys()
        
        print(f"\n📊 查询结果 ({len(rows)} 行):")
        print("-" * 60)
        
        # 打印表头
        header = " | ".join(f"{col:15}" for col in columns)
        print(header)
        print("-" * len(header))
        
        # 打印数据
        for row in rows[:20]:  # 最多显示20行
            row_data = " | ".join(f"{str(val)[:15]:15}" for val in row)
            print(row_data)
        
        if len(rows) > 20:
            print(f"... 还有 {len(rows) - 20} 行数据")
            
    except Exception as e:
        print(f"❌ 查询失败: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print("🎉 欢迎使用数据库查看工具！")
    print("这个工具可以帮助你轻松查看数据库中的所有内容")
    main_menu()