#!/usr/bin/env python3
# diagnose_database.py - 诊断数据库和对话保存问题

import sys
import os
from datetime import datetime, date
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据库连接配置
DATABASE_URL = "postgresql://postgres:052756@localhost:5432/chatbot_db"

def connect_database():
    """连接数据库"""
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        return engine, SessionLocal()
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return None, None

def diagnose_conversations(db):
    """诊断对话记录"""
    print("🔍 诊断对话记录...")
    print("=" * 50)
    
    try:
        # 检查总对话数量
        total_result = db.execute(text("SELECT COUNT(*) FROM conversations"))
        total_count = total_result.scalar()
        print(f"📊 总对话记录数: {total_count}")
        
        # 检查今天的对话
        today_result = db.execute(text("""
            SELECT COUNT(*) FROM conversations 
            WHERE DATE(created_at) = CURRENT_DATE
        """))
        today_count = today_result.scalar()
        print(f"📅 今天的对话记录数: {today_count}")
        
        # 检查昨天的对话
        yesterday_result = db.execute(text("""
            SELECT COUNT(*) FROM conversations 
            WHERE DATE(created_at) = CURRENT_DATE - INTERVAL '1 day'
        """))
        yesterday_count = yesterday_result.scalar()
        print(f"📅 昨天的对话记录数: {yesterday_count}")
        
        # 检查最近7天的对话统计
        print(f"\n📈 最近7天对话统计:")
        week_result = db.execute(text("""
            SELECT DATE(created_at) as 日期, COUNT(*) as 对话数量
            FROM conversations 
            WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY DATE(created_at)
            ORDER BY 日期 DESC
        """))
        
        for row in week_result:
            date_str = row[0].strftime("%Y-%m-%d")
            is_today = "（今天）" if row[0] == date.today() else ""
            print(f"  - {date_str}{is_today}: {row[1]} 条对话")
        
        # 检查最新的5条对话
        print(f"\n📄 最新的5条对话:")
        latest_result = db.execute(text("""
            SELECT u.email, c.question, c.created_at
            FROM conversations c
            JOIN users u ON c.user_id = u.id
            ORDER BY c.created_at DESC
            LIMIT 5
        """))
        
        for row in latest_result:
            question = row[1][:50] + "..." if len(row[1]) > 50 else row[1]
            time_str = row[2].strftime("%Y-%m-%d %H:%M:%S")
            print(f"  - {row[0]}: {question} ({time_str})")
            
    except Exception as e:
        print(f"❌ 诊断对话记录失败: {e}")

def diagnose_users(db):
    """诊断用户信息"""
    print(f"\n👥 诊断用户信息...")
    print("=" * 50)
    
    try:
        # 检查用户总数
        user_result = db.execute(text("SELECT COUNT(*) FROM users"))
        user_count = user_result.scalar()
        print(f"👤 总用户数: {user_count}")
        
        # 检查活跃用户
        active_result = db.execute(text("SELECT COUNT(*) FROM users WHERE is_active = true"))
        active_count = active_result.scalar()
        print(f"✅ 活跃用户数: {active_count}")
        
        # 检查用户列表
        print(f"\n📋 用户列表:")
        users_result = db.execute(text("""
            SELECT id, email, created_at, is_active
            FROM users 
            ORDER BY created_at DESC
        """))
        
        for row in users_result:
            status = "✅" if row[3] else "❌"
            time_str = row[2].strftime("%Y-%m-%d %H:%M:%S")
            print(f"  - {status} {row[1]} (ID: {row[0]}, 注册: {time_str})")
            
    except Exception as e:
        print(f"❌ 诊断用户信息失败: {e}")

def diagnose_chat_sessions(db):
    """诊断聊天会话"""
    print(f"\n💭 诊断聊天会话...")
    print("=" * 50)
    
    try:
        # 检查会话总数
        session_result = db.execute(text("SELECT COUNT(*) FROM chat_sessions"))
        session_count = session_result.scalar()
        print(f"💬 总会话数: {session_count}")
        
        # 检查消息总数
        message_result = db.execute(text("SELECT COUNT(*) FROM messages"))
        message_count = message_result.scalar()
        print(f"📝 总消息数: {message_count}")
        
        # 检查今天的会话和消息
        today_sessions = db.execute(text("""
            SELECT COUNT(*) FROM chat_sessions 
            WHERE DATE(created_at) = CURRENT_DATE
        """)).scalar()
        
        today_messages = db.execute(text("""
            SELECT COUNT(*) FROM messages 
            WHERE DATE(created_at) = CURRENT_DATE
        """)).scalar()
        
        print(f"📅 今天的会话数: {today_sessions}")
        print(f"📅 今天的消息数: {today_messages}")
        
    except Exception as e:
        print(f"❌ 诊断聊天会话失败: {e}")

def check_database_structure(db):
    """检查数据库结构"""
    print(f"\n🏗️  检查数据库结构...")
    print("=" * 50)
    
    try:
        # 检查表是否存在
        tables_result = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """))
        
        tables = [row[0] for row in tables_result]
        print(f"📚 数据库表: {', '.join(tables)}")
        
        # 检查关键表的结构
        for table in ['users', 'conversations', 'chat_sessions', 'messages']:
            if table in tables:
                print(f"\n✅ 表 {table} 存在")
                # 检查记录数
                count_result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = count_result.scalar()
                print(f"   记录数: {count}")
            else:
                print(f"❌ 表 {table} 不存在")
                
    except Exception as e:
        print(f"❌ 检查数据库结构失败: {e}")

def main():
    """主函数"""
    print("🔍 诊断聊天机器人数据库问题")
    print("=" * 60)
    print(f"🕐 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 连接数据库
    engine, db = connect_database()
    if not db:
        return
    
    try:
        # 执行各项诊断
        check_database_structure(db)
        diagnose_users(db)
        diagnose_conversations(db)
        diagnose_chat_sessions(db)
        
        print(f"\n" + "=" * 60)
        print("🎯 诊断建议:")
        
        # 检查今天是否有对话
        today_result = db.execute(text("""
            SELECT COUNT(*) FROM conversations 
            WHERE DATE(created_at) = CURRENT_DATE
        """))
        today_count = today_result.scalar()
        
        if today_count == 0:
            print("⚠️  今天没有对话记录，可能的原因：")
            print("   1. WebSocket认证失败")
            print("   2. 对话保存逻辑有问题")
            print("   3. 今天确实没有发送对话")
            print("   建议：重新启动应用，发送一条测试消息")
        else:
            print(f"✅ 今天有 {today_count} 条对话记录，系统正常")
        
        print(f"\n✅ 诊断完成!")
        
    except Exception as e:
        print(f"❌ 诊断过程失败: {e}")
    
    finally:
        db.close()

if __name__ == "__main__":
    main()