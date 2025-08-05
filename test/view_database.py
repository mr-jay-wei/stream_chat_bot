#!/usr/bin/env python3
# view_database.py - 查看数据库内容的简单脚本

import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

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

def show_table_info(db, table_name):
    """显示表的结构和数据"""
    print(f"\n📋 表: {table_name}")
    print("=" * 50)
    
    try:
        # 查看表结构
        result = db.execute(text(f"""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """))
        
        print("🏗️  表结构:")
        for row in result:
            nullable = "可空" if row[2] == "YES" else "不可空"
            default = f" (默认: {row[3]})" if row[3] else ""
            print(f"  - {row[0]}: {row[1]} ({nullable}){default}")
        
        # 查看数据数量
        count_result = db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        count = count_result.scalar()
        print(f"\n📊 数据数量: {count} 条记录")
        
        # 显示前5条数据
        if count > 0:
            print(f"\n📄 前5条数据:")
            data_result = db.execute(text(f"SELECT * FROM {table_name} LIMIT 5"))
            
            # 获取列名
            columns = [desc[0] for desc in data_result.description]
            print(f"  列名: {' | '.join(columns)}")
            print("  " + "-" * (len(' | '.join(columns)) + 10))
            
            for row in data_result:
                # 格式化每一行数据，截断长文本
                formatted_row = []
                for item in row:
                    if isinstance(item, str) and len(item) > 30:
                        formatted_row.append(item[:30] + "...")
                    else:
                        formatted_row.append(str(item))
                print(f"  {' | '.join(formatted_row)}")
        
    except Exception as e:
        print(f"❌ 查看表 {table_name} 失败: {e}")

def show_user_conversations(db):
    """显示用户对话统计"""
    print(f"\n💬 用户对话统计")
    print("=" * 50)
    
    try:
        # 统计每个用户的对话数量
        result = db.execute(text("""
            SELECT u.email, COUNT(c.id) as conversation_count
            FROM users u
            LEFT JOIN conversations c ON u.id = c.user_id
            GROUP BY u.id, u.email
            ORDER BY conversation_count DESC
        """))
        
        print("👥 用户对话统计:")
        for row in result:
            print(f"  - {row[0]}: {row[1]} 条对话")
            
        # 统计聊天会话
        session_result = db.execute(text("""
            SELECT u.email, COUNT(cs.id) as session_count, COUNT(m.id) as message_count
            FROM users u
            LEFT JOIN chat_sessions cs ON u.id = cs.user_id
            LEFT JOIN messages m ON cs.id = m.chat_session_id
            GROUP BY u.id, u.email
            ORDER BY session_count DESC
        """))
        
        print("\n💭 用户会话统计:")
        for row in session_result:
            print(f"  - {row[0]}: {row[1]} 个会话, {row[2]} 条消息")
            
    except Exception as e:
        print(f"❌ 查看用户对话统计失败: {e}")

def main():
    """主函数"""
    print("🔍 查看聊天机器人数据库内容")
    print("=" * 60)
    
    # 连接数据库
    engine, db = connect_database()
    if not db:
        return
    
    try:
        # 查看所有表
        tables_result = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """))
        
        tables = [row[0] for row in tables_result]
        print(f"📚 数据库中的表: {', '.join(tables)}")
        
        # 显示每个表的信息
        for table in tables:
            show_table_info(db, table)
        
        # 显示用户对话统计
        show_user_conversations(db)
        
        print(f"\n✅ 数据库查看完成!")
        
    except Exception as e:
        print(f"❌ 查看数据库失败: {e}")
    
    finally:
        db.close()

if __name__ == "__main__":
    main()