#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查用户数据库
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import DATABASE_CONFIG

def main():
    # 构建数据库URL
    DATABASE_URL = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
    
    # 创建引擎和会话
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    
    with SessionLocal() as session:
        print("🔍 检查用户数据...")
        print("=" * 50)
        
        # 查询所有用户
        result = session.execute(text("SELECT id, email, created_at FROM users ORDER BY id"))
        users = result.fetchall()
        
        print(f"📊 总用户数: {len(users)}")
        print("\n👥 用户列表:")
        for user in users:
            print(f"  - ID: {user.id}, Email: {user.email}, 注册时间: {user.created_at}")
        
        print("\n🔍 检查对话记录...")
        print("=" * 50)
        
        # 查询今天的对话记录
        result = session.execute(text("""
            SELECT user_id, question, answer, created_at 
            FROM conversations 
            WHERE DATE(created_at) = CURRENT_DATE 
            ORDER BY created_at DESC
        """))
        today_conversations = result.fetchall()
        
        print(f"📅 今天的对话记录数: {len(today_conversations)}")
        
        if today_conversations:
            print("\n📄 今天的对话记录:")
            for conv in today_conversations:
                print(f"  - 用户ID: {conv.user_id}, 问题: {conv.question[:50]}..., 时间: {conv.created_at}")
        else:
            print("❌ 今天没有对话记录")
        
        # 查询最新的5条对话记录
        result = session.execute(text("""
            SELECT c.user_id, u.email, c.question, c.created_at 
            FROM conversations c
            JOIN users u ON c.user_id = u.id
            ORDER BY c.created_at DESC
            LIMIT 5
        """))
        recent_conversations = result.fetchall()
        
        print(f"\n📄 最新的5条对话记录:")
        for conv in recent_conversations:
            print(f"  - 用户ID: {conv.user_id}, Email: {conv.email}, 问题: {conv.question[:50]}..., 时间: {conv.created_at}")

if __name__ == "__main__":
    main()