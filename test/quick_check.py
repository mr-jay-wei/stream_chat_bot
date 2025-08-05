#!/usr/bin/env python3
# 快速检查今天的对话记录

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import Conversation, User
from sqlalchemy import func, text
from datetime import datetime, date

def quick_check():
    """快速检查今天的对话记录"""
    db = SessionLocal()
    try:
        print("🔍 快速检查今天的对话记录...")
        print("=" * 50)
        
        # 检查今天的对话记录数
        today = date.today()
        today_conversations = db.query(Conversation).filter(
            func.date(Conversation.created_at) == today
        ).all()
        
        print(f"📅 今天的对话记录数: {len(today_conversations)}")
        
        if today_conversations:
            print(f"\n📄 今天的对话记录:")
            for conv in today_conversations:
                user = db.query(User).filter(User.id == conv.user_id).first()
                user_email = user.email if user else f"用户ID:{conv.user_id}"
                time_str = conv.created_at.strftime("%H:%M:%S")
                question_preview = conv.question[:20] + "..." if len(conv.question) > 20 else conv.question
                print(f"  - {user_email}: {question_preview} ({time_str})")
        
        # 检查总对话记录数
        total_count = db.query(Conversation).count()
        print(f"\n📊 总对话记录数: {total_count}")
        
        # 检查最新的5条记录
        latest_conversations = db.query(Conversation).order_by(
            Conversation.created_at.desc()
        ).limit(5).all()
        
        print(f"\n📄 最新的5条对话记录:")
        for conv in latest_conversations:
            user = db.query(User).filter(User.id == conv.user_id).first()
            user_email = user.email if user else f"用户ID:{conv.user_id}"
            time_str = conv.created_at.strftime("%Y-%m-%d %H:%M:%S")
            question_preview = conv.question[:30] + "..." if len(conv.question) > 30 else conv.question
            print(f"  - {user_email}: {question_preview} ({time_str})")
            
    except Exception as e:
        print(f"❌ 检查失败: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    quick_check()