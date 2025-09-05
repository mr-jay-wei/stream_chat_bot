# app/database.py

import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship
from datetime import datetime
from typing import Generator
import sqlite3

from .logger_config import get_logger

logger = get_logger(__name__)

class Base(DeclarativeBase):
    pass
'''
# 数据库配置
DATABASE_CONFIG = {
    "user": "postgres",
    "password": "052756", 
    "host": "127.0.0.1",
    "port": "5432",
    "database": "mydb"
}

# 构建数据库URL（使用同步驱动）
DATABASE_URL = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
'''
DATABASE_URL = "sqlite:///stream_chat_bot.db"
# 创建同步引擎

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  
    echo=False  # 设为True可以看到SQL语句
)

# 创建同步会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator:
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"数据库会话错误: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def init_database():
    """初始化数据库表"""
    try:
        # 检查数据库文件是否存在
        db_file = 'stream_chat_bot.db'
        is_new_db = not os.path.exists(db_file)
        
        Base.metadata.create_all(bind=engine)
        if is_new_db:
            logger.info(f"新的SQLite数据库文件 '{db_file}' 已创建并初始化。")
        else:
            logger.info("数据库表初始化检查完成。")

        # 检查新表是否创建成功
        from sqlalchemy import text
        db = SessionLocal()
        try:
            # 检查chat_sessions表
            result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_sessions'"))
            if result.fetchone():
                logger.info("✅ chat_sessions表已存在")
            
            # 检查messages表
            result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"))
            if result.fetchone():
                logger.info("✅ messages表已存在")
                
        except Exception as e:
            logger.warning(f"检查新表时出错: {e}")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise