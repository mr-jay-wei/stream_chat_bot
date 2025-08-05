# app/database.py

import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship
from datetime import datetime
from typing import Generator

from .logger_config import get_logger

logger = get_logger(__name__)

class Base(DeclarativeBase):
    pass

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

# 创建同步引擎
from sqlalchemy import create_engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # 设为True可以看到SQL语句
    pool_size=10,
    max_overflow=20
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
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表初始化完成")
        
        # 检查新表是否创建成功
        from sqlalchemy import text
        db = SessionLocal()
        try:
            # 检查chat_sessions表
            result = db.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'chat_sessions'"))
            if result.scalar() > 0:
                logger.info("✅ chat_sessions表已创建")
            
            # 检查messages表
            result = db.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'messages'"))
            if result.scalar() > 0:
                logger.info("✅ messages表已创建")
                
        except Exception as e:
            logger.warning(f"检查新表时出错: {e}")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise