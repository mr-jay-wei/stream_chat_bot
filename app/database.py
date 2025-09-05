# app/database.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from typing import Generator
from dotenv import load_dotenv
load_dotenv()
from .logger_config import get_logger
logger = get_logger(__name__)

class Base(DeclarativeBase):
    pass
# --- 从环境变量读取数据库配置 ---
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "mydb")

# --- 生产环境 PostgreSQL 配置 ---
# 使用 psycopg2 驱动
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- 用于本地开发的 SQLite 配置 (备用) ---
# 如果需要切换回SQLite，只需取消注释下面这行，并注释掉上面的PostgreSQL配置，init_database也要改回上面的init_database函数
# DATABASE_URL = "sqlite:///stream_chat_bot.db"

# 创建引擎
try:
    if DATABASE_URL.startswith("sqlite"):
        # SQLite 的特定配置
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            echo=False
        )
        logger.info("正在使用 SQLite 数据库 (用于本地开发)")
    else:
        # PostgreSQL 的配置
        engine = create_engine(
            DATABASE_URL,
            echo=False,
            pool_size=10,
            max_overflow=20
        )
        logger.info(f"正在连接 PostgreSQL 数据库: {DB_HOST}:{DB_PORT}/{DB_NAME}")

except Exception as e:
    logger.error(f"创建数据库引擎失败: {e}")
    raise

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

# def init_database():
#     """初始化数据库表"""
#     try:
#         # 检查数据库文件是否存在
#         db_file = 'stream_chat_bot.db'
#         is_new_db = not os.path.exists(db_file)
        
#         Base.metadata.create_all(bind=engine)
#         if is_new_db:
#             logger.info(f"新的SQLite数据库文件 '{db_file}' 已创建并初始化。")
#         else:
#             logger.info("数据库表初始化检查完成。")

#         # 检查新表是否创建成功
#         from sqlalchemy import text
#         db = SessionLocal()
#         try:
#             # 检查chat_sessions表
#             result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_sessions'"))
#             if result.fetchone():
#                 logger.info("✅ chat_sessions表已存在")
            
#             # 检查messages表
#             result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"))
#             if result.fetchone():
#                 logger.info("✅ messages表已存在")
                
#         except Exception as e:
#             logger.warning(f"检查新表时出错: {e}")
#         finally:
#             db.close()
            
#     except Exception as e:
#         logger.error(f"数据库初始化失败: {e}")
#         raise

def init_database():
    """初始化数据库表"""
    try:
        logger.info("正在初始化/检查数据库表...")
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表初始化完成。")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise