# app/limiter.py

import os
import redis.asyncio as redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
load_dotenv()
from .logger_config import get_logger
logger = get_logger(__name__)

# --- Redis 连接 ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# 构建 Redis URL
redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
logger.info(f"速率限制器正在连接到 Redis: {REDIS_HOST}:{REDIS_PORT}")

# --- 初始化 Limiter ---
# 使用异步 redis 客户端
limiter = Limiter(
    key_func=get_remote_address,  # 使用客户端的IP地址作为唯一标识
    storage_uri=redis_url,
    strategy="fixed-window",  # 固定时间窗口算法
    storage_options={"socket_connect_timeout": 3}
)