# chatbot_core/config.py

from typing import Dict, Any

# --- 模型配置 ---
# (此部分配置从.env文件中读取)

# --- 核心对话配置 ---
# 定义机器人默认使用的系统角色提示词文件名 (不含.txt)
SYSTEM_PROMPT_NAME: str = "assistant_prompt"

# --- 短期记忆配置 ---
ENABLE_SHORT_TERM_MEMORY: bool = True
SHORT_TERM_MEMORY_MAX_LENGTH: int = 100_000 # 最大字符长度
SINGLE_CONVERSATION_MAX_LENGTH: int = 20_000 # 单轮对话最大长度
MIN_CONVERSATION_ROUNDS: int = 1 # 最小保留轮数
MEMORY_CLEANUP_STRATEGY: str = "auto" # "auto"或"sliding_window"
SLIDING_WINDOW_SIZE: int = 20 # 滑动窗口大小时使用

# --- 提示词热重载配置 ---
ENABLE_HOT_RELOAD: bool = True
HOT_RELOAD_DEBOUNCE_TIME: float = 0.5 # 防抖时间（秒）