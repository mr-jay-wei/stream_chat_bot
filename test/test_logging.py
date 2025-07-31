#!/usr/bin/env python3
# test_logging.py

"""
测试日志系统
"""

import sys
from pathlib import Path

# 添加app目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from app.logger_config import get_logger

def test_logging():
    """测试日志功能"""
    logger = get_logger("test_module")
    
    logger.info("🧪 开始测试日志系统...")
    logger.debug("这是一条调试信息")
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    
    logger.info("✅ 日志系统测试完成")
    
    # 显示日志统计
    from app.logger_config import logger_config
    stats = logger_config.get_log_stats()
    print(f"\n📊 测试后日志统计:")
    print(f"文件总数: {stats['total_files']}")
    print(f"总大小: {stats['total_size_mb']} MB")

if __name__ == "__main__":
    test_logging()