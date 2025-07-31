# app/logger_config.py

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime, timedelta
import glob


class LoggerConfig:
    """日志配置管理器"""
    
    def __init__(self, log_dir: str = None, max_days: int = None):
        """
        初始化日志配置
        
        Args:
            log_dir: 日志目录，None时从config读取
            max_days: 保留日志的最大天数，None时从config读取
        """
        # 延迟导入避免循环依赖
        try:
            from . import config
            self.log_dir = Path(log_dir or config.LOG_DIR)
            self.max_days = max_days or config.LOG_RETENTION_DAYS
            self.enable_console = config.ENABLE_CONSOLE_LOG
            self.enable_file = config.ENABLE_FILE_LOG
            self.enable_error = config.ENABLE_ERROR_LOG
            self.log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
        except ImportError:
            # 如果无法导入config，使用默认值
            self.log_dir = Path(log_dir or "log")
            self.max_days = max_days or 30
            self.enable_console = True
            self.enable_file = True
            self.enable_error = True
            self.log_level = logging.INFO
        
        self.log_dir.mkdir(exist_ok=True)
        
        # 清理旧日志
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """清理超过保留期的日志文件"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_days)
            
            # 查找所有日志文件
            log_files = glob.glob(str(self.log_dir / "*.log"))
            
            for log_file in log_files:
                file_path = Path(log_file)
                # 获取文件修改时间
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_mtime < cutoff_date:
                    try:
                        file_path.unlink()
                        print(f"🗑️ 已删除过期日志文件: {file_path.name}")
                    except Exception as e:
                        print(f"❌ 删除日志文件失败 {file_path.name}: {e}")
                        
        except Exception as e:
            print(f"❌ 清理旧日志失败: {e}")
    
    def setup_logger(self, name: str = None, level: int = None) -> logging.Logger:
        """
        设置并返回配置好的日志器
        
        Args:
            name: 日志器名称，默认为调用模块名
            level: 日志级别，None时使用配置文件中的级别
            
        Returns:
            配置好的日志器
        """
        if name is None:
            name = __name__
        
        if level is None:
            level = self.log_level
        
        logger = logging.getLogger(name)
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
        
        logger.setLevel(level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 1. 控制台处理器（根据配置决定是否启用）
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 2. 文件处理器 - 按日期轮转（根据配置决定是否启用）
        if self.enable_file:
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"chatbot_{today}.log"
            
            file_handler = logging.FileHandler(
                log_file, 
                mode='a', 
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # 3. 错误日志单独文件（根据配置决定是否启用）
        if self.enable_error:
            today = datetime.now().strftime("%Y-%m-%d")
            error_log_file = self.log_dir / f"error_{today}.log"
            error_handler = logging.FileHandler(
                error_log_file,
                mode='a',
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)
        
        # 防止日志向上传播到根日志器
        logger.propagate = False
        
        return logger
    
    def get_log_stats(self) -> dict:
        """
        获取日志统计信息
        
        Returns:
            日志统计信息字典
        """
        try:
            log_files = list(self.log_dir.glob("*.log"))
            total_size = sum(f.stat().st_size for f in log_files)
            
            return {
                "log_directory": str(self.log_dir),
                "total_files": len(log_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "retention_days": self.max_days,
                "files": [
                    {
                        "name": f.name,
                        "size_kb": round(f.stat().st_size / 1024, 2),
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    }
                    for f in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)
                ]
            }
        except Exception as e:
            return {"error": f"获取日志统计失败: {e}"}


# 创建全局日志配置实例
logger_config = LoggerConfig()

# 便捷函数
def get_logger(name: str = None) -> logging.Logger:
    """获取配置好的日志器"""
    return logger_config.setup_logger(name)