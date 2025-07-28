# rag/hot_reload_manager.py

import os
import time
import threading
from pathlib import Path
from typing import Dict, Set, Optional, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

from .prompt_manager import prompt_manager
from . import config


class PromptFileHandler(FileSystemEventHandler):
    """提示词文件变化处理器"""
    
    def __init__(self, callback: Optional[Callable[[str, str], None]] = None):
        """
        初始化文件处理器
        
        Args:
            callback: 文件变化时的回调函数，参数为(event_type, prompt_name)
        """
        super().__init__()
        self.callback = callback
        self.last_modified: Dict[str, float] = {}
        self.debounce_time = config.HOT_RELOAD_DEBOUNCE_TIME  # 防抖时间（秒）
        
    def _should_process_event(self, file_path: str) -> bool:
        """
        判断是否应该处理该事件（防抖处理）
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否应该处理
        """
        current_time = time.time()
        last_time = self.last_modified.get(file_path, 0)
        
        if current_time - last_time < self.debounce_time:
            return False
        
        self.last_modified[file_path] = current_time
        return True
    
    def _get_prompt_name(self, file_path: str) -> Optional[str]:
        """
        从文件路径获取提示词名称
        
        Args:
            file_path: 文件路径
            
        Returns:
            提示词名称，如果不是提示词文件则返回None
        """
        path = Path(file_path)
        
        # 检查是否是提示词文件
        if (path.suffix == '.txt' and 
            'prompts' in str(path) and 
            path.parent.name == 'prompts'):
            return path.stem
        
        return None
    
    def on_modified(self, event):
        """文件修改事件处理"""
        if event.is_directory:
            return
        
        prompt_name = self._get_prompt_name(event.src_path)
        if not prompt_name:
            return
        
        if not self._should_process_event(event.src_path):
            return
        
        try:
            print(f"🔄 检测到提示词文件修改: {prompt_name}")
            
            # 清除所有相关缓存
            prompt_manager._prompt_cache.pop(prompt_name, None)
            prompt_manager._template_cache.pop(prompt_name, None)
            
            # 重新加载提示词（这会重新填充缓存）
            prompt_manager.load_prompt(prompt_name)
            print(f"✅ 自动重载完成: {prompt_name}")
            
            # 调用回调函数
            if self.callback:
                self.callback("modified", prompt_name)
                
        except Exception as e:
            print(f"❌ 自动重载失败 {prompt_name}: {e}")
    
    def on_created(self, event):
        """文件创建事件处理"""
        if event.is_directory:
            return
        
        prompt_name = self._get_prompt_name(event.src_path)
        if not prompt_name:
            return
        
        try:
            print(f"➕ 检测到新提示词文件: {prompt_name}")
            
            # 加载新提示词
            prompt_manager.load_prompt(prompt_name)
            print(f"✅ 自动加载完成: {prompt_name}")
            
            # 调用回调函数
            if self.callback:
                self.callback("created", prompt_name)
                
        except Exception as e:
            print(f"❌ 自动加载失败 {prompt_name}: {e}")
    
    def on_deleted(self, event):
        """文件删除事件处理"""
        if event.is_directory:
            return
        
        prompt_name = self._get_prompt_name(event.src_path)
        if not prompt_name:
            return
        
        try:
            print(f"🗑️ 检测到提示词文件删除: {prompt_name}")
            
            # 从缓存中移除
            prompt_manager._prompt_cache.pop(prompt_name, None)
            prompt_manager._template_cache.pop(prompt_name, None)
            print(f"✅ 缓存清理完成: {prompt_name}")
            
            # 调用回调函数
            if self.callback:
                self.callback("deleted", prompt_name)
                
        except Exception as e:
            print(f"❌ 缓存清理失败 {prompt_name}: {e}")


class HotReloadManager:
    """热重载管理器"""
    
    def __init__(self, enable_hot_reload: bool = True):
        """
        初始化热重载管理器
        
        Args:
            enable_hot_reload: 是否启用热重载功能
        """
        self.enable_hot_reload = enable_hot_reload
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[PromptFileHandler] = None
        self.is_running = False
        self.callbacks: Set[Callable[[str, str], None]] = set()
        
        # 监控的目录
        self.watch_directory = prompt_manager.prompts_dir
        
        if self.enable_hot_reload:
            self._setup_file_watcher()
    
    def _setup_file_watcher(self):
        """设置文件监控器"""
        try:
            # 确保监控目录存在
            self.watch_directory.mkdir(exist_ok=True)
            
            # 创建事件处理器
            self.event_handler = PromptFileHandler(callback=self._on_file_change)
            
            # 创建观察者
            self.observer = Observer()
            self.observer.schedule(
                self.event_handler,
                str(self.watch_directory),
                recursive=False
            )
            
            print(f"🔍 热重载监控已设置，监控目录: {self.watch_directory}")
            
        except Exception as e:
            print(f"❌ 设置文件监控器失败: {e}")
            self.enable_hot_reload = False
    
    def _on_file_change(self, event_type: str, prompt_name: str):
        """文件变化回调处理"""
        # 通知所有注册的回调函数
        for callback in self.callbacks:
            try:
                callback(event_type, prompt_name)
            except Exception as e:
                print(f"❌ 回调函数执行失败: {e}")
    
    def start(self):
        """启动热重载监控"""
        if not self.enable_hot_reload:
            print("⚠️ 热重载功能未启用")
            return False
        
        if self.is_running:
            print("⚠️ 热重载监控已在运行中")
            return True
        
        # 如果observer已经停止，需要重新创建
        if self.observer and not self.observer.is_alive():
            self._setup_file_watcher()
        
        if not self.observer:
            print("❌ 文件监控器初始化失败")
            return False
        
        try:
            self.observer.start()
            self.is_running = True
            print(f"🔥 热重载监控已启动，正在监控: {self.watch_directory}")
            return True
            
        except Exception as e:
            print(f"❌ 启动热重载监控失败: {e}")
            # 尝试重新创建observer
            self._setup_file_watcher()
            if self.observer:
                try:
                    self.observer.start()
                    self.is_running = True
                    print(f"🔥 热重载监控已重新启动，正在监控: {self.watch_directory}")
                    return True
                except Exception as e2:
                    print(f"❌ 重新启动也失败: {e2}")
            return False
    
    def stop(self):
        """停止热重载监控"""
        if not self.observer or not self.is_running:
            return
        
        try:
            self.observer.stop()
            self.observer.join(timeout=5)  # 等待最多5秒
            self.is_running = False
            print("🛑 热重载监控已停止")
            
        except Exception as e:
            print(f"❌ 停止热重载监控失败: {e}")
    
    def add_callback(self, callback: Callable[[str, str], None]):
        """
        添加文件变化回调函数
        
        Args:
            callback: 回调函数，参数为(event_type, prompt_name)
        """
        self.callbacks.add(callback)
        print(f"📝 已添加热重载回调函数")
    
    def remove_callback(self, callback: Callable[[str, str], None]):
        """
        移除文件变化回调函数
        
        Args:
            callback: 要移除的回调函数
        """
        self.callbacks.discard(callback)
        print(f"🗑️ 已移除热重载回调函数")
    
    def get_status(self) -> Dict[str, any]:
        """
        获取热重载状态信息
        
        Returns:
            状态信息字典
        """
        return {
            "enabled": self.enable_hot_reload,
            "running": self.is_running,
            "watch_directory": str(self.watch_directory),
            "callbacks_count": len(self.callbacks),
            "observer_alive": self.observer.is_alive() if self.observer else False
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


# 检查是否安装了watchdog库
try:
    import watchdog
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️ 未安装watchdog库，热重载功能不可用")
    print("   安装命令: uv add watchdog")


# 创建全局热重载管理器实例
hot_reload_manager = HotReloadManager(
    enable_hot_reload=WATCHDOG_AVAILABLE and getattr(config, 'ENABLE_HOT_RELOAD', True)
) if WATCHDOG_AVAILABLE else None


def enable_hot_reload():
    """启用热重载功能"""
    if not WATCHDOG_AVAILABLE:
        print("❌ watchdog库未安装，无法启用热重载功能")
        print("   安装命令: uv add watchdog")
        return False
    
    if hot_reload_manager:
        return hot_reload_manager.start()
    return False


def disable_hot_reload():
    """禁用热重载功能"""
    if hot_reload_manager:
        hot_reload_manager.stop()


def is_hot_reload_enabled() -> bool:
    """检查热重载是否启用"""
    return (hot_reload_manager is not None and 
            hot_reload_manager.is_running if hot_reload_manager else False)


def get_hot_reload_status() -> Dict[str, any]:
    """获取热重载状态"""
    if hot_reload_manager:
        return hot_reload_manager.get_status()
    else:
        return {
            "enabled": False,
            "running": False,
            "error": "watchdog库未安装" if not WATCHDOG_AVAILABLE else "热重载管理器未初始化"
        }