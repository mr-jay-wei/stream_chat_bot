#!/usr/bin/env python3
# log_manager.py

"""
日志管理工具
提供日志查看、清理、统计等功能
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# 添加app目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from app.logger_config import logger_config


def show_log_stats():
    """显示日志统计信息"""
    stats = logger_config.get_log_stats()
    
    if "error" in stats:
        print(f"❌ {stats['error']}")
        return
    
    print("📊 日志统计信息")
    print("=" * 50)
    print(f"日志目录: {stats['log_directory']}")
    print(f"文件总数: {stats['total_files']}")
    print(f"总大小: {stats['total_size_mb']} MB")
    print(f"保留天数: {stats['retention_days']} 天")
    print()
    
    if stats['files']:
        print("📁 日志文件列表:")
        for file_info in stats['files']:
            print(f"  {file_info['name']} - {file_info['size_kb']} KB - {file_info['modified']}")
    else:
        print("📁 暂无日志文件")


def view_log(log_type="chatbot", date=None, lines=50):
    """查看日志内容"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    log_file = logger_config.log_dir / f"{log_type}_{date}.log"
    
    if not log_file.exists():
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    print(f"📖 查看日志: {log_file}")
    print("=" * 50)
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            
        # 显示最后N行
        display_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        for line in display_lines:
            print(line.rstrip())
            
        if len(all_lines) > lines:
            print(f"\n... (显示最后 {lines} 行，共 {len(all_lines)} 行)")
            
    except Exception as e:
        print(f"❌ 读取日志文件失败: {e}")


def cleanup_old_logs(days=None):
    """清理旧日志"""
    if days is None:
        days = logger_config.max_days
    
    print(f"🧹 清理 {days} 天前的日志文件...")
    
    cutoff_date = datetime.now() - timedelta(days=days)
    cleaned_count = 0
    
    for log_file in logger_config.log_dir.glob("*.log"):
        try:
            file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if file_mtime < cutoff_date:
                log_file.unlink()
                print(f"  ✅ 已删除: {log_file.name}")
                cleaned_count += 1
        except Exception as e:
            print(f"  ❌ 删除失败 {log_file.name}: {e}")
    
    print(f"🎉 清理完成，共删除 {cleaned_count} 个文件")


def main():
    parser = argparse.ArgumentParser(description="日志管理工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # stats命令
    subparsers.add_parser("stats", help="显示日志统计信息")
    
    # view命令
    view_parser = subparsers.add_parser("view", help="查看日志内容")
    view_parser.add_argument("--type", default="chatbot", help="日志类型 (chatbot/error)")
    view_parser.add_argument("--date", help="日期 (YYYY-MM-DD)")
    view_parser.add_argument("--lines", type=int, default=50, help="显示行数")
    
    # cleanup命令
    cleanup_parser = subparsers.add_parser("cleanup", help="清理旧日志")
    cleanup_parser.add_argument("--days", type=int, help="保留天数")
    
    args = parser.parse_args()
    
    if args.command == "stats":
        show_log_stats()
    elif args.command == "view":
        view_log(args.type, args.date, args.lines)
    elif args.command == "cleanup":
        cleanup_old_logs(args.days)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()