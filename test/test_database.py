# test/test_database.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from app.database import init_database, get_db, engine

async def test_database_connection():
    """测试数据库连接"""
    try:
        print("正在测试数据库连接...")
        
        # 测试数据库初始化
        await init_database()
        print("✅ 数据库初始化成功")
        
        # 测试数据库会话
        async for db in get_db():
            print("✅ 数据库会话创建成功")
            break
        
        print("✅ 数据库连接测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_database_connection())
    if success:
        print("数据库配置正确，可以继续运行应用")
    else:
        print("请检查数据库配置和连接")