#!/usr/bin/env python3
# 测试长消息传输

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from app.chatbot_pipeline import ChatbotPipeline, StreamEventType, StreamEvent
from app.database import SessionLocal
from app.models import User

async def test_long_message():
    """测试长消息的生成和传输"""
    print("🧪 测试长消息传输...")
    
    # 创建一个模拟的长回复
    long_response = """根据印度农业部及世界银行最新数据（2023年）：

🌾 **印度耕地总面积**：约 **1.56亿公顷**（全球第一），占全球耕地面积的 **11%** 左右。

📊 **核心特点**：
- **人均耕地**：约 **0.12公顷**（1.8亩），略高于中国但低于世界均值。
- **农业依赖度**：超 **50%** 人口从事农业，贡献GDP的 **~20%**。
- **主产作物**：水稻、小麦、棉花、甘蔗（全球第二大水稻产国）。

⚠️ **挑战**：
- 土地碎片化（平均农场规模仅 **1.1公顷**）
- 灌溉设施不足（60%耕地依赖季风）

🌍 **全球对比**：
- **总量**：印度（1.56亿公顷）> 美国（1.52亿）> 中国（1.28亿）
- **单产**：中国 > 美国 > 印度（受技术/气候影响）

需要细分数据（如邦级分布）可随时问！ 🌾

*数据来源：印度农业部、FAO*

这是一个完整的长回复，用于测试消息传输是否会被截断。如果你能看到这句话，说明长消息传输是正常的。"""

    print(f"测试消息长度: {len(long_response)} 字符")
    
    # 模拟流式传输
    print("\n📡 模拟流式传输:")
    accumulated_content = ""
    
    for i, char in enumerate(long_response):
        accumulated_content += char
        if i % 50 == 0:  # 每50个字符打印一次进度
            print(f"已传输: {i+1}/{len(long_response)} 字符")
    
    print(f"\n✅ 完整消息长度: {len(accumulated_content)} 字符")
    print(f"消息开头: {accumulated_content[:100]}...")
    print(f"消息结尾: ...{accumulated_content[-100:]}")
    
    # 检查是否完整
    if len(accumulated_content) == len(long_response):
        print("🎉 消息传输完整！")
        return True
    else:
        print("❌ 消息传输不完整！")
        return False

if __name__ == "__main__":
    asyncio.run(test_long_message())