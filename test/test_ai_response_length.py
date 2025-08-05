#!/usr/bin/env python3
# 测试AI回复长度和完整性

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from app.chatbot_pipeline import ChatbotPipeline
from app.database import SessionLocal
from app.models import User

async def test_ai_response_length():
    """测试AI回复的长度和完整性"""
    print("🧪 测试AI回复长度和完整性...")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        # 获取测试用户
        user = db.query(User).filter(User.email == "xiaobai@123.com").first()
        if not user:
            print("❌ 未找到测试用户")
            return False
        
        print(f"✅ 使用测试用户: {user.email} (ID: {user.id})")
        
        # 创建chatbot实例
        pipeline = ChatbotPipeline()
        
        # 测试一个需要长回复的问题
        test_question = "请详细介绍Python编程语言的特点、应用场景、学习路径，以及与其他编程语言的对比，包括具体的代码示例"
        
        print(f"\n📝 测试问题: {test_question}")
        print(f"问题长度: {len(test_question)} 字符")
        
        # 收集完整的AI回复
        complete_response = ""
        chunk_count = 0
        
        print(f"\n🤖 AI回复流式输出:")
        print("-" * 60)
        
        async for event in pipeline.ask_stream(test_question, db, user.id):
            if event.type.value == "generation_chunk":
                chunk = event.data.get("chunk", "")
                complete_response += chunk
                chunk_count += 1
                
                # 每100个字符显示一次进度
                if len(complete_response) % 100 == 0:
                    print(f"已接收: {len(complete_response)} 字符 (第{chunk_count}个chunk)")
            
            elif event.type.value == "generation_end":
                print(f"\n✅ 生成完成!")
                break
            
            elif event.type.value == "error":
                print(f"❌ 错误: {event.data.get('error')}")
                return False
        
        print(f"\n📊 回复统计:")
        print(f"  总长度: {len(complete_response)} 字符")
        print(f"  总chunks: {chunk_count}")
        print(f"  平均chunk长度: {len(complete_response)/chunk_count:.1f} 字符" if chunk_count > 0 else "  无chunks")
        
        print(f"\n📄 回复内容预览:")
        print(f"开头: {complete_response[:200]}...")
        print(f"结尾: ...{complete_response[-200:]}")
        
        # 检查回复是否被截断
        truncation_indicators = [
            "...[内容过长已截断]",
            "[truncated",
            "内容被截断",
            "回复不完整"
        ]
        
        is_truncated = any(indicator in complete_response for indicator in truncation_indicators)
        
        if is_truncated:
            print("⚠️  检测到回复可能被截断")
        else:
            print("✅ 回复看起来是完整的")
        
        # 检查回复长度是否合理
        if len(complete_response) < 100:
            print("⚠️  回复长度过短，可能有问题")
            return False
        elif len(complete_response) > 3000:
            print("✅ 回复长度充足，说明长文本处理正常")
        else:
            print("✅ 回复长度正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    success = asyncio.run(test_ai_response_length())
    if success:
        print("\n🎉 AI回复长度测试完成！")
    else:
        print("\n❌ AI回复长度测试失败！")