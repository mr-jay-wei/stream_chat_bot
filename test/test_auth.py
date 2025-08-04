# test/test_auth.py

import pytest
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import AsyncSessionLocal, init_database
from app.auth import create_user, authenticate_user, verify_password, get_password_hash
from app.user_service import UserService

@pytest.fixture
async def db_session():
    """创建测试数据库会话"""
    await init_database()
    async with AsyncSessionLocal() as session:
        yield session

@pytest.mark.asyncio
async def test_password_hashing():
    """测试密码哈希功能"""
    password = "test123456"
    hashed = get_password_hash(password)
    
    # 验证哈希后的密码不等于原密码
    assert hashed != password
    
    # 验证密码验证功能
    assert verify_password(password, hashed) == True
    assert verify_password("wrong_password", hashed) == False

@pytest.mark.asyncio
async def test_user_creation(db_session: AsyncSession):
    """测试用户创建功能"""
    email = "test@example.com"
    password = "test123456"
    
    # 创建用户
    user = await create_user(db_session, email, password)
    
    assert user.email == email
    assert user.is_active == True
    assert user.id is not None

@pytest.mark.asyncio
async def test_user_authentication(db_session: AsyncSession):
    """测试用户认证功能"""
    email = "auth_test@example.com"
    password = "test123456"
    
    # 先创建用户
    await create_user(db_session, email, password)
    
    # 测试正确的认证
    user = await authenticate_user(db_session, email, password)
    assert user is not None
    assert user.email == email
    
    # 测试错误的密码
    user = await authenticate_user(db_session, email, "wrong_password")
    assert user is None
    
    # 测试不存在的用户
    user = await authenticate_user(db_session, "nonexistent@example.com", password)
    assert user is None

@pytest.mark.asyncio
async def test_user_service(db_session: AsyncSession):
    """测试用户服务功能"""
    email = "service_test@example.com"
    password = "test123456"
    
    # 创建用户
    user = await create_user(db_session, email, password)
    
    # 测试根据邮箱获取用户
    found_user = await UserService.get_user_by_email(db_session, email)
    assert found_user is not None
    assert found_user.email == email
    
    # 测试根据ID获取用户
    found_user_by_id = await UserService.get_user_by_id(db_session, user.id)
    assert found_user_by_id is not None
    assert found_user_by_id.email == email
    
    # 测试添加对话记录
    question = "测试问题"
    answer = "测试回答"
    conversation = await UserService.add_conversation(db_session, user.id, question, answer)
    assert conversation is not None
    assert conversation.question == question
    assert conversation.answer == answer
    
    # 测试获取对话记录
    conversations = await UserService.get_user_conversations(db_session, user.id)
    assert len(conversations) == 1
    assert conversations[0].question == question
    assert conversations[0].answer == answer

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_password_hashing())
    print("密码哈希测试通过")
    
    async def run_db_tests():
        async with AsyncSessionLocal() as session:
            await test_user_creation(session)
            print("用户创建测试通过")
            
            await test_user_authentication(session)
            print("用户认证测试通过")
            
            await test_user_service(session)
            print("用户服务测试通过")
    
    asyncio.run(run_db_tests())
    print("所有测试通过！")