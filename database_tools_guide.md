# 数据库查看工具指南

## 🎯 方法1：使用Python脚本（推荐新手）

### 快速开始
```bash
python view_database.py
```

这个脚本提供了一个友好的菜单界面，可以：
- 📋 查看所有表的结构和记录数
- 👥 查看用户信息
- 💬 查看会话详情
- 💭 查看消息内容
- 📜 查看旧对话记录
- 👤 查看特定用户的所有活动
- 🔍 执行自定义SQL查询

## 🎯 方法2：使用pgAdmin（图形界面）

### 安装pgAdmin
1. 下载：https://www.pgadmin.org/download/
2. 安装后启动pgAdmin
3. 添加服务器连接：
   - Host: localhost
   - Port: 5432
   - Database: chatbot_db
   - Username: chatbot_user
   - Password: chatbot_password

### 使用pgAdmin
- 左侧树形结构浏览数据库
- 右键表名 → "View/Edit Data" → "All Rows" 查看数据
- 使用Query Tool执行SQL查询

## 🎯 方法3：使用命令行工具

### 连接到PostgreSQL
```bash
# Windows (如果安装了PostgreSQL)
psql -h localhost -p 5432 -U chatbot_user -d chatbot_db

# 输入密码: chatbot_password
```

### 常用SQL命令
```sql
-- 查看所有表
\dt

-- 查看表结构
\d users
\d chat_sessions
\d messages
\d conversations

-- 查看表数据
SELECT * FROM users;
SELECT * FROM chat_sessions LIMIT 10;
SELECT * FROM messages ORDER BY created_at DESC LIMIT 5;

-- 统计查询
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM messages;
SELECT user_id, COUNT(*) as message_count 
FROM messages 
GROUP BY user_id;

-- 退出
\q
```

## 🎯 方法4：使用DBeaver（免费图形工具）

### 安装DBeaver
1. 下载：https://dbeaver.io/download/
2. 安装后创建新连接
3. 选择PostgreSQL，输入连接信息

### 连接信息
- Server Host: localhost
- Port: 5432
- Database: chatbot_db
- Username: chatbot_user
- Password: chatbot_password

## 🎯 方法5：使用VS Code扩展

### 安装PostgreSQL扩展
1. 在VS Code中安装"PostgreSQL"扩展
2. 添加数据库连接
3. 直接在VS Code中查看和编辑数据

## 📊 常用查询示例

### 查看用户统计
```sql
SELECT 
    u.email,
    COUNT(DISTINCT cs.id) as session_count,
    COUNT(m.id) as message_count
FROM users u
LEFT JOIN chat_sessions cs ON u.id = cs.user_id
LEFT JOIN messages m ON cs.id = m.chat_session_id
GROUP BY u.id, u.email;
```

### 查看最活跃的会话
```sql
SELECT 
    cs.title,
    COUNT(m.id) as message_count,
    cs.created_at
FROM chat_sessions cs
LEFT JOIN messages m ON cs.id = m.chat_session_id
GROUP BY cs.id, cs.title, cs.created_at
ORDER BY message_count DESC
LIMIT 10;
```

### 查看今天的活动
```sql
SELECT 
    DATE(created_at) as date,
    COUNT(*) as message_count
FROM messages 
WHERE DATE(created_at) = CURRENT_DATE
GROUP BY DATE(created_at);
```

## 🔧 故障排除

### 连接问题
如果无法连接数据库：
1. 确认PostgreSQL服务正在运行
2. 检查防火墙设置
3. 验证用户名和密码
4. 确认数据库名称正确

### 权限问题
如果提示权限不足：
1. 确认用户有相应的表访问权限
2. 联系数据库管理员
3. 检查用户角色设置

## 💡 小贴士

1. **备份数据**：在进行任何修改前，先备份数据库
2. **只读查询**：新手建议只使用SELECT查询
3. **限制结果**：使用LIMIT避免查询过多数据
4. **索引优化**：了解表的索引结构可以提高查询效率
5. **定期维护**：定期清理和优化数据库