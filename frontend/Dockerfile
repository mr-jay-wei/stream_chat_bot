# --- Stage 1: Build ---
# 使用一个包含 Node.js 的官方镜像作为构建环境
FROM node:20-alpine AS build

# 设置工作目录
WORKDIR /app

# 复制 package.json 和 package-lock.json (如果存在)
COPY package*.json ./

# 安装项目依赖
RUN npm install

# 复制所有前端代码到容器中
COPY . .

# 执行构建命令，生成静态文件
RUN npm run build

# --- Stage 2: Serve ---
# 使用一个非常轻量的 Nginx 镜像作为最终的运行环境
FROM nginx:1.27-alpine

# 将构建阶段生成的静态文件复制到 Nginx 的网站根目录
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
# 暴露 80 端口
EXPOSE 80

# 当容器启动时，运行 Nginx 服务
CMD ["nginx", "-g", "daemon off;"]