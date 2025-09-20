import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // 监听所有网络接口，方便手机等设备访问
    host: '0.0.0.0', 
    port: 5173, // 你可以指定一个端口
    proxy: {
      // 代理规则：所有/api和/ws的请求，都转发到后端服务器
      '/api': {
        target: 'http://localhost:28501', // 这是你的Python后端地址
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:28501', // WebSocket也需要代理
        ws: true,
      },
    },
  },
})