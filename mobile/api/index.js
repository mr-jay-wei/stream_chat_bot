// mobile/api/index.js
import axios from 'axios';

// 你的电脑在局域网中的IP地址。这是让手机能访问到电脑上运行的后端服务的关键。
// 你需要将 'YOUR_COMPUTER_IP' 替换成你自己的IP地址。
const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL;

if (!API_BASE_URL) {
  alert('错误：API URL 未在环境变量中配置！请检查 mobile/.env 文件。');
}

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;