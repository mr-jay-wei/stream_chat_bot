// mobile/api/index.js
import axios from 'axios';

// 你的电脑在局域网中的IP地址。这是让手机能访问到电脑上运行的后端服务的关键。
// 你需要将 'YOUR_COMPUTER_IP' 替换成你自己的IP地址。
const YOUR_COMPUTER_IP = '192.168.31.134'; // 例如: '172.20.10.2'

const API_BASE_URL = `http://${YOUR_COMPUTER_IP}:28501/api`;

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;