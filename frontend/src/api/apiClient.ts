import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api', // Vite会帮我们代理到后端
});

// 请求拦截器：在每次发送请求前，都检查一下有没有token，有就带上
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export default apiClient;