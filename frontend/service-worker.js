// Service Worker - PWA的核心

const CACHE_NAME = 'ai-jay-chat-cache-v1';
// 定义应用核心文件的列表，以便缓存它们
const urlsToCache = [
  '/',
  '/frontend/index.html',
  '/frontend/style.css',
  '/frontend/main.js'
];

// 1. 安装事件：当Service Worker首次被浏览器安装时触发
self.addEventListener('install', event => {
  console.log('Service Worker: 正在安装...');
  // 等待缓存操作完成后再完成安装
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Service Worker: 已打开缓存, 正在缓存核心文件...');
        return cache.addAll(urlsToCache);
      })
  );
});

// 2. 激活事件：当Service Worker被激活时触发，常用于清理旧缓存
self.addEventListener('activate', event => {
  console.log('Service Worker: 正在激活...');
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            console.log('Service Worker: 正在删除旧缓存', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  return self.clients.claim();
});

// 3. 抓取事件：当应用发起任何网络请求时触发
self.addEventListener('fetch', event => {
  // 我们只处理GET请求
  if (event.request.method !== 'GET') {
    return;
  }
  
  // API和WebSocket请求总是直接访问网络，不通过缓存
  if (event.request.url.includes('/api/') || event.request.url.includes('/ws')) {
    event.respondWith(fetch(event.request));
    return;
  }

  // 对于其他请求（如HTML, CSS, JS），我们采用"先缓存后网络"策略
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // 如果在缓存中找到了匹配的响应，则直接返回它
        if (response) {
          return response;
        }
        // 否则，从网络获取
        return fetch(event.request);
      })
  );
});