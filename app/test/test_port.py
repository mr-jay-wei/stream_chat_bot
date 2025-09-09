# test_port.py
import socket

def check_port(port):
    """检查指定端口是否可以被绑定"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # 尝试绑定到 0.0.0.0 和指定端口
            s.bind(("0.0.0.0", port))
            print(f"✅ 端口 {port} 可用，绑定成功！")
            return True
    except OSError as e:
        # 捕获权限错误或端口被占用错误
        print(f"❌ 端口 {port} 不可用: {e}")
        return False

if __name__ == "__main__":
    print("--- 正在测试端口可用性 ---")
    
    # 测试我们之前失败的端口
    print("\n测试旧端口:")
    check_port(8003)
    check_port(8501)
    
    # 测试一个肯定安全的高位端口
    print("\n测试推荐的高位端口:")
    check_port(28501)
    
    print("\n--- 测试完成 ---")