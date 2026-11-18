import os
import sys
import time
import json
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.server import ServingServer
from serving.client import ServingClient
from serving.predictor import Predictor

def run_server():
    server = ServingServer(host="localhost", port=8080)
    server.run()

def main():
    print("=" * 60)
    print("EnNeuro Serving 模块演示")
    print("=" * 60)
    print()

    print("[步骤1] 启动服务...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(3)
    print("服务启动成功!")
    print()

    client = ServingClient(host="localhost", port=8080)

    print("[步骤2] 健康检查 (ping)...")
    result = client.ping()
    print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print()

    print("[步骤3] 服务状态检查 (health)...")
    result = client.health()
    print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print()

    print("[步骤4] 模型预测演示 (predict)...")
    print("输入数据: 模拟自动驾驶场景图片数据 (3x120x160)")
    print("输出结果: 方向盘转角预测值")
    
    import numpy as np
    input_data = np.random.rand(3 * 120 * 160).tolist()
    inputs = [input_data]
    
    result = client.predict(inputs)
    print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print()

    print("[步骤5] 批量预测演示...")
    batch_inputs = [
        np.random.rand(3 * 120 * 160).tolist(),
        np.random.rand(3 * 120 * 160).tolist(),
        np.random.rand(3 * 120 * 160).tolist()
    ]
    
    result = client.predict(batch_inputs)
    print(f"输入批量: {len(batch_inputs)} 个样本")
    print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print()

    print("[步骤6] 性能测试...")
    start_time = time.time()
    for i in range(10):
        client.predict([np.random.rand(3 * 120 * 160).tolist()])
    elapsed = (time.time() - start_time) * 1000
    print(f"10次预测平均延迟: {elapsed / 10:.2f} ms")
    print(f"吞吐量: {10 / (elapsed / 1000):.2f} req/s")
    print()

    print("=" * 60)
    print("演示完成!")
    print("=" * 60)
    print()
    print("服务仍在运行中，可以继续测试...")
    print("按 Ctrl+C 停止服务")

if __name__ == "__main__":
    try:
        main()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n服务已停止")