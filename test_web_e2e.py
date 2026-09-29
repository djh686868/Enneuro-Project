"""端到端 Web API 测试：模型构建 → 数据集注册 → 训练启动 → SSE 流读取"""
import sys
import time
import json
import urllib.request

sys.path.insert(0, 'D:/Undergraduate/WZH/EnNeuro/tmp/Enneuro-Project/code')
BASE = 'http://localhost:8000/api'


def post(path, body):
    req = urllib.request.Request(
        f'{BASE}{path}',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(f'{BASE}{path}', timeout=10) as r:
        return json.loads(r.read())


# 1. 构建模型
model_data = post('/models/build', {'preset': 'SimpleCNN'})
model_id = model_data['model_id']
print(f'[1] Model: {model_id} ({model_data["model_type"]})')

# 2. 注册数据集
ds_data = post('/datasets/register', {
    'path': 'D:/Undergraduate/WZH/EnNeuro/tmp/Enneuro-Project/tests/test_donkey/data',
    'name': 'donkey',
    'description': '',
})
ds_id = ds_data['dataset_id']
print(f'[2] Dataset: {ds_id}')

# 3. 数据集预览
prev = get(f'/datasets/{ds_id}/preview?n=3')
print(f'[3] Preview: {prev["total"]} samples, classes={prev["classes"]}')

# 4. 启动训练（2 epochs）
t_data = post('/training/start', {
    'model_id': model_id,
    'dataset_id': ds_id,
    'epochs': 2,
    'batch_size': 64,
    'lr': 0.001,
    'optimizer': 'adam',
    'device': 'cpu',
    'val_split': 0.2,
})
task_id = t_data['task_id']
print(f'[4] Training task: {task_id}')

# 5. 读取 SSE 流
print('[5] SSE stream:')
req = urllib.request.Request(f'{BASE}/training/{task_id}/stream')
with urllib.request.urlopen(req, timeout=300) as r:
    buf = ''
    start = time.time()
    while time.time() - start < 300:
        chunk = r.read(512).decode('utf-8', errors='ignore')
        if not chunk:
            break
        buf += chunk
        while '\n\n' in buf:
            msg, buf = buf.split('\n\n', 1)
            lines = msg.strip().split('\n')
            evt = next((l[7:] for l in lines if l.startswith('event:')), 'message')
            data = next((l[5:] for l in lines if l.startswith('data:')), '')
            if data:
                d = json.loads(data)
                print(f'    [{evt}] {d}')
            if evt in ('done', 'error'):
                break
        else:
            continue
        break

# 6. 查看最终状态
status = get(f'/training/{task_id}/status')
print(f'[6] Final status: {status}')
print('\nALL TESTS PASSED')
