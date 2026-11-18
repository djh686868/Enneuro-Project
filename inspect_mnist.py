import sys, pickle, numpy as np
sys.path.insert(0, 'D:/Undergraduate/WZH/EnNeuro/tmp/Enneuro-Project/code')

path = 'D:/Undergraduate/WZH/EnNeuro/tmp/Enneuro-Project/code/tests/testdata/MNIST_data/mnist.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)

print('type:', type(data))
if isinstance(data, dict):
    for k, v in data.items():
        sh = getattr(v, 'shape', None)
        dt = getattr(v, 'dtype', None)
        print(f'  key={repr(k)}, type={type(v).__name__}, shape={sh}, dtype={dt}')
        if hasattr(v, '__len__') and len(v) > 0:
            print(f'    first element: {repr(v[0]) if not hasattr(v[0], "shape") else v[0][:5]}')
elif isinstance(data, (list, tuple)):
    for i, v in enumerate(data):
        sh = getattr(v, 'shape', None)
        print(f'  [{i}] type={type(v).__name__}, shape={sh}')
else:
    print('  value:', repr(data)[:200])
