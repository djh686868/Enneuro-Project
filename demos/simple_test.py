import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))

print("1. Importing eneuro...")
from eneuro.base import Tensor, Config
from eneuro.nn.loss import MSELoss
from eneuro.nn.optim import Adam
print("   OK")

print("\n2. Creating simple tensor...")
x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
print(f"   Shape: {x.shape}")
print("   OK")

print("\n3. Testing simple computation...")
y = x + 2
print(f"   Result: {y.data}")
print("   OK")

print("\n4. Testing backward...")
y.backward()
print(f"   Gradient: {x.grad}")
print("   OK")

print("\n=== Simple test completed ===")