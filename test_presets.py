import sys
sys.path.insert(0, 'D:/Undergraduate/WZH/EnNeuro/tmp/Enneuro-Project/code')
from web_server.services.model_registry import build_model_from_config, list_presets

presets = list_presets()
print(f'Total presets: {len(presets)}')
for p in presets:
    mid, m = build_model_from_config({'preset': p['name']})
    print(f"  {p['name']:20s} -> {type(m).__name__}")
print('ALL OK')
