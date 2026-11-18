import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path(__file__).resolve().parent / 'results'

results = {}

files = {
    'EnNeuro': 'benchmark_enneuro.json',
    'PyTorch': 'benchmark_pytorch.json',
    'PaddlePaddle': 'benchmark_paddle.json'
}

for framework, filename in files.items():
    filepath = results_dir / filename
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results[framework] = data
            print(f"{framework}: {json.dumps(data, indent=2)}")
    else:
        print(f"{framework}: 未找到结果文件")

frameworks = list(results.keys())
durations = [results[f]['duration'] for f in frameworks]
params = [results[f]['params'] for f in frameworks]
mem_peaks = [results[f]['mem_peak'] for f in frameworks]
gpu_peaks = [results[f]['gpu_peak'] for f in frameworks]
gpu_avgs = [results[f]['gpu_avg'] for f in frameworks]
avg_mses = [results[f].get('avg_mse', 0) for f in frameworks]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

axes[0, 0].bar(frameworks, durations, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0, 0].set_title('Epoch Duration (seconds)')
axes[0, 0].set_ylabel('Time (s)')
for i, v in enumerate(durations):
    axes[0, 0].text(i, v + max(durations)*0.02, f'{v:.1f}', ha='center', fontsize=9)

axes[0, 1].bar(frameworks, [p / 1e6 for p in params], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0, 1].set_title('Model Parameters (Million)')
axes[0, 1].set_ylabel('Params (M)')
for i, v in enumerate(params):
    axes[0, 1].text(i, (v / 1e6) + max(params)/1e6*0.05, f'{v/1e6:.2f}', ha='center', fontsize=9)

axes[0, 2].bar(frameworks, avg_mses, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0, 2].set_title('Average MSE After 1 Epoch')
axes[0, 2].set_ylabel('MSE')
for i, v in enumerate(avg_mses):
    axes[0, 2].text(i, v + max(avg_mses)*0.05, f'{v:.4f}', ha='center', fontsize=9)

axes[1, 0].bar(frameworks, mem_peaks, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1, 0].set_title('Memory Peak (MB)')
axes[1, 0].set_ylabel('Memory (MB)')
for i, v in enumerate(mem_peaks):
    axes[1, 0].text(i, v + max(mem_peaks)*0.02, f'{v:.1f}', ha='center', fontsize=9)

x = range(len(frameworks))
width = 0.35
axes[1, 1].bar([i - width/2 for i in x], gpu_peaks, width=width, label='GPU Peak', color='#1f77b4')
axes[1, 1].bar([i + width/2 for i in x], gpu_avgs, width=width, label='GPU Avg', color='#ff7f0e')
axes[1, 1].set_title('GPU Memory Usage (MB)')
axes[1, 1].set_ylabel('GPU Memory (MB)')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(frameworks)
axes[1, 1].legend()
for i, (p, a) in enumerate(zip(gpu_peaks, gpu_avgs)):
    axes[1, 1].text(i - width/2, p + max(gpu_peaks)*0.02, f'{p:.1f}', ha='center', fontsize=8)
    axes[1, 1].text(i + width/2, a + max(gpu_peaks)*0.02, f'{a:.1f}', ha='center', fontsize=8)

fig.delaxes(axes[1, 2])

plt.tight_layout()
output_path = results_dir / 'benchmark_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nComparison chart saved to: {output_path}")
plt.show()

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"{'Framework':<15} {'Duration(s)':<12} {'Params(M)':<10} {'Avg MSE':<10} {'MemPeak(MB)':<12} {'GPU Peak(MB)':<12} {'GPU Avg(MB)':<12}")
print("-"*70)
for f in frameworks:
    r = results[f]
    print(f"{f:<15} {r['duration']:<12.2f} {r['params']/1e6:<10.2f} {r.get('avg_mse', 0):<10.4f} {r['mem_peak']:<12.1f} {r['gpu_peak']:<12.1f} {r['gpu_avg']:<12.1f}")
print("="*70)
