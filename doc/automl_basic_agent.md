# Basic AutoML tuning agent

This branch contains a small, dependency-free sequential HPO loop for EnNeuro.
It follows the AutoML MOOC's introductory setup: sample a low-dimensional
configuration space, evaluate one configuration, observe its validation metric,
and choose the next configuration under a fixed budget.

The core API is `SequentialRandomSearch`. It records `observe`, `decide`,
`execute`, and `feedback` events and writes a complete JSON report plus JSONL
event stream. The default search space samples learning rate uniformly in
log10 space and treats batch size and optimizer as discrete choices.

Run a small real MNIST search from the repository root with:

```powershell
$env:PYTHONPATH = "code"
python demos/automl_mnist_agent.py `
  --data "D:\Undergraduate\WZH\EnNeuro\tmp\Enneuro-Project\code\tests\testdata\MNIST_data\mnist.pkl" `
  --trials 4 --train-samples 256 --test-samples 256 --epochs 1 `
  --output artifacts/automl_mnist_real
```

The behavioral-cloning demo reads `driving_log.csv` and all three camera image
streams directly from `car_behavioral_cloning_data.zip` (with the standard
left/right steering correction), so the complete image set is never
materialized as one array:

```powershell
$env:PYTHONPATH = "code"
python demos/automl_car_agent.py `
  --zip "D:\Undergraduate\WZH\EnNeuro\car_behavioral_cloning_data.zip" `
  --device cpu --trials 3 --epochs 1 --image-width 32 --image-height 16 `
  --output artifacts/automl_car_full_cpu
```

The car objective is validation MSE; the report also includes validation/test
MAE, sample counts, elapsed time, and the selected configuration. A larger CNN
or CUDA backend can be plugged into the same executor interface after the
search loop has been validated.
