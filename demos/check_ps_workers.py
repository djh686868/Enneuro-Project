"""Verify local synchronous PS training against a single-process run."""

import argparse
import json
import multiprocessing
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from eneuro.data.dataloader import DataLoader
from eneuro.distributed.demo import make_dataset, make_model, make_optimizer
from eneuro.distributed.launcher import DistributedConfig, launch
from eneuro.distributed.state import install_gradients, named_state
from eneuro.nn.loss import meanSquaredError
from eneuro.train.trainer import Trainer


def single_process_reference(dataset, batch_size, steps):
    model = make_model()
    optimizer = make_optimizer(model.get_params_list())
    trainer = Trainer(model, meanSquaredError, optimizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses = []
    for _ in range(steps):
        for inputs, targets in loader:
            loss, _, gradients = trainer.compute_batch_gradients(inputs, targets)
            install_gradients(model, gradients)
            optimizer.step()
            losses.append(loss)
    return losses, named_state(model)


def read_worker_losses(metrics_dir, workers, batch_size, steps):
    weighted_losses = np.zeros(steps)
    expected_steps = list(range(steps))
    for rank in range(workers):
        with (metrics_dir / f"worker-{rank}.jsonl").open(encoding="utf-8") as log:
            events = [json.loads(line) for line in log]
        compute = [event for event in events if event["event"] == "compute"]
        pushed_steps = [event["step"] for event in events if event["event"] == "push"]
        if [event["step"] for event in compute] != expected_steps or pushed_steps != expected_steps:
            raise RuntimeError(f"Worker {rank} did not compute and push every step.")

        samples_per_step = len(range(rank, batch_size, workers))
        if any(event["sample_count"] != samples_per_step for event in compute):
            raise RuntimeError(f"Worker {rank} received an unexpected number of samples.")
        weighted_losses += samples_per_step * np.array([event["loss"] for event in compute])
        print(f"Worker {rank} contributed {samples_per_step} samples per step across {steps} steps.")
    return weighted_losses / batch_size


def main():
    dataset = make_dataset()
    batch_size = len(dataset)
    steps = 3
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, choices=range(2, batch_size + 1))
    args = parser.parse_args()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    with TemporaryDirectory(prefix="enneuro-ps-check-") as directory:
        config = DistributedConfig(
            port=port,
            world_size=args.workers,
            epochs=steps,
            global_batch_size=batch_size,
            shuffle=False,
            metrics_dir=Path(directory),
        )
        result = launch(make_model, make_dataset, meanSquaredError, make_optimizer, config)
        if result["model_version"] != steps:
            raise RuntimeError(f"Expected {steps} PS updates, got {result['model_version']}.")
        ps_losses = read_worker_losses(config.metrics_dir, args.workers, batch_size, steps)

    reference_losses, reference_parameters = single_process_reference(dataset, batch_size, steps)
    np.testing.assert_allclose(ps_losses, reference_losses, rtol=1e-5, atol=1e-6, equal_nan=False)
    if result["parameters"].keys() != reference_parameters.keys():
        raise RuntimeError("Parameter names differ from the single-process model.")
    for name, expected in reference_parameters.items():
        np.testing.assert_allclose(
            result["parameters"][name], expected, rtol=1e-5, atol=1e-6, equal_nan=False, err_msg=name
        )
    print(f"PASS | {args.workers} workers, {steps} PS updates, losses and parameters match.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
