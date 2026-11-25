"""Exercise the public training API with real spawned workers and checkpoints."""

import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import numpy as np
import pytest

from eneuro.base import Tensor
from eneuro.distributed.demo import make_adam_optimizer, make_dataset, make_model, make_optimizer
from eneuro.distributed.monitor import Monitor
from eneuro.distributed.ring import _allreduce, _broadcast, _make_links
from eneuro.distributed.state import install_gradients, named_state
from eneuro.base.parameter import Parameter
from eneuro.nn.loss import meanSquaredError
from eneuro.profiling.distributed import benchmark_metrics, summarize_timings, worker_loss_trace
from eneuro.train import DistributedConfig, Trainer, launch_distributed
from eneuro.utils.serializer import Serializer


def test_ring_broadcast_replaces_independently_initialized_parameters(tmp_path):
    workers = 3
    models = [make_model() for _ in range(workers)]
    for rank, model in enumerate(models):
        model.scale = Parameter(np.array(float(rank)))
        model.scale.requires_grad = False
        for _, parameter in model.named_parameters():
            parameter.data[...] += rank
    expected = named_state(models[0])
    dataset = make_dataset()
    with ExitStack() as resources:
        incoming, outgoing = _make_links(DistributedConfig(world_size=workers), resources)
        for link in incoming + outgoing:
            link.settimeout(10)
        ranks = resources.enter_context(ThreadPoolExecutor(max_workers=workers))
        pending = []
        for rank, model in enumerate(models):
            monitor = Monitor(tmp_path / f"worker-{rank}.jsonl", "worker", rank)
            pending.append(ranks.submit(_broadcast, rank, workers, model, dataset,
                                        incoming[rank], outgoing[rank], monitor))
        for task in pending:
            task.result()
    for model in models:
        for name, value in named_state(model).items():
            np.testing.assert_array_equal(value, expected[name])
        assert model.scale.data.shape == ()


@pytest.mark.parametrize("size", [2, 262147])
def test_every_rank_receives_all_chunks(size):
    # Cover an empty chunk and a large uneven vector spanning socket buffers.
    workers = 3
    vectors = [np.arange(size, dtype=np.float64) + rank for rank in range(workers)]
    expected = np.sum(vectors, axis=0)
    with ExitStack() as resources:
        incoming, outgoing = _make_links(DistributedConfig(world_size=workers), resources)
        for link in incoming + outgoing:
            link.settimeout(10)
        senders = resources.enter_context(ThreadPoolExecutor(max_workers=workers))
        ranks = resources.enter_context(ThreadPoolExecutor(max_workers=workers))
        pending = [ranks.submit(_allreduce, vector, rank, workers, incoming[rank], outgoing[rank], senders)
                   for rank, vector in enumerate(vectors)]
        transfers = [task.result() for task in pending]
    for vector in vectors:
        np.testing.assert_array_equal(vector, expected)
    sent, received = zip(*transfers)
    assert sum(sent) == sum(received) == 2 * (workers - 1) * expected.nbytes


def _serial_reference(config):
    model = make_model()
    optimizer = make_adam_optimizer(model.get_params_list())
    trainer = Trainer(model, meanSquaredError, optimizer)
    dataset = make_dataset()
    losses = []
    for epoch in range(config.epochs):
        order = np.random.default_rng(config.seed + epoch).permutation(len(dataset))
        for start in range(0, len(dataset), config.global_batch_size):
            samples = [dataset[i] for i in order[start:start + config.global_batch_size]]
            x, y = zip(*samples)
            loss, _, gradients = trainer.compute_batch_gradients(Tensor.stack(x), Tensor.stack(y))
            install_gradients(model, gradients)
            optimizer.step()
            losses.append(loss)
    return named_state(model), optimizer.to_dict(), losses


@pytest.mark.parametrize("workers", [1, 2, 3, 4])
def test_ring_matches_global_batches_and_restores_adam(workers, tmp_path):
    # Sixteen examples yield batches of 6, 6 and 4. With 3 or 4 workers,
    # this exercises uneven local counts, including the retained final batch.
    config = DistributedConfig(backend="ring", world_size=workers, port=0,
                               epochs=2, global_batch_size=6, drop_last=False,
                               metrics_dir=tmp_path, checkpoint_path=tmp_path / "state.pkl")
    result = launch_distributed(make_model, make_dataset, meanSquaredError, make_adam_optimizer, config)
    expected, state, losses = _serial_reference(config)
    assert result["model_version"] == len(losses)

    restored = make_model()
    optimizer = make_adam_optimizer(restored.get_params_list())
    assert Serializer.load_checkpoint(config.checkpoint_path, restored, optimizer) == config.epochs
    assert result["optimizer_state"]["t"] == optimizer.to_dict()["t"] == len(losses)
    for name, value in expected.items():
        np.testing.assert_allclose(result["parameters"][name], value, atol=1e-6, rtol=1e-5)
        np.testing.assert_array_equal(named_state(restored)[name], result["parameters"][name])
    for moment in ("v", "s"):
        for index, value in state[moment].items():
            np.testing.assert_allclose(optimizer.to_dict()[moment][index], value, atol=1e-6, rtol=1e-5)

    weighted = np.zeros(len(losses))
    counts = np.zeros(len(losses), dtype=int)
    for rank in range(workers):
        records = [json.loads(line) for line in (tmp_path / f"worker-{rank}.jsonl").read_text().splitlines()]
        reductions = [r for r in records if r["event"] == "allreduce"]
        assert [r["step"] for r in reductions] == list(range(len(losses)))
        assert all((r["bytes_sent"] > 0) == (workers > 1) for r in reductions)
        for record in records:
            if record["event"] == "compute":
                weighted[record["step"]] += record["loss"] * record["sample_count"]
                counts[record["step"]] += record["sample_count"]
    np.testing.assert_array_equal(counts, [6, 6, 4] * config.epochs)
    np.testing.assert_allclose(weighted / counts, losses, atol=1e-6, rtol=1e-5)


def _invalid_dataset():
    dataset = make_dataset()
    dataset.x[1] = np.nan
    return dataset


def test_ring_propagates_peer_failure_and_reaps_workers(tmp_path):
    before = {child.pid for child in mp.active_children()}
    config = DistributedConfig(backend="ring", world_size=2, global_batch_size=4,
                               port=0, shuffle=False, metrics_dir=tmp_path)
    with pytest.raises(RuntimeError, match="ring worker"):
        launch_distributed(make_model, _invalid_dataset, meanSquaredError, make_optimizer, config)
    assert not {child.pid for child in mp.active_children()} - before


def test_independent_callers_can_launch_ps_and_ring_concurrently(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as callers:
        jobs = []
        for backend in ("ps", "ring"):
            config = DistributedConfig(backend=backend, world_size=2, global_batch_size=4,
                                       port=0, metrics_dir=tmp_path / backend)
            jobs.append(callers.submit(launch_distributed, make_model, make_dataset,
                                       meanSquaredError, make_optimizer, config,
                                       profiling={"system_metrics": False, "output_dir": config.metrics_dir}))
        ps, ring = [job.result(timeout=60) for job in jobs]
    assert ps["model_version"] == ring["model_version"] == 4
    for name, value in ps["parameters"].items():
        np.testing.assert_allclose(ring["parameters"][name], value, atol=1e-6, rtol=1e-5)
    for backend, result, comm in (("ps", ps, "push"), ("ring", ring, "allreduce")):
        report = result["profiling"]["report"]
        assert report["timings"]["distributed.launch"]["wall_ms"]["count"] == 1
        summary = report["observations"]["distributed"]["metrics"]
        assert summary["backend"] == backend
        assert summary["errors"] == {}
        assert (tmp_path / backend / f"{report['run_id']}.json").is_file()
        for rank in range(2):
            worker = summary["roles"][f"worker-{rank}"]
            assert worker["samples"] == 8
            assert worker["timings"][comm]["count"] == 4
            assert worker["timings"][comm]["bytes_sent"] > 0
        if backend == "ps":
            assert summary["roles"]["ps"]["timings"]["aggregate_update"]["count"] == 4
        else:
            assert "ps" not in summary["roles"]


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        launch_distributed(make_model, make_dataset, meanSquaredError, make_optimizer,
                           DistributedConfig(backend="unknown"))


def test_distributed_log_summary_preserves_timing_and_loss_contract(tmp_path):
    def record(event, duration_ms, sent=0, received=0, **details):
        return {"event": event, "duration_ms": duration_ms, "bytes_sent": sent,
                "bytes_received": received, **details}

    rank_logs = [
        [record("compute", 40, step=0, sample_count=3, loss=1.5),
         record("pull", 5, 1, 80),
         record("push", 60, 100, 80, barrier_wait_ms=10),
         record("step", 100, start_perf_ns=0, end_perf_ns=100_000_000)],
        [record("compute", 50, step=0, sample_count=1, loss=5.5),
         record("pull", 6, 1, 85),
         record("push", 70, 120, 85, barrier_wait_ms=15),
         record("step", 120, start_perf_ns=30_000_000, end_perf_ns=150_000_000)],
    ]
    for rank, records in enumerate(rank_logs):
        (tmp_path / f"worker-{rank}.jsonl").write_text(
            "".join(json.dumps(entry) + "\n" for entry in records), encoding="utf-8",
        )
    (tmp_path / "ps.jsonl").write_text(json.dumps(record("aggregate_update", 30)) + "\n", encoding="utf-8")

    summary = summarize_timings(tmp_path, backend="ps", world_size=2)
    assert summary["errors"] == {}
    assert summary["roles"]["worker-1"]["timings"]["push"]["barrier_wait_ms"] == 15
    assert summary["roles"]["ps"]["timings"]["aggregate_update"]["mean_ms"] == 30

    assert benchmark_metrics(tmp_path, 2) == {
        "training_elapsed_s": 0.15,
        "samples_processed": 4,
        "train_samples_per_second": 4 / 0.15,
        "compute_ms_sum": 90.0,
        "push_ms_sum": 130.0,
        "pull_ms_sum": 11.0,
        "allreduce_ms_sum": 0.0,
        "step_ms_sum": 220.0,
        "communication_ratio": 130 / 220,
        "bytes_sent": 222,
        "bytes_received": 330,
    }
    assert worker_loss_trace(tmp_path, 2) == [(1, 2.5, 4)]
