import socket
import tempfile
import unittest
import json
import pickle
from functools import partial
from pathlib import Path

import numpy as np

from eneuro.base import Tensor
from eneuro.distributed.aggregator import aggregate
from eneuro.distributed.demo import make_model, make_dataset, make_optimizer, make_adam_optimizer
from eneuro.distributed.launcher import DistributedConfig, launch
from eneuro.distributed.partition import worker_batches
from eneuro.distributed.protocol import send_message, recv_message
from eneuro.distributed.state import install_gradients, named_state
from eneuro.nn.loss import meanSquaredError
from eneuro.nn.loss import crossEntropyError
from eneuro.distributed.mnist import make_lenet, make_mnist_dataset, make_optimizer as make_mnist_optimizer
from eneuro.train.trainer import Trainer
from eneuro.data.dataloader import DataLoader
from eneuro.data.dataset import Dataset


class DistributedPSTests(unittest.TestCase):
    def test_spawn_prerequisites(self):
        self.assertIsNot(make_adam_optimizer(make_model().get_params_list())._state,
                         make_adam_optimizer(make_model().get_params_list())._state)
        pickle.dumps(Dataset())
        self.assertEqual([name for name, _ in make_model().named_parameters()],
                         sorted(name for name, _ in make_model().named_parameters()))

    def test_partition_complete_and_reproducible(self):
        shards = [worker_batches(12, 4, 2, rank, 7, 0, shuffle=True) for rank in range(2)]
        self.assertEqual(sorted(index for shard in shards for batch in shard for index in batch), list(range(12)))
        self.assertEqual(shards[0], worker_batches(12, 4, 2, 0, 7, 0, shuffle=True))


    def test_protocol_binary_roundtrip(self):
        left, right = socket.socketpair()
        try:
            values = {'weight': np.arange(6, dtype=np.float32).reshape(2, 3)}
            send_message(left, {'type': 'PUSH', 'model_version': 2}, values)
            header, loaded, _ = recv_message(right)
            self.assertEqual(header['type'], 'PUSH')
            self.assertEqual(loaded['weight'].dtype, values['weight'].dtype)
            np.testing.assert_array_equal(loaded['weight'], values['weight'])
        finally:
            left.close()
            right.close()


    def test_weighted_gradients_match_global_batch(self):
        dataset = make_dataset()
        whole = make_model()
        reference = make_model()
        full_x = Tensor.stack([dataset[i][0] for i in range(4)])
        full_y = Tensor.stack([dataset[i][1] for i in range(4)])
        _, _, expected = Trainer(whole, meanSquaredError, None).compute_batch_gradients(full_x, full_y)
        pieces = []
        for rank in range(2):
            model = make_model()
            indices = [rank, rank + 2]
            xb = Tensor.stack([dataset[i][0] for i in indices])
            yb = Tensor.stack([dataset[i][1] for i in indices])
            _, count, gradients = Trainer(model, meanSquaredError, None).compute_batch_gradients(xb, yb)
            pieces.append((count, gradients))
        shapes = {name: value.shape for name, value in expected.items()}
        averaged = aggregate(pieces, shapes)
        for name in expected:
            np.testing.assert_allclose(averaged[name], expected[name], atol=1e-6, rtol=1e-6)
        install_gradients(whole, expected)
        install_gradients(reference, averaged)
        make_optimizer(whole.get_params_list()).step()
        make_optimizer(reference.get_params_list()).step()
        for name in named_state(whole):
            np.testing.assert_allclose(named_state(whole)[name], named_state(reference)[name], atol=1e-6)

    def test_uneven_batch_weighting_and_validation(self):
        result = aggregate([(3, {'w': np.array([2.])}), (1, {'w': np.array([6.])})], {'w': (1,)})
        np.testing.assert_allclose(result['w'], [3.])
        with self.assertRaises(ValueError):
            aggregate([(1, {'wrong': np.array([1.])})], {'w': (1,)})


    def test_spawn_ps_matches_single_process(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            config = DistributedConfig(port=port, world_size=2, epochs=2, global_batch_size=4,
                                       shuffle=False, timeout_s=20, metrics_dir=Path(directory))
            actual = launch(make_model, make_dataset, meanSquaredError, make_optimizer, config)
            reference = make_model()
            optimizer = make_optimizer(reference.get_params_list())
            dataset = make_dataset()
            trainer = Trainer(reference, meanSquaredError, optimizer)
            reference_losses = []
            for _ in range(config.epochs):
                for start in range(0, len(dataset), config.global_batch_size):
                    indices = range(start, start + config.global_batch_size)
                    xb = Tensor.stack([dataset[i][0] for i in indices])
                    yb = Tensor.stack([dataset[i][1] for i in indices])
                    loss, _, gradients = trainer.compute_batch_gradients(xb, yb)
                    reference_losses.append(loss)
                    install_gradients(reference, gradients)
                    optimizer.step()
            self.assertEqual(actual['model_version'], 8)
            for name, value in named_state(reference).items():
                np.testing.assert_allclose(actual['parameters'][name], value, atol=1e-5, rtol=1e-5)
            self.assertTrue((Path(directory) / 'ps.jsonl').exists())
            self.assertTrue((Path(directory) / 'worker-0.jsonl').exists())
            distributed_losses = {}
            for rank in range(2):
                with (Path(directory) / f'worker-{rank}.jsonl').open(encoding='utf-8') as stream:
                    for line in stream:
                        record = json.loads(line)
                        if record['event'] == 'compute':
                            distributed_losses.setdefault(record['step'], []).append(
                                (record['loss'], record['sample_count']))
            self.assertEqual(len(distributed_losses), len(reference_losses))
            for step, expected in enumerate(reference_losses):
                entries = distributed_losses[step]
                self.assertEqual(len(entries), 2)
                actual_loss = sum(loss * count for loss, count in entries) / sum(count for _, count in entries)
                self.assertAlmostEqual(actual_loss, expected, places=5)

    def test_existing_trainer_still_trains(self):
        model = make_model()
        trainer = Trainer(model, meanSquaredError, make_optimizer(model.get_params_list()))
        loader = DataLoader(make_dataset(), batch_size=4, shuffle=False)
        trainer.fit(loader, loader, epochs=1, batch_size=4, verbose=False)

    def test_spawn_adam_matches_single_process(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            config = DistributedConfig(port=port, world_size=2, epochs=1, global_batch_size=4,
                                       shuffle=False, timeout_s=20, metrics_dir=Path(directory),
                                       checkpoint_path=Path(directory) / 'checkpoint.json')
            actual = launch(make_model, make_dataset, meanSquaredError, make_adam_optimizer, config)
            reference = make_model()
            optimizer = make_adam_optimizer(reference.get_params_list())
            dataset = make_dataset()
            trainer = Trainer(reference, meanSquaredError, optimizer)
            for start in range(0, len(dataset), 4):
                indices = range(start, start + 4)
                xb = Tensor.stack([dataset[i][0] for i in indices])
                yb = Tensor.stack([dataset[i][1] for i in indices])
                _, _, gradients = trainer.compute_batch_gradients(xb, yb)
                install_gradients(reference, gradients)
                optimizer.step()
            self.assertEqual(actual['optimizer_state']['t'], 4)
            self.assertEqual(actual['optimizer_state']['t'], optimizer.to_dict()['t'])
            for name, value in named_state(reference).items():
                np.testing.assert_allclose(actual['parameters'][name], value, atol=1e-5, rtol=1e-5)
            self.assertTrue((Path(directory) / 'checkpoint.json').exists())

    def test_mnist_lenet_two_worker_step_matches_single_process(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            config = DistributedConfig(port=port, world_size=2, epochs=1, global_batch_size=32,
                                       shuffle=False, timeout_s=120, metrics_dir=Path(directory))
            dataset_factory = partial(make_mnist_dataset, limit=32)
            actual = launch(make_lenet, dataset_factory, crossEntropyError, make_mnist_optimizer, config)
            model = make_lenet()
            dataset = dataset_factory()
            xb = Tensor.stack([Tensor(np.asarray(dataset[i][0])) for i in range(32)])
            yb = Tensor.stack([Tensor(np.asarray(dataset[i][1])) for i in range(32)])
            _, _, gradients = Trainer(model, crossEntropyError, None).compute_batch_gradients(xb, yb)
            install_gradients(model, gradients)
            make_mnist_optimizer(model.get_params_list()).step()
            self.assertEqual(actual['model_version'], 1)
            for name, value in named_state(model).items():
                np.testing.assert_allclose(actual['parameters'][name], value, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
