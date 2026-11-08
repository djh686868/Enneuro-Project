"""Real-image smoke and multi-step equivalence checks for the PS AutoDriveNet."""

import json
import socket
import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np

from eneuro.base import Tensor
from eneuro.distributed.autodrive import (make_autodrive_model, make_optimizer,
                                          make_steering_dataset)
from eneuro.distributed.launcher import DistributedConfig, launch
from eneuro.distributed.state import install_gradients, named_state
from eneuro.nn.loss import meanSquaredError
from eneuro.train.trainer import Trainer


class AutoDrivePSTests(unittest.TestCase):
    def test_real_csv_sample_and_eager_model(self):
        dataset = make_steering_dataset(limit=8)
        image, steering = dataset[0]
        self.assertEqual(image.shape, (3, 120, 160))
        self.assertEqual(image.dtype, np.float32)
        self.assertTrue(0 <= image.min() <= image.max() <= 1)
        self.assertEqual(steering.shape, (1,))
        self.assertEqual(steering.dtype, np.float32)
        model = make_autodrive_model()
        self.assertTrue(all(param.data is not None for _, param in model.named_parameters()))
        prediction = model(Tensor.stack([Tensor(image)]))
        self.assertEqual(prediction.shape, (1, 1))

    def test_two_worker_loss_and_parameters_match_single_process(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            metrics_dir = Path(directory)
            config = DistributedConfig(port=port, world_size=2, epochs=1,
                                       global_batch_size=4, shuffle=False,
                                       timeout_s=120, metrics_dir=metrics_dir)
            factory = partial(make_steering_dataset, limit=8)
            actual = launch(make_autodrive_model, factory, meanSquaredError,
                            make_optimizer, config)

            dataset = factory()
            reference = make_autodrive_model()
            optimizer = make_optimizer(reference.get_params_list())
            trainer = Trainer(reference, meanSquaredError, None)
            expected_losses = []
            for start in range(0, len(dataset), config.global_batch_size):
                batch = [dataset[i] for i in range(start, start + config.global_batch_size)]
                images = Tensor.stack([Tensor(image) for image, _ in batch])
                labels = Tensor.stack([Tensor(steering) for _, steering in batch])
                loss, _, gradients = trainer.compute_batch_gradients(images, labels)
                expected_losses.append(loss)
                install_gradients(reference, gradients)
                optimizer.step()
                reference.cleargrads()

            self.assertEqual(actual['model_version'], 2)
            worker_losses = [[] for _ in expected_losses]
            for rank in range(2):
                with (metrics_dir / f'worker-{rank}.jsonl').open(encoding='utf-8') as stream:
                    for line in stream:
                        record = json.loads(line)
                        if record['event'] == 'compute':
                            worker_losses[record['step']].append(
                                (record['loss'], record['sample_count']))
            for expected, parts in zip(expected_losses, worker_losses):
                self.assertEqual(len(parts), 2)
                distributed_loss = sum(loss * count for loss, count in parts) / sum(count for _, count in parts)
                self.assertAlmostEqual(distributed_loss, expected, places=5)
            for name, expected in named_state(reference).items():
                np.testing.assert_allclose(actual['parameters'][name], expected,
                                           atol=1e-5, rtol=1e-5, err_msg=name)


if __name__ == '__main__':
    unittest.main()
