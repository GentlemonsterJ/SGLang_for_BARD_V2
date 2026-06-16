import unittest
from types import SimpleNamespace

import torch

from sglang.srt.dllm.algorithm.dstar_low_confidence import (
    low_confidence_dynamic,
    low_confidence_static,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.entrypoints.openai.utils import (
    process_token_ids_from_ret,
    process_token_logprobs_from_ret,
)
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin


class _DummyModelRunner:
    def __init__(self, logits_sequence):
        self.logits_sequence = list(logits_sequence)
        self.forward_calls = 0

    def forward(self, forward_batch, pp_proxy_tensors=None):
        logits = self.logits_sequence[min(self.forward_calls, len(self.logits_sequence) - 1)]
        self.forward_calls += 1
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=logits),
            can_run_graph=False,
        )


class TestDstarStepMap(unittest.TestCase):
    @staticmethod
    def _make_forward_batch(return_step_map=False, return_step_confidence_map=False):
        return SimpleNamespace(
            batch_size=1,
            input_ids=torch.tensor([9, 9, 9, 9], dtype=torch.int64),
            return_step_map=[return_step_map],
            return_step_confidence_map=[return_step_confidence_map],
        )

    def test_low_confidence_static_returns_step_map(self):
        config = DllmConfig(
            algorithm="low_confidence_static",
            algorithm_config={"denoising_steps": 2},
            block_size=4,
            mask_id=9,
            max_running_requests=1,
            arch="DstarVLForConditonalGeneration",
        )
        algorithm = low_confidence_static(config)
        logits = torch.tensor(
            [
                [0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        model_runner = _DummyModelRunner([logits, logits, logits])
        forward_batch = self._make_forward_batch(return_step_map=True)

        _, next_token_ids, _, step_map, step_confidence_map = algorithm.run(
            model_runner, forward_batch
        )

        self.assertEqual(next_token_ids[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(step_map, [[1, 1, 2, 2]])
        self.assertIsNone(step_confidence_map)

    def test_low_confidence_dynamic_returns_step_map(self):
        config = DllmConfig(
            algorithm="low_confidence_dynamic",
            algorithm_config={"threshold": 0.8, "denoising_steps": 4},
            block_size=4,
            mask_id=9,
            max_running_requests=1,
            arch="DstarVLForConditonalGeneration",
        )
        algorithm = low_confidence_dynamic(config)
        logits_step_1 = torch.tensor(
            [
                [0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        logits_step_2 = torch.tensor(
            [
                [0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        logits_step_3 = torch.tensor(
            [
                [0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        model_runner = _DummyModelRunner(
            [logits_step_1, logits_step_2, logits_step_3, logits_step_3]
        )
        forward_batch = self._make_forward_batch(
            return_step_map=True, return_step_confidence_map=True
        )

        _, next_token_ids, _, step_map, step_confidence_map = algorithm.run(
            model_runner, forward_batch
        )

        self.assertEqual(next_token_ids[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(step_map, [[1, 1, 2, 3]])
        self.assertEqual(len(step_confidence_map[0]), 4)
        self.assertTrue(all(conf > 0.0 for conf in step_confidence_map[0]))

    def test_dstar_metadata_is_skipped_when_not_requested(self):
        config = DllmConfig(
            algorithm="low_confidence_dynamic",
            algorithm_config={"threshold": 0.8, "denoising_steps": 4},
            block_size=4,
            mask_id=9,
            max_running_requests=1,
            arch="DstarVLForConditonalGeneration",
        )
        algorithm = low_confidence_dynamic(config)
        logits = torch.tensor(
            [
                [0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        model_runner = _DummyModelRunner([logits, logits, logits, logits])
        forward_batch = self._make_forward_batch()

        result = algorithm.run(model_runner, forward_batch)

        self.assertEqual(len(result), 3)

    def test_scheduler_accumulates_step_map_across_blocks(self):
        scheduler = SchedulerDllmMixin()
        req = SimpleNamespace(customized_info=None)

        scheduler._append_dllm_step_map(req, [2, 1, 4, 3])
        scheduler._append_dllm_step_map(req, [2, 1, 4, 3])

        self.assertEqual(req.customized_info["step_map"], [2, 1, 4, 3, 6, 5, 8, 7])

    def test_scheduler_accumulates_step_confidence_map_across_blocks(self):
        scheduler = SchedulerDllmMixin()
        req = SimpleNamespace(customized_info=None)

        scheduler._append_dllm_step_confidence_map(req, [0.9, 0.8])
        scheduler._append_dllm_step_confidence_map(req, [0.7, 0.6])

        self.assertEqual(
            req.customized_info["step_confidence_map"], [0.9, 0.8, 0.7, 0.6]
        )

    def test_process_token_logprobs_from_ret_returns_flat_list(self):
        ret_item = {
            "meta_info": {
                "output_token_logprobs": [
                    (-0.1, 10, "A"),
                    (-0.2, 11, "B"),
                    (-0.3, 12, "C"),
                ]
            }
        }

        self.assertEqual(
            process_token_logprobs_from_ret(ret_item),
            [-0.1, -0.2, -0.3],
        )

    def test_process_token_ids_from_ret_returns_flat_list(self):
        ret_item = {
            "meta_info": {
                "output_token_logprobs": [
                    (-0.1, 10, "A"),
                    (-0.2, 11, "B"),
                    (-0.3, 12, "C"),
                ]
            }
        }

        self.assertEqual(process_token_ids_from_ret(ret_item), [10, 11, 12])


if __name__ == "__main__":
    unittest.main()
