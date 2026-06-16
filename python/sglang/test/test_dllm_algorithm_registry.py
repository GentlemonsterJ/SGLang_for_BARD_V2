import unittest

from sglang.srt.dllm.algorithm import algo_name_to_cls, get_algorithm
from sglang.srt.dllm.config import DllmConfig


class TestDllmAlgorithmRegistry(unittest.TestCase):
    def test_preserves_original_algorithm_names(self):
        self.assertIn("LowConfidence", algo_name_to_cls)
        self.assertIn("JointThreshold", algo_name_to_cls)

    def test_registers_dstar_specific_algorithm_names(self):
        self.assertIn("low_confidence_static", algo_name_to_cls)
        self.assertIn("low_confidence_dynamic", algo_name_to_cls)

    def test_original_low_confidence_defaults_are_unchanged(self):
        algo = get_algorithm(
            DllmConfig(
                algorithm="LowConfidence",
                algorithm_config={},
                block_size=4,
                mask_id=1,
                max_running_requests=1,
            )
        )
        self.assertEqual(algo.threshold, 0.95)
        self.assertFalse(hasattr(algo, "denoising_steps"))

    def test_static_dstar_algorithm_defaults(self):
        algo = get_algorithm(
            DllmConfig(
                algorithm="low_confidence_static",
                algorithm_config={},
                block_size=4,
                mask_id=1,
                max_running_requests=1,
            )
        )
        self.assertEqual(algo.denoising_steps, 4)

    def test_dynamic_dstar_algorithm_defaults(self):
        algo = get_algorithm(
            DllmConfig(
                algorithm="low_confidence_dynamic",
                algorithm_config={},
                block_size=4,
                mask_id=1,
                max_running_requests=1,
            )
        )
        self.assertEqual(algo.threshold, 0.95)
        self.assertEqual(algo.denoising_steps, 4)
        self.assertEqual(algo.pad_target_penalty, 1.0)


if __name__ == "__main__":
    unittest.main()
