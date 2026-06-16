import unittest

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase, ReqDllmMixin


class _DummyReq(ReqDllmMixin):
    def __init__(self, origin_input_ids, dllm_config):
        self.origin_input_ids = origin_input_ids
        self.output_ids = []
        self.prefix_indices = []
        self.fill_ids = []
        self.init_diffusion_llm(dllm_config)


class TestSdarDllmReqFlow(unittest.TestCase):
    def test_sdar_prefill_appends_mask_block_like_raw(self):
        config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=4,
            mask_id=151669,
            max_running_requests=1,
            arch="SDARForCausalLM",
        )
        req = _DummyReq([1, 2, 3, 4, 5], config)

        req._init_fill_ids_for_dllm()

        self.assertEqual(req.dllm_phase, DllmReqPhase.INCOMING_PREFILL)
        self.assertEqual(req.fill_ids, [1, 2, 3, 4, 5, 151669, 151669, 151669, 151669])

    def test_sdar_short_prompt_starts_in_incoming_decode(self):
        config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=4,
            mask_id=151669,
            max_running_requests=1,
            arch="SDARForCausalLM",
        )
        req = _DummyReq([1, 2, 3], config)

        self.assertEqual(req.dllm_phase, DllmReqPhase.INCOMING_DECODE)

    def test_non_sdar_keeps_current_prefill_behavior(self):
        config = DllmConfig(
            algorithm="low_confidence_dynamic",
            algorithm_config={},
            block_size=4,
            mask_id=151671,
            max_running_requests=1,
            arch="DstarVLForConditonalGeneration",
        )
        req = _DummyReq([1, 2, 3, 4, 5], config)

        req._init_fill_ids_for_dllm()

        self.assertEqual(req.dllm_phase, DllmReqPhase.INCOMING_PREFILL)
        self.assertEqual(req.fill_ids, [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
