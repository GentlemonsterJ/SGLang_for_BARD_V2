import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from sglang.srt.model_executor.cuda_graph_runner import (
    _is_dllm_multimodal_cuda_graph_unsupported,
)


class TestDllmMultimodalCudaGraph(unittest.TestCase):
    def test_rejects_dllm_multimodal_batches_with_mm_inputs(self):
        forward_batch = SimpleNamespace(
            input_embeds=None,
            mm_input_embeds=None,
            contains_mm_inputs=lambda: True,
        )

        self.assertTrue(
            _is_dllm_multimodal_cuda_graph_unsupported(
                True, True, forward_batch
            )
        )

    def test_rejects_dllm_multimodal_batches_with_cached_mm_embeddings(self):
        forward_batch = SimpleNamespace(
            input_embeds=None,
            mm_input_embeds=object(),
            contains_mm_inputs=lambda: False,
        )

        self.assertTrue(
            _is_dllm_multimodal_cuda_graph_unsupported(
                True, True, forward_batch
            )
        )

    def test_allows_non_multimodal_or_non_dllm_batches(self):
        forward_batch = SimpleNamespace(
            input_embeds=None,
            mm_input_embeds=None,
            contains_mm_inputs=lambda: False,
        )

        self.assertFalse(
            _is_dllm_multimodal_cuda_graph_unsupported(
                False, True, forward_batch
            )
        )
        self.assertFalse(
            _is_dllm_multimodal_cuda_graph_unsupported(
                True, False, forward_batch
            )
        )


if __name__ == "__main__":
    unittest.main()
