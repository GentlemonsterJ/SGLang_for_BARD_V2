import sys
import types
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

sgl_kernel_module = types.ModuleType("sgl_kernel")
sgl_kernel_kvcacheio_module = types.ModuleType("sgl_kernel.kvcacheio")
for _name in [
    "transfer_kv_all_layer",
    "transfer_kv_all_layer_direct_lf_pf",
    "transfer_kv_all_layer_lf_pf",
    "transfer_kv_all_layer_lf_ph",
    "transfer_kv_all_layer_mla",
    "transfer_kv_all_layer_mla_lf_pf",
    "transfer_kv_direct",
    "transfer_kv_per_layer",
    "transfer_kv_per_layer_direct_pf_lf",
    "transfer_kv_per_layer_mla",
    "transfer_kv_per_layer_mla_pf_lf",
    "transfer_kv_per_layer_pf_lf",
    "transfer_kv_per_layer_ph_lf",
]:
    setattr(sgl_kernel_kvcacheio_module, _name, lambda *args, **kwargs: None)
sgl_kernel_module.kvcacheio = sgl_kernel_kvcacheio_module
sys.modules.setdefault("sgl_kernel", sgl_kernel_module)
sys.modules.setdefault("sgl_kernel.kvcacheio", sgl_kernel_kvcacheio_module)

from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _DummyForwardMode:
    def is_decode(self):
        return False

    def is_target_verify(self):
        return False


class _DummyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.pp_group = SimpleNamespace(is_first_rank=True)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids, forward_batch, input_embeds=None, **kwargs):
        return input_embeds


class TestGeneralMmEmbedRoutine(unittest.TestCase):
    def test_dllm_extend_is_treated_as_extend_like_mode(self):
        self.assertTrue(ForwardMode.DLLM_EXTEND.is_extend())
        self.assertTrue(
            ForwardMode.DLLM_EXTEND.is_extend_or_draft_extend_or_mixed()
        )

    def test_reuses_cached_mm_embeddings_for_oov_placeholders(self):
        model = _DummyLanguageModel(vocab_size=10, hidden_size=4)
        with torch.no_grad():
            model.embed_tokens.weight.copy_(torch.arange(40).view(10, 4))

        input_ids = torch.tensor([2, 1_000_123, 7, 3], dtype=torch.int64)
        cached_mm_input_embeds = torch.tensor(
            [
                [8.0, 9.0, 10.0, 11.0],
                [101.0, 102.0, 103.0, 104.0],
                [28.0, 29.0, 30.0, 31.0],
                [12.0, 13.0, 14.0, 15.0],
            ]
        )

        forward_batch = SimpleNamespace(
            forward_mode=_DummyForwardMode(),
            mm_inputs=None,
            mm_input_embeds=cached_mm_input_embeds.clone(),
            input_embeds=None,
            contains_mm_inputs=lambda: False,
        )

        output = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=model,
        )

        expected = model.embed_tokens(input_ids.clamp(max=9))
        expected[1] = cached_mm_input_embeds[1]

        self.assertTrue(torch.equal(output, expected))


if __name__ == "__main__":
    unittest.main()
