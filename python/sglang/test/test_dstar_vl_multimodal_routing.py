import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
    _uses_qwen3_vl_style_t_index,
    _uses_qwen3_vl_style_video_grid,
)
from sglang.srt.multimodal.processors.qwen_vl import (
    QwenVLImageProcessor,
    _uses_qwen3_vl_video_metadata,
)


class TestDstarVLMultimodalRouting(unittest.TestCase):
    def test_dstar_vl_uses_qwen3_vl_multimodal_routing(self):
        self.assertTrue(_uses_qwen3_vl_video_metadata("dstar_vl"))
        self.assertTrue(_uses_qwen3_vl_style_video_grid("dstar_vl"))
        self.assertTrue(_uses_qwen3_vl_style_t_index("dstar_vl"))
        self.assertFalse(_uses_qwen3_vl_video_metadata("qwen2_vl"))

    def test_qwen_vl_processor_passes_fallback_image_grid_thw_to_mrope(self):
        processor = object.__new__(QwenVLImageProcessor)
        processor.hf_config = SimpleNamespace(
            model_type="dstar_vl",
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )
        processor.model_type = "dstar_vl"
        processor.mm_tokens = SimpleNamespace(
            image_token_id=151655,
            video_token_id=151656,
            audio_token_id=None,
        )
        processor.vision_start_token_id = 151652
        processor.vision_end_token_id = 151653
        processor.audio_start_token_id = None
        processor.video_config = {}

        processor.load_mm_data = lambda **kwargs: SimpleNamespace(videos=[])
        processor.process_and_combine_mm_data = (
            lambda *args, **kwargs: (
                [],
                torch.tensor([[151652, 151655, 151653]], dtype=torch.long),
                SimpleNamespace(),
            )
        )

        captured = {}

        def fake_get_rope_index(**kwargs):
            captured.update(kwargs)
            seq_len = kwargs["input_ids"].shape[1]
            return (
                torch.zeros((3, 1, seq_len), dtype=torch.long),
                torch.zeros((1, 1), dtype=torch.long),
            )

        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
        request_obj = SimpleNamespace(video_data=[], audio_data=None, rid="test-rid")

        with patch(
            "sglang.srt.multimodal.processors.qwen_vl.MRotaryEmbedding.get_rope_index",
            side_effect=fake_get_rope_index,
        ):
            result = asyncio.run(
                processor.process_mm_data_async(
                    image_data=[{"image_grid_thw": image_grid_thw}],
                    input_text="ignored",
                    request_obj=request_obj,
                )
            )

        self.assertTrue(torch.equal(captured["image_grid_thw"], image_grid_thw))
        self.assertEqual(tuple(result["mrope_positions"].shape), (3, 3))


if __name__ == "__main__":
    unittest.main()
