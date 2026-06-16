import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.sdar import _normalize_sdar_rope_positions_dtype


class TestSdarRopePositions(unittest.TestCase):
    def test_aligns_positions_dtype_with_out_cache_loc(self):
        positions = torch.arange(4, dtype=torch.int64)
        forward_batch = SimpleNamespace(out_cache_loc=torch.arange(4, dtype=torch.int32))

        normalized = _normalize_sdar_rope_positions_dtype(positions, forward_batch)

        self.assertEqual(normalized.dtype, torch.int32)
        self.assertTrue(torch.equal(normalized, positions.to(torch.int32)))

    def test_keeps_positions_when_dtype_already_matches(self):
        positions = torch.arange(4, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=torch.arange(4, dtype=torch.int32))

        normalized = _normalize_sdar_rope_positions_dtype(positions, forward_batch)

        self.assertIs(normalized, positions)


if __name__ == "__main__":
    unittest.main()
