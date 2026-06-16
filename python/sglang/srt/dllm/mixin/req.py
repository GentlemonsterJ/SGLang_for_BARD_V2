from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def _use_raw_dllm_req_flow(self: Req) -> bool:
        return self.dllm_config is not None and self.dllm_config.arch in {
            "SDARForCausalLM",
            "SDARMoeForCausalLM",
        }

    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config

        if self.dllm_config is not None:
            if self._use_raw_dllm_req_flow():
                if len(self.origin_input_ids) < self.dllm_config.block_size:
                    self.dllm_phase = DllmReqPhase.INCOMING_DECODE
                else:
                    self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
            else:
                if len(self.origin_input_ids) > 0:
                    self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
                else:
                    self.dllm_phase = DllmReqPhase.INCOMING_DECODE

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        if self._use_raw_dllm_req_flow():
            prefix_length = len(self.prefix_indices)
            min_required_length = prefix_length + self.dllm_config.block_size

            if len(self.fill_ids) < min_required_length:
                return

            input_block = self.fill_ids[prefix_length:min_required_length]
            is_prefill_phase = self.dllm_config.mask_id not in input_block

            if is_prefill_phase:
                self.dllm_phase = DllmReqPhase.STAGING_PREFILL
            else:
                self.dllm_phase = DllmReqPhase.STAGING_DECODE
            return

        prefix_length = len(self.prefix_indices)
        filled_length = len(self.origin_input_ids) + len(self.output_ids)

        if prefix_length < filled_length:
            if prefix_length == 0:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
            else:
                self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            if prefix_length == 0:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if self._use_raw_dllm_req_flow():
            self.dllm_block_offset = (
                0
                if not self.fill_ids
                else self.dllm_block_offset + self.dllm_config.block_size
            )
            self.fill_ids = (
                self.origin_input_ids
                + self.output_ids
                + [self.dllm_config.mask_id] * self.dllm_config.block_size
            )
            return

        self.fill_ids = self.origin_input_ids + self.output_ids
        if self.dllm_phase in [
            DllmReqPhase.INCOMING_DECODE,
            DllmReqPhase.STAGING_DECODE,
        ]:
            self.fill_ids = self.fill_ids + [
                self.dllm_config.mask_id
            ] * self.dllm_config.block_size

    def _update_block_offset_for_dllm(self):
        prefix_len = len(self.prefix_indices)
        if self._use_raw_dllm_req_flow():
            assert (
                prefix_len % self.dllm_config.block_size == 0
            ), f"Unexpected prefix len: {prefix_len}"
            if prefix_len > self.dllm_block_offset:
                self.dllm_block_offset = prefix_len
            return

        if prefix_len > self.dllm_block_offset:
            self.dllm_block_offset = prefix_len
