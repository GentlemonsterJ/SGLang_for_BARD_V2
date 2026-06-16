from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.dstar_low_confidence import _DstarLowConfidenceBase
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class Dflash(_DstarLowConfidenceBase):
    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.denoising_steps = max(
            config.algorithm_config.get("denoising_steps", self.block_size),
            self.block_size,
        )
        self.pad_target_penalty = config.algorithm_config.get(
            "pad_target_penalty", 1.0
        )

    @staticmethod
    def _init_verify_accept_maps(
        batch_size: int, block_size: int
    ) -> List[List[bool]]:
        return [[False] * block_size for _ in range(batch_size)]

    @staticmethod
    def _mark_verify_accept(
        verify_accept_maps: List[List[bool]],
        batch_id: int,
        transfer_index: torch.Tensor,
    ) -> None:
        for token_idx in transfer_index.nonzero(as_tuple=False).flatten().tolist():
            verify_accept_maps[batch_id][token_idx] = True

    @staticmethod
    def _finalize_verify_accept_maps(
        verify_accept_maps: List[List[bool]], start_list: List[int]
    ) -> List[List[bool]]:
        return [
            verify_accept_maps[i][start_list[i] :]
            for i in range(len(verify_accept_maps))
        ]

    @staticmethod
    def _init_verify_logprob_maps(
        batch_size: int, block_size: int
    ) -> List[List[float]]:
        return [[0.0] * block_size for _ in range(batch_size)]

    @staticmethod
    def _mark_verify_logprob(
        verify_logprob_maps: List[List[float]],
        batch_id: int,
        transfer_index: torch.Tensor,
        verifier_scores: torch.Tensor,
    ) -> None:
        for token_idx in transfer_index.nonzero(as_tuple=False).flatten().tolist():
            verify_logprob_maps[batch_id][token_idx] = float(
                verifier_scores[token_idx].item()
            )

    @staticmethod
    def _finalize_verify_logprob_maps(
        verify_logprob_maps: List[List[float]], start_list: List[int]
    ) -> List[List[float]]:
        return [
            verify_logprob_maps[i][start_list[i] :]
            for i in range(len(verify_logprob_maps))
        ]

    def _build_transfer_index(
        self,
        block_mask_index: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        valid_candidate_mask = block_mask_index & candidate_token_ids.ne(self.mask_id)
        selection_confidence = torch.where(
            valid_candidate_mask,
            confidence,
            torch.full_like(confidence, -np.inf),
        )

        transfer_index = selection_confidence >= self.threshold
        if transfer_index.sum().item() > 0:
            return transfer_index

        fallback_mask = valid_candidate_mask
        fallback_confidence = selection_confidence
        if fallback_mask.sum().item() == 0:
            fallback_mask = block_mask_index
            fallback_confidence = torch.where(
                block_mask_index,
                confidence,
                torch.full_like(confidence, -np.inf),
            )

        transfer_index = torch.zeros_like(block_mask_index)
        _, select_index = torch.topk(fallback_confidence, k=1)
        transfer_index[select_index] = True
        return transfer_index

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        *,
        model_worker_batch: ModelWorkerBatch | None = None,
        verifier=None,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor],
        List[torch.Tensor],
        bool,
        List[List[int]],
        List[List[float]],
        List[List[bool]],
        List[List[float]],
    ]:
        if verifier is None:
            raise RuntimeError("Dflash requires a configured local verifier.")
        if model_worker_batch is None or model_worker_batch.reqs is None:
            raise RuntimeError("Dflash requires access to the current model worker batch.")

        batch_size = forward_batch.batch_size
        mask_index = forward_batch.input_ids == self.mask_id

        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph, None, None, None, None

        start_list = self._compute_start_list(
            batch_size, forward_batch.input_ids, self.mask_id, self.block_size
        )
        return_step_map, return_step_confidence_map = self._get_return_flags(
            forward_batch, batch_size
        )
        record_step_map = any(return_step_map)
        record_step_confidence_map = any(return_step_confidence_map)
        step_maps = (
            self._init_step_maps(batch_size, self.block_size) if record_step_map else None
        )
        step_confidence_maps = (
            self._init_step_confidence_maps(batch_size, self.block_size)
            if record_step_confidence_map
            else None
        )
        verify_accept_maps = self._init_verify_accept_maps(batch_size, self.block_size)
        verify_logprob_maps = self._init_verify_logprob_maps(batch_size, self.block_size)

        can_run_cuda_graph = False
        for step in range(self.denoising_steps):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size

            transfer_indices = []
            confidence_maps = []

            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    transfer_indices.append(torch.zeros_like(block_mask_index))
                    confidence_maps.append(
                        torch.full_like(
                            block_input_ids,
                            -np.inf,
                            dtype=torch.float32,
                        )
                    )
                    continue

                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end
                ]
                topk_index = torch.topk(curr_logits, k=2, dim=-1).indices
                candidate_token_ids = topk_index[:, 0]
                candidate_token_ids = torch.where(
                    block_mask_index & candidate_token_ids.eq(self.mask_id),
                    topk_index[:, 1],
                    candidate_token_ids,
                )
                token_probs = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits.to(torch.float64), dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(candidate_token_ids, -1),
                    ),
                    -1,
                )
                candidate_token_ids = torch.where(
                    block_mask_index,
                    candidate_token_ids,
                    block_input_ids,
                )
                if self.pad_target_penalty != 1.0:
                    token_probs = torch.where(
                        candidate_token_ids == self.mask_id,
                        token_probs / self.pad_target_penalty,
                        token_probs,
                    )

                confidence = torch.where(
                    block_mask_index,
                    token_probs,
                    torch.full_like(token_probs, -np.inf),
                ).float()
                transfer_index = self._build_transfer_index(
                    block_mask_index,
                    candidate_token_ids,
                    confidence,
                )
                block_input_ids[transfer_index] = candidate_token_ids[transfer_index]

                transfer_indices.append(transfer_index)
                confidence_maps.append(confidence)

            candidate_blocks = [
                forward_batch.input_ids[
                    batch_id * self.block_size : (batch_id + 1) * self.block_size
                ].clone()
                for batch_id in range(batch_size)
            ]
            verify_accept_masks, verify_score_maps = verifier.score_proposals(
                source_reqs=model_worker_batch.reqs,
                multimodal_inputs=model_worker_batch.multimodal_inputs,
                block_offsets=model_worker_batch.dllm_block_offsets,
                candidate_blocks=candidate_blocks,
                transfer_masks=transfer_indices,
                mask_id=self.mask_id,
            )

            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end
                ]
                accepted_index = verify_accept_masks[batch_id]
                rejected_index = transfer_indices[batch_id] & (~accepted_index)
                block_input_ids[rejected_index] = self.mask_id

                if record_step_map and return_step_map[batch_id]:
                    self._mark_step(step_maps, batch_id, accepted_index, step + 1)
                if record_step_confidence_map and return_step_confidence_map[batch_id]:
                    self._mark_step_confidence(
                        step_confidence_maps,
                        batch_id,
                        accepted_index,
                        confidence_maps[batch_id],
                    )

                if accepted_index.any():
                    self._mark_verify_accept(
                        verify_accept_maps,
                        batch_id,
                        accepted_index,
                    )
                    self._mark_verify_logprob(
                        verify_logprob_maps,
                        batch_id,
                        accepted_index,
                        verify_score_maps[batch_id],
                    )

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        next_token_ids_list = self._finalize_next_token_ids(
            forward_batch, batch_size, start_list
        )
        step_map_list = (
            self._finalize_step_maps(step_maps, start_list) if record_step_map else None
        )
        step_confidence_map_list = (
            self._finalize_step_confidence_maps(step_confidence_maps, start_list)
            if record_step_confidence_map
            else None
        )
        verify_accept_map_list = self._finalize_verify_accept_maps(
            verify_accept_maps, start_list
        )
        verify_logprob_map_list = self._finalize_verify_logprob_maps(
            verify_logprob_maps, start_list
        )
        return (
            out.logits_output,
            next_token_ids_list,
            out.can_run_graph,
            step_map_list,
            step_confidence_map_list,
            verify_accept_map_list,
            verify_logprob_map_list,
        )


ALGORITHMS = [Dflash]
