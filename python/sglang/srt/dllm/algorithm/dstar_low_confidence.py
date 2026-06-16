from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class _DstarLowConfidenceBase(DllmAlgorithm):
    @staticmethod
    def _get_return_flags(
        forward_batch: ForwardBatch, batch_size: int
    ) -> Tuple[List[bool], List[bool]]:
        return_step_map = getattr(forward_batch, "return_step_map", None)
        return_step_confidence_map = getattr(
            forward_batch, "return_step_confidence_map", None
        )
        if return_step_map is None:
            return_step_map = [False] * batch_size
        if return_step_confidence_map is None:
            return_step_confidence_map = [False] * batch_size
        return return_step_map, return_step_confidence_map

    @staticmethod
    def _compute_start_list(
        batch_size: int, input_ids: torch.Tensor, mask_id: int, block_size: int
    ) -> List[int]:
        start_list = []
        for block_id in range(batch_size):
            block_start = block_id * block_size
            block_end = block_start + block_size
            block_input_ids = input_ids[block_start:block_end]
            block_mask_index = block_input_ids == mask_id
            start = block_size - torch.sum(block_mask_index).item()
            start_list.append(start)
        return start_list

    @staticmethod
    def _finalize_next_token_ids(
        forward_batch: ForwardBatch, batch_size: int, start_list: List[int]
    ) -> List[torch.Tensor]:
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        return [next_token_ids[i, start_list[i] :] for i in range(batch_size)]

    @staticmethod
    def _init_step_maps(batch_size: int, block_size: int) -> List[List[int]]:
        return [[0] * block_size for _ in range(batch_size)]

    @staticmethod
    def _mark_step(
        step_maps: List[List[int]],
        batch_id: int,
        transfer_index: torch.Tensor,
        step: int,
    ) -> None:
        for token_idx in transfer_index.nonzero(as_tuple=False).flatten().tolist():
            if step_maps[batch_id][token_idx] == 0:
                step_maps[batch_id][token_idx] = step

    @staticmethod
    def _finalize_step_maps(
        step_maps: List[List[int]], start_list: List[int]
    ) -> List[List[int]]:
        return [step_maps[i][start_list[i] :] for i in range(len(step_maps))]

    @staticmethod
    def _init_step_confidence_maps(
        batch_size: int, block_size: int
    ) -> List[List[float]]:
        return [[0.0] * block_size for _ in range(batch_size)]

    @staticmethod
    def _mark_step_confidence(
        step_confidence_maps: List[List[float]],
        batch_id: int,
        transfer_index: torch.Tensor,
        confidence: torch.Tensor,
    ) -> None:
        for token_idx in transfer_index.nonzero(as_tuple=False).flatten().tolist():
            step_confidence_maps[batch_id][token_idx] = float(confidence[token_idx].item())

    @staticmethod
    def _finalize_step_confidence_maps(
        step_confidence_maps: List[List[float]], start_list: List[int]
    ) -> List[List[float]]:
        return [
            step_confidence_maps[i][start_list[i] :]
            for i in range(len(step_confidence_maps))
        ]


class low_confidence_static(_DstarLowConfidenceBase):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.denoising_steps = config.algorithm_config.get(
            "denoising_steps", self.block_size
        )

    @staticmethod
    def _get_num_transfer_tokens(
        mask_index: torch.Tensor, steps: int
    ) -> torch.Tensor:
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(
            mask_num.size(0),
            steps,
            device=mask_index.device,
            dtype=torch.int64,
        ) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1

        return num_transfer_tokens

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        mask_index = forward_batch.input_ids == self.mask_id

        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

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
        num_transfer_tokens = self._get_num_transfer_tokens(
            torch.reshape(mask_index, (batch_size, self.block_size)),
            self.denoising_steps,
        )

        for step in range(self.denoising_steps):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size

            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue

                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end
                ]
                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)

                k = min(
                    num_transfer_tokens[batch_id, step].item(),
                    int(block_mask_index.sum().item()),
                )
                if k <= 0:
                    continue

                transfer_index = torch.zeros_like(block_mask_index)
                _, select_index = torch.topk(confidence, k=k)
                transfer_index[select_index] = True

                block_input_ids[transfer_index] = x[transfer_index]
                if record_step_map and return_step_map[batch_id]:
                    self._mark_step(step_maps, batch_id, transfer_index, step + 1)
                if record_step_confidence_map and return_step_confidence_map[batch_id]:
                    self._mark_step_confidence(
                        step_confidence_maps, batch_id, transfer_index, confidence
                    )

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        next_token_ids_list = self._finalize_next_token_ids(
            forward_batch, batch_size, start_list
        )
        if not record_step_map and not record_step_confidence_map:
            return out.logits_output, next_token_ids_list, out.can_run_graph
        step_map_list = (
            self._finalize_step_maps(step_maps, start_list) if record_step_map else None
        )
        step_confidence_map_list = (
            self._finalize_step_confidence_maps(step_confidence_maps, start_list)
            if record_step_confidence_map
            else None
        )
        return (
            out.logits_output,
            next_token_ids_list,
            out.can_run_graph,
            step_map_list,
            step_confidence_map_list,
        )


class low_confidence_dynamic(_DstarLowConfidenceBase):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.denoising_steps = config.algorithm_config.get(
            "denoising_steps", self.block_size
        )
        self.pad_target_penalty = config.algorithm_config.get(
            "pad_target_penalty", 1.0
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        mask_index = forward_batch.input_ids == self.mask_id

        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

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

        for step in range(self.denoising_steps):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size

            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue

                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end
                ]
                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits.to(torch.float64), dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                if self.pad_target_penalty != 1.0:
                    p = torch.where(
                        x == self.mask_id,
                        p / self.pad_target_penalty,
                        p,
                    )
                confidence = torch.where(block_mask_index, p, -np.inf)

                transfer_index = confidence >= self.threshold
                if transfer_index.sum().item() == 0:
                    transfer_index = torch.zeros_like(block_mask_index)
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                block_input_ids[transfer_index] = x[transfer_index]
                if record_step_map and return_step_map[batch_id]:
                    self._mark_step(step_maps, batch_id, transfer_index, step + 1)
                if record_step_confidence_map and return_step_confidence_map[batch_id]:
                    self._mark_step_confidence(
                        step_confidence_maps, batch_id, transfer_index, confidence
                    )

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        next_token_ids_list = self._finalize_next_token_ids(
            forward_batch, batch_size, start_list
        )
        if not record_step_map and not record_step_confidence_map:
            return out.logits_output, next_token_ids_list, out.can_run_graph
        step_map_list = (
            self._finalize_step_maps(step_maps, start_list) if record_step_map else None
        )
        step_confidence_map_list = (
            self._finalize_step_confidence_maps(step_confidence_maps, start_list)
            if record_step_confidence_map
            else None
        )
        return (
            out.logits_output,
            next_token_ids_list,
            out.can_run_graph,
            step_map_list,
            step_confidence_map_list,
        )


ALGORITHMS = [low_confidence_static, low_confidence_dynamic]
