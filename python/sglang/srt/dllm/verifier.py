from __future__ import annotations

import contextlib
import copy
import dataclasses
from typing import Iterable, List, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class _TemporaryVerifierTreeCache:
    def __init__(self, req_to_token_pool, token_to_kv_pool_allocator):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = token_to_kv_pool_allocator.page_size
        self.sliding_window_size = None

    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def evict(self, *args, **kwargs) -> None:
        return

    def pretty_print(self) -> None:
        return


@contextlib.contextmanager
def _server_args_context(server_args: ServerArgs):
    previous = get_global_server_args()
    set_global_server_args_for_scheduler(server_args)
    try:
        yield
    finally:
        set_global_server_args_for_scheduler(previous)


def _clone_sampling_params(sampling_params):
    cloned = copy.copy(sampling_params)
    custom_params = getattr(cloned, "custom_params", None)
    if isinstance(custom_params, dict):
        cloned.custom_params = dict(custom_params)
        cloned.custom_params.pop("__req__", None)
    return cloned


def _clone_mm_inputs_without_precomputed(mm_input):
    if mm_input is None:
        return None

    cloned = copy.copy(mm_input)
    cloned.mm_items = []
    for item in mm_input.mm_items:
        new_item = copy.copy(item)
        if getattr(new_item, "feature", None) is None and getattr(
            new_item, "precomputed_embeddings", None
        ) is not None:
            raise RuntimeError(
                "Verifier scoring requires raw multimodal features; "
                "precomputed multimodal embeddings are not supported."
            )
        new_item.precomputed_embeddings = None
        cloned.mm_items.append(new_item)
    return cloned


class DllmLocalVerifier:
    def __init__(
        self,
        server_args: ServerArgs,
        *,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        dp_rank: Optional[int],
        attn_cp_rank: Optional[int],
        moe_dp_rank: Optional[int],
    ):
        verifier_model_path = server_args.dllm_verifier_model_path
        if verifier_model_path is None:
            raise ValueError("dllm_verifier_model_path must be set for verifier init.")

        self.main_server_args = server_args
        self.server_args = dataclasses.replace(
            server_args,
            model_path=verifier_model_path,
            tokenizer_path=verifier_model_path,
            served_model_name=verifier_model_path,
            revision=server_args.dllm_verifier_revision,
            dllm_algorithm=None,
            dllm_algorithm_config=None,
            speculative_algorithm=None,
            speculative_draft_model_path=None,
            speculative_draft_model_revision=None,
            skip_tokenizer_init=True,
            enable_mm_global_cache=False,
        )
        self.model_config = ModelConfig.from_server_args(self.server_args)
        self.verify_threshold = server_args.dllm_verifier_threshold

        with _server_args_context(self.server_args):
            self.model_runner = ModelRunner(
                model_config=self.model_config,
                mem_fraction_static=self.server_args.mem_fraction_static,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                tp_size=tp_size,
                moe_ep_rank=moe_ep_rank,
                moe_ep_size=moe_ep_size,
                pp_rank=pp_rank,
                pp_size=pp_size,
                nccl_port=nccl_port,
                dp_rank=dp_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                server_args=self.server_args,
            )

        set_global_server_args_for_scheduler(self.main_server_args)
        self.tree_cache = _TemporaryVerifierTreeCache(
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def score_proposals(
        self,
        source_reqs: List[Req],
        multimodal_inputs,
        block_offsets: List[int],
        candidate_blocks: List[torch.Tensor],
        transfer_masks: List[torch.Tensor],
        mask_id: int,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        if not source_reqs:
            return [], []

        temp_batch = None
        temp_reqs = []
        try:
            with _server_args_context(self.server_args):
                temp_reqs = self._build_temp_reqs(
                    source_reqs,
                    multimodal_inputs,
                    block_offsets,
                    candidate_blocks,
                )
                temp_batch = ScheduleBatch.init_new(
                    reqs=temp_reqs,
                    req_to_token_pool=self.model_runner.req_to_token_pool,
                    token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
                    tree_cache=self.tree_cache,
                    model_config=self.model_config,
                    enable_overlap=False,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                )
                temp_batch.prepare_for_extend()
                model_worker_batch = temp_batch.get_model_worker_batch()
                forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
                output = self.model_runner.forward(forward_batch)
                input_token_logprobs = output.logits_output.input_token_logprobs
                if input_token_logprobs is None:
                    raise RuntimeError("Verifier forward did not produce input token logprobs.")

                seq_logprobs = torch.split(
                    input_token_logprobs,
                    model_worker_batch.extend_seq_lens,
                )
                return self._build_acceptance_maps(
                    seq_logprobs=seq_logprobs,
                    candidate_blocks=candidate_blocks,
                    transfer_masks=transfer_masks,
                    block_offsets=block_offsets,
                    mask_id=mask_id,
                )
        finally:
            if temp_batch is not None:
                self._release_temp_batch(temp_batch, temp_reqs)

    def _build_temp_reqs(
        self,
        source_reqs: List[Req],
        multimodal_inputs,
        block_offsets: List[int],
        candidate_blocks: List[torch.Tensor],
    ) -> List[Req]:
        temp_reqs = []
        for source_req, mm_input, block_offset, candidate_block in zip(
            source_reqs,
            multimodal_inputs,
            block_offsets,
            candidate_blocks,
        ):
            full_sequence = list(source_req.fill_ids)
            block_end = block_offset + candidate_block.numel()
            full_sequence[block_offset:block_end] = candidate_block.tolist()

            temp_req = Req(
                rid=f"{source_req.rid}::verifier",
                origin_input_text=source_req.origin_input_text,
                origin_input_ids=full_sequence,
                sampling_params=_clone_sampling_params(source_req.sampling_params),
                return_logprob=True,
                stream=False,
                lora_id=source_req.lora_id,
                token_type_ids=source_req.token_type_ids,
                require_reasoning=source_req.require_reasoning,
                vocab_size=self.model_config.vocab_size,
            )
            temp_req.multimodal_inputs = _clone_mm_inputs_without_precomputed(mm_input)
            temp_req.fill_ids = full_sequence
            temp_req.prefix_indices = torch.empty((0,), dtype=torch.int64)
            temp_req.extend_input_len = len(full_sequence)
            temp_req.extend_logprob_start_len = 0
            temp_req.logprob_start_len = 0
            temp_reqs.append(temp_req)
        return temp_reqs

    def _build_acceptance_maps(
        self,
        *,
        seq_logprobs: Iterable[torch.Tensor],
        candidate_blocks: List[torch.Tensor],
        transfer_masks: List[torch.Tensor],
        block_offsets: List[int],
        mask_id: int,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        accept_masks = []
        score_maps = []
        neg_inf = float("-inf")

        for logprobs, candidate_block, transfer_mask, block_offset in zip(
            seq_logprobs,
            candidate_blocks,
            transfer_masks,
            block_offsets,
        ):
            transfer_mask = transfer_mask.to(dtype=torch.bool)
            score_map = torch.full(
                candidate_block.shape,
                neg_inf,
                device=candidate_block.device,
                dtype=torch.float32,
            )

            for local_idx in transfer_mask.nonzero(as_tuple=False).flatten().tolist():
                absolute_pos = block_offset + local_idx
                if absolute_pos <= 0:
                    continue
                logprob_idx = absolute_pos - 1
                if logprob_idx >= logprobs.shape[0]:
                    continue
                score_map[local_idx] = logprobs[logprob_idx].float()

            valid_transfer_mask = transfer_mask & candidate_block.ne(mask_id)
            accept_mask = valid_transfer_mask & (score_map >= self.verify_threshold)

            if valid_transfer_mask.any() and not accept_mask.any():
                best_positions = valid_transfer_mask.nonzero(as_tuple=False).flatten()
                best_idx = best_positions[
                    torch.argmax(score_map[best_positions]).item()
                ].item()
                accept_mask[best_idx] = True

            accept_masks.append(accept_mask)
            score_maps.append(score_map)

        return accept_masks, score_maps

    def _release_temp_batch(self, temp_batch: ScheduleBatch, temp_reqs: List[Req]) -> None:
        indices_to_free = []
        if temp_batch.out_cache_loc is not None and temp_batch.out_cache_loc.numel() > 0:
            indices_to_free.append(temp_batch.out_cache_loc)
        if (
            temp_batch.encoder_out_cache_loc is not None
            and temp_batch.encoder_out_cache_loc.numel() > 0
        ):
            indices_to_free.append(temp_batch.encoder_out_cache_loc)
        if indices_to_free:
            self.model_runner.token_to_kv_pool_allocator.free(torch.cat(indices_to_free))

        for req in temp_reqs:
            if req.req_pool_idx is not None:
                self.model_runner.req_to_token_pool.free(req)
