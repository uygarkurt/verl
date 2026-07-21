# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Behavioral test for periodic reference-policy reset.

This test is the specification for the take-home assignment. It FAILS on stock
verl at the pinned commit; your implementation must make it pass.

Contract under test:
  - ``ActorRolloutRefWorker`` exposes ``reset_ref_policy(local_path)`` where
    ``local_path`` points to the actor checkpoint saved at the current step.
    After the call, the reference policy's log-probs must match the current
    actor's log-probs. Only this behavior is asserted.

Requires 2 GPUs (one for the actor worker group, one for the standalone ref
worker group). Run with:

    python -m pytest tests/models/test_ref_reset.py -x -s
"""

import os
from functools import partial

import pytest
import ray
import torch
from hydra import compose, initialize_config_dir
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

import verl
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.workers.config import ActorConfig
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

VERL_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl.__file__)), "trainer", "config")

BATCH_SIZE = 8
SEQLEN = 32
RESPONSE_LENGTH = SEQLEN // 2
TRAIN_STEPS = 3


def _make_tiny_model(tmp_path) -> str:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    model = AutoModelForCausalLM.from_config(config)
    path = os.path.join(str(tmp_path), "tiny_model")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    config.save_pretrained(path)
    return path


def _compose_worker_config(model_path: str):
    with initialize_config_dir(config_dir=VERL_CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                f"actor_rollout_ref.model.path={model_path}",
                "actor_rollout_ref.actor.strategy=fsdp2",
                "actor_rollout_ref.actor.optim.lr=1e-2",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={BATCH_SIZE}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={BATCH_SIZE}",
                "actor_rollout_ref.actor.use_dynamic_bsz=false",
                f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={BATCH_SIZE}",
                "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false",
                f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={BATCH_SIZE}",
                "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false",
            ],
        )
    return cfg.actor_rollout_ref


def _make_worker_group(arr_cfg, role: str) -> RayWorkerGroup:
    ray_cls = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=arr_cfg, role=role)
    resource_pool = RayResourcePool(process_on_nodes=[1], max_colocate_count=1)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls)
    wg.init_model()
    return wg


def _make_batch(vocab_size: int) -> DataProto:
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQLEN))
    attention_mask = create_random_mask(
        input_ids=input_ids,
        max_ratio_of_valid_token=0.8,
        max_ratio_of_left_padding=0.2,
        min_ratio_of_valid_token=0.6,
    )
    position_ids = compute_position_id_with_mask(attention_mask)
    global_token_num = torch.sum(attention_mask, dim=-1).tolist()
    return DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :RESPONSE_LENGTH],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, RESPONSE_LENGTH:],
            "response_mask": attention_mask[:, RESPONSE_LENGTH:],
        },
        meta_info={"temperature": 1.0, "global_token_num": global_token_num, "compute_loss": False},
    )


def _log_probs(wg: RayWorkerGroup, data: DataProto, method: str) -> torch.Tensor:
    batch_td = data.to_tensordict()
    batch_td = left_right_2_no_padding(batch_td)
    tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
    output = getattr(wg, method)(batch_td)
    log_probs = tu.get(output, "log_probs")
    return no_padding_2_padding(log_probs, batch_td).float().cpu()


def _train_steps(wg: RayWorkerGroup, data: DataProto, num_steps: int) -> None:
    old_log_probs = _log_probs(wg, data, "compute_log_prob")
    train_data = DataProto.from_single_dict(
        {
            **{k: v for k, v in data.batch.items()},
            "old_log_probs": old_log_probs,
            "advantages": torch.rand(BATCH_SIZE, RESPONSE_LENGTH, dtype=torch.float32),
            "ref_log_prob": torch.rand(BATCH_SIZE, RESPONSE_LENGTH, dtype=torch.float32),
        },
        meta_info=dict(data.meta_info),
    )
    for _ in range(num_steps):
        batch_td = train_data.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, global_batch_size=BATCH_SIZE, mini_batch_size=BATCH_SIZE)
        wg.update_actor(batch_td)


def test_ref_reset(tmp_path):
    ray.init()
    try:
        if ray.cluster_resources().get("GPU", 0) < 2:
            pytest.skip("requires 2 GPUs (actor + standalone ref)")
        model_path = _make_tiny_model(tmp_path)
        arr_cfg = _compose_worker_config(model_path)

        actor_wg = _make_worker_group(arr_cfg, role="actor")
        ref_wg = _make_worker_group(arr_cfg, role="ref")
        actor_config = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=-1)
        actor_wg.set_loss_fn(partial(ppo_loss, config=actor_config))

        vocab_size = len(AutoTokenizer.from_pretrained(model_path))
        data = _make_batch(vocab_size)

        # 1. freshly initialized, actor and ref share weights
        actor_lp_0 = _log_probs(actor_wg, data, "compute_log_prob")
        ref_lp_0 = _log_probs(ref_wg, data, "compute_ref_log_prob")
        torch.testing.assert_close(actor_lp_0, ref_lp_0, atol=1e-3, rtol=1e-2)

        # 2. after training, the actor must have moved away from the ref
        _train_steps(actor_wg, data, TRAIN_STEPS)
        actor_lp_1 = _log_probs(actor_wg, data, "compute_log_prob")
        ref_lp_1 = _log_probs(ref_wg, data, "compute_ref_log_prob")
        divergence = (actor_lp_1 - ref_lp_1).abs().max().item()
        assert divergence > 5e-3, f"actor did not diverge from ref (max diff {divergence}); test setup broken"

        # 3. save the actor checkpoint, as the trainer does every save_freq steps
        ckpt_path = os.path.join(str(tmp_path), f"global_step_{TRAIN_STEPS}", "actor")
        actor_wg.save_checkpoint(ckpt_path, None, TRAIN_STEPS)

        # 4. THE FEATURE: reset the ref policy to the current actor
        ref_wg.reset_ref_policy(local_path=ckpt_path)

        # 5. ref now matches the trained actor, and has moved from its old state
        ref_lp_2 = _log_probs(ref_wg, data, "compute_ref_log_prob")
        torch.testing.assert_close(ref_lp_2, actor_lp_1, atol=1e-3, rtol=1e-2)
        reset_delta = (ref_lp_2 - ref_lp_1).abs().max().item()
        assert reset_delta > 5e-3, f"ref did not move on reset (max diff {reset_delta})"
    finally:
        ray.shutdown()
