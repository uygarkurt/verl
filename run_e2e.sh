#!/usr/bin/env bash
# GSM8K e2e demo for ref_reset_freq: 10 steps on Qwen2.5-0.5B-Instruct, 2 GPUs.
set -euo pipefail

python3 gsm8k.py --local_dir "$HOME/data/gsm8k"

CKPT_DIR="${CKPT_DIR:-$HOME/ref_reset_e2e_ckpts}"
rm -rf "$CKPT_DIR"

VLLM_DISABLE_FLASHINFER=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
python3 -m verl.trainer.main_ppo \
    data.train_files="$HOME/data/gsm8k/train.parquet" \
    data.val_files="$HOME/data/gsm8k/test.parquet" \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.adv_estimator=grpo \
    trainer.use_v1=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.logger='["console"]' \
    trainer.save_freq=4 \
    trainer.ref_reset_freq="${REF_RESET_FREQ:-4}" \
    trainer.total_training_steps=10 \
    trainer.total_epochs=1 \
    trainer.test_freq=-1 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.project_name=hiring-validation \
    trainer.experiment_name=ref-reset-e2e
