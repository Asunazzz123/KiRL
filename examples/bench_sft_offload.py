"""Compare native AdamW and ZeRO-3 optimizer placement on fixed-length SFT data."""
import argparse
import json
import time

import torch
from transformers import AutoConfig, TrainingArguments, set_seed

from kirl.algorithms.sft import SFTTrainer
from kirl.datasets import BaseDataset
from kirl.engine.optim import CPUOffloadAdamW


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cpu", action="store_true", help="Run the native smoke test on CPU")
    parser.add_argument("--backend", choices=["native", "deepspeed"], default="native")
    parser.add_argument("--cpu-ratio", type=float)
    parser.add_argument("--sub-group-size", type=int, default=1000000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--output-dir", default="outputs/offload")
    args = parser.parse_args()
    if args.cpu_ratio is not None and not 0 <= args.cpu_ratio <= 1:
        parser.error("--cpu-ratio must be between 0 and 1")
    if args.sub_group_size < 1 or args.steps < 1 or args.length < 2:
        parser.error("invalid subgroup size, steps or sequence length")
    zero = {"stage": 3, "sub_group_size": args.sub_group_size}
    opts = {}
    if args.cpu_ratio is not None:
        zero["offload_optimizer"] = dict(device="cpu", ratio=args.cpu_ratio, pin_memory=True)
        if args.backend == "native":
            opts["optimizer_cls_and_kwargs"] = (CPUOffloadAdamW, dict(
                lr=1e-5, cpu_ratio=args.cpu_ratio, sub_group_size=args.sub_group_size))
    ds = dict(zero_optimization=zero, optimizer=dict(type="AdamW", params=dict(
        lr="auto", weight_decay="auto", eps="auto", betas="auto")),
        train_micro_batch_size_per_gpu="auto", gradient_accumulation_steps="auto",
        train_batch_size="auto", gradient_clipping="auto", bf16={"enabled": "auto"})
    training = TrainingArguments(output_dir=args.output_dir, use_cpu=args.cpu, deepspeed=ds if args.backend == "deepspeed" else None,
        max_steps=args.steps, per_device_train_batch_size=1, learning_rate=1e-5,
        lr_scheduler_type="constant", optim="adamw_torch", bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        save_strategy="no", report_to="none", seed=19)
    set_seed(19)
    vocab = AutoConfig.from_pretrained(args.model).vocab_size
    rows = []
    for _ in range(args.steps * training.world_size):
        ids = torch.randint(vocab, (args.length,)).tolist()
        rows.append(dict(input_ids=ids, labels=ids.copy(), attention_mask=[1]*args.length))
    trainer = SFTTrainer(args.model, BaseDataset(rows), args=training, **opts)
    trainer.accelerator.wait_for_everyone()
    start = time.perf_counter()
    trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    trainer.accelerator.wait_for_everyone()
    elapsed = time.perf_counter() - start
    if trainer.is_world_process_zero():
        print(json.dumps(dict(backend=args.backend, requested_cpu_ratio=args.cpu_ratio,
            end_to_end_seconds=elapsed, target_tokens_per_second=len(rows)*(args.length-1)/elapsed)))


if __name__ == "__main__":
    main()
