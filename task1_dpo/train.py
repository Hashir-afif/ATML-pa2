from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import (
    count_parameters,
    load_policy,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prompt_fits(tokenizer, row: dict, max_length: int) -> bool:
    """Same condition under which `encode_prompt_response` refuses an example."""
    ids = tokenizer.apply_chat_template(
        prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True
    )
    return len(ids) < max_length


def filter_overlong_prompts(tokenizer, rows: list[dict], max_length: int, policy: str):
    """Apply the configured policy for pairs whose prompt alone exceeds max_length.

    Returns (kept_rows, dropped_ids). The rule depends only on the tokenizer and the fixed
    file, so the same pairs are dropped for every condition that uses the same file.
    """
    if policy == "error":
        return rows, []
    if policy != "skip":
        raise ValueError(f"Unknown overlong_prompt_policy={policy!r}")
    kept, dropped = [], []
    for row in rows:
        if prompt_fits(tokenizer, row, max_length):
            kept.append(row)
        else:
            dropped.append(str(row.get("source_index")))
    return kept, dropped


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    max_length = int(cfg["max_sequence_length"])
    rows, dropped = filter_overlong_prompts(
        tokenizer, rows, max_length, cfg.get("overlong_prompt_policy", "error")
    )
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=make_collate(tokenizer, max_length),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "dataset_path": path,
        "rows": rows,
        "dropped_overlong": dropped,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def pair_logprobs(model, chosen: dict, rejected: dict, with_grad: bool):
    """Summed response-token log-probs for chosen/rejected under the policy and the reference.

    The reference is the same base model with the LoRA adapter disabled (frozen initial policy).
    """
    with torch.no_grad(), reference_mode(model):
        ref_c, _, _ = response_sequence_logprobs(model, chosen)
        ref_r, _, _ = response_sequence_logprobs(model, rejected)
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        pol_c, _, mask_c = response_sequence_logprobs(model, chosen)
        pol_r, _, mask_r = response_sequence_logprobs(model, rejected)
    return pol_c, pol_r, ref_c, ref_r, mask_c.sum(-1), mask_r.sum(-1)


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None, max_updates: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model, loader, optimizer = bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = repo_path(cfg["results_dir"]) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()  # a rerun of the same run name replaces its own training log

    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg["epochs"])
    max_grad_norm = float(cfg["max_grad_norm"])
    micro_per_epoch = len(loader)
    updates_per_epoch = math.ceil(micro_per_epoch / accum)
    total_updates = updates_per_epoch * epochs
    if max_updates is not None:
        total_updates = min(total_updates, int(max_updates))
    device = next(model.parameters()).device
    params = trainable_parameters(model)
    n_total, n_trainable = count_parameters(model)

    print(f"[{run_name}] pairs={len(bundle['rows'])} dropped_overlong={len(bundle['dropped_overlong'])} "
          f"beta={beta} micro_batches/epoch={micro_per_epoch} updates={total_updates}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    update = 0
    history = []
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        micro_in_group, group_stats = 0, []
        for micro_idx, (chosen, rejected) in enumerate(loader):
            # The final accumulation group of an epoch may be shorter; scale by its true size.
            group_start = (micro_idx // accum) * accum
            group_size = min(accum, micro_per_epoch - group_start)
            chosen, rejected = _to_device(chosen, device), _to_device(rejected, device)
            pol_c, pol_r, ref_c, ref_r, len_c, len_r = pair_logprobs(model, chosen, rejected, with_grad=True)
            loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            (loss / group_size).backward()
            group_stats.append({
                "loss": float(loss.item()),
                "accuracy": float(diag["preference_accuracy"].item()),
                "margin": float(((pol_c - ref_c) - (pol_r - ref_r)).mean().item()),
                "chosen_logratio": float((pol_c - ref_c).mean().item()),
                "rejected_logratio": float((pol_r - ref_r).mean().item()),
                "chosen_tokens": float(len_c.float().mean().item()),
                "rejected_tokens": float(len_r.float().mean().item()),
            })
            micro_in_group += 1
            if micro_in_group < group_size:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            record = {"update": update, "epoch": epoch, "micro_batches_in_update": len(group_stats)}
            for key in group_stats[0]:
                record[key] = sum(s[key] for s in group_stats) / len(group_stats)
            record["grad_norm"] = float(grad_norm.item())
            record["lr"] = float(optimizer.param_groups[0]["lr"])
            record["elapsed_s"] = elapsed()
            if torch.cuda.is_available():
                record["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
            append_jsonl(log_path, record)
            history.append(record)
            if update % 5 == 0 or update == total_updates:
                print(f"[{run_name}] update {update}/{total_updates} loss={record['loss']:.4f} "
                      f"acc={record['accuracy']:.3f} margin={record['margin']:.3f} "
                      f"gnorm={record['grad_norm']:.3f} t={record['elapsed_s']:.0f}s")
            micro_in_group, group_stats = 0, []
            if update >= total_updates:
                break
        if update >= total_updates:
            break

    model.save_pretrained(str(output))
    run_info = {
        "run_name": run_name,
        "status": "COMPLETED" if max_updates is None else "SMOKE_TEST",
        "config_path": config_path,
        "config": cfg,
        "seed": int(cfg["seed"]),
        "beta": beta,
        "dataset_path": bundle["dataset_path"],
        "max_examples": max_examples,
        "num_pairs_used": len(bundle["rows"]),
        "used_source_indices": [str(r.get("source_index")) for r in bundle["rows"]],
        "dropped_overlong_source_indices": bundle["dropped_overlong"],
        "overlong_prompt_policy": cfg.get("overlong_prompt_policy", "error"),
        "optimizer": {"name": "AdamW", "lr": float(cfg["learning_rate"]),
                      "weight_decay": float(cfg.get("weight_decay", 0.0)), "schedule": "constant"},
        "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": accum,
        "effective_batch_pairs": int(cfg["batch_size"]) * accum,
        "epochs": epochs,
        "updates": update,
        "max_grad_norm": max_grad_norm,
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "lora": cfg["lora"],
        "dtype": cfg.get("dtype"),
        "parameters_total": n_total,
        "parameters_trainable": n_trainable,
        "adapter_path": str(Path(output).relative_to(repo_path("."))).replace("\\", "/"),
        "wall_clock_s": elapsed(),
        "peak_vram_gib": (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else None,
        "train_log": str(log_path.relative_to(repo_path("."))).replace("\\", "/"),
    }
    save_json(run_dir / "run_info.json", run_info)
    save_json(Path(output) / "run_info.json", run_info)
    print(f"[{run_name}] saved adapter -> {output}  ({run_info['wall_clock_s']:.0f}s, "
          f"peak {run_info['peak_vram_gib'] or 0:.2f} GiB)")
    return {"model": model, "tokenizer": bundle["tokenizer"], "run_info": run_info, "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--max-updates", type=int, help="smoke tests only; full runs use the epoch budget")
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples, args.max_updates)


if __name__ == "__main__":
    main()
