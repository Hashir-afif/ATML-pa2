from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.eval_utils import generate_with_kl, length_stats
from common.generation import score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from task2_ppo.continue_train import filter_long_prompts


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def heldout_items(cfg: dict, tokenizer, limit: int | None = None):
    """The fixed held-out prompt set: rl_prompt_pool_eval in file order, minus over-long prompts."""
    rows, dropped = filter_long_prompts(tokenizer, read_jsonl(cfg["paths"]["rl_prompt_eval"]),
                                        int(cfg["max_prompt_length"]))
    if limit is not None:  # smoke tests only
        rows = rows[:limit]
    items = [{"messages": prompt_messages(r),
              "meta": {"item_key": f"eval:{r['source_index']}", "source_index": str(r["source_index"]),
                       "prompt_id": r["prompt_id"]}} for r in rows]
    return items, dropped


def add_rewards(records: list[dict], items: list[dict], cfg: dict, rm) -> None:
    """Learned reward (raw) plus the effective reward with the missing-EOS penalty used in training."""
    rm_model, rm_tok = rm
    bs = int(cfg["eval_batch_size"])
    by_key = {it["meta"]["item_key"]: it["messages"] for it in items}
    penalty = float(cfg["missing_eos_penalty"])
    for start in range(0, len(records), bs):
        chunk = records[start:start + bs]
        scores = score_reward_pairs(rm_model, rm_tok, [by_key[r["item_key"]] for r in chunk],
                                    [r["response"] for r in chunk], max_length=int(cfg["reward_max_length"]))
        for r, s in zip(chunk, scores.tolist()):
            r["reward_raw"] = float(s)
            r["reward_effective"] = float(s) - (0.0 if r["terminated_with_eos"] else penalty)


def heldout_summary(records: list[dict]) -> dict:
    tok_cnt = sum(r["kl_token_count"] for r in records)
    raw = np.array([r["reward_raw"] for r in records])
    eff = np.array([r["reward_effective"] for r in records])
    return {
        "n": len(records),
        "reward_raw_mean": float(raw.mean()), "reward_raw_std": float(raw.std()),
        "reward_effective_mean": float(eff.mean()), "reward_effective_std": float(eff.std()),
        "kl_token_mean_pooled": sum(r["kl_token_sum"] for r in records) / max(tok_cnt, 1.0),
        "kl_per_sequence_sum_mean": float(np.mean([r["kl_token_sum"] for r in records])),
        # common.metrics.sample_entropy convention: -mean sampled log-prob, pooled over response tokens
        "entropy_token_mean_pooled": -sum(r["policy_logp_sum"] for r in records) / max(tok_cnt, 1.0),
        "response_tokens": length_stats([r["response_tokens"] for r in records]),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
    }


def evaluate_policy(config_path: str, adapter: str | None, name: str, limit: int | None = None) -> dict:
    """Fixed held-out protocol, identical for every Task 2 model (midpoint, standard, all forks)."""
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    out_dir = repo_path(cfg["results_dir"]) / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = wall_timer()
    tokenizer = load_tokenizer(cfg["base_model"])
    items, dropped = heldout_items(cfg, tokenizer, limit)
    max_new = int(cfg["eval_max_response_length"])

    set_seed(seed)
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    print(f"[eval:{name}] adapter={adapter} prompts={len(items)} max_new_tokens={max_new}", flush=True)
    records = generate_with_kl(model, tokenizer, items, cfg, max_new, seed)
    del model
    clear_gpu()
    rm = load_reward_model(cfg)
    add_rewards(records, items, cfg, rm)
    clear_gpu(rm[0])
    write_jsonl(out_dir / "generations.jsonl", records)

    metrics = {
        "name": name,
        "adapter": adapter,
        "status": "COMPLETED" if limit is None else "SMOKE_TEST",
        "seed": seed,
        "config_path": config_path,
        "prompt_set": cfg["paths"]["rl_prompt_eval"],
        "prompt_source_indices": [it["meta"]["source_index"] for it in items],
        "long_prompts_dropped": dropped,
        "decoding": {**cfg["generation"], "max_new_tokens": max_new, "max_prompt_length": int(cfg["max_prompt_length"])},
        "missing_eos_penalty": float(cfg["missing_eos_penalty"]),
        "heldout": heldout_summary(records),
        "wall_clock_s": timer(),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
    }
    save_json(out_dir / "metrics.json", metrics)
    h = metrics["heldout"]
    print(f"[eval:{name}] reward_eff={h['reward_effective_mean']:.4f} reward_raw={h['reward_raw_mean']:.4f} "
          f"kl={h['kl_token_mean_pooled']:.5f} entropy={h['entropy_token_mean_pooled']:.4f} "
          f"len={h['response_tokens']['mean']:.1f} eos={h['eos_rate']:.3f} -> {out_dir}", flush=True)
    return metrics


def ensure_evaluated(config_path: str, adapter: str | None, name: str, force: bool = False) -> dict:
    """Evaluate unless a completed evaluation of the same adapter already exists (resume-safe)."""
    cfg = load_yaml(config_path)
    path = repo_path(cfg["results_dir"]) / "eval" / name / "metrics.json"
    if not force and path.exists():
        m = load_json(path)
        if m.get("status") == "COMPLETED" and m.get("adapter") == adapter:
            print(f"[eval:{name}] reusing completed evaluation")
            return m
    return evaluate_policy(config_path, adapter, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help="e.g. checkpoints/ppo_midpoint_policy or outputs/task2_ppo/standard")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--limit", type=int, help="smoke tests only")
    args = ap.parse_args()
    evaluate_policy(args.config, args.adapter, args.name, args.limit)


if __name__ == "__main__":
    main()
