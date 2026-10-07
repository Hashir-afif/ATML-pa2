from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.data import (
    load_yaml,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.eval_utils import generate_with_kl, length_stats
from common.generation import score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.metrics import parse_word_limit, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from task1_dpo.train import filter_overlong_prompts, make_collate, pair_logprobs


# --------------------------------------------------------------------------- statistics

def preference_summary(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    m = np.array([r["margin"] for r in records])
    loss = np.array([r["dpo_loss"] for r in records])
    tie = np.array([r["rating_tie"] for r in records])
    correct = m > 0
    return {
        "n": int(len(records)),
        "preference_accuracy": float(correct.mean()),
        "dpo_loss": float(loss.mean()),
        "margin_mean": float(m.mean()),
        "margin_std": float(m.std(ddof=0)),
        "chosen_logratio_mean": float(np.mean([r["chosen_logratio"] for r in records])),
        "rejected_logratio_mean": float(np.mean([r["rejected_logratio"] for r in records])),
        "n_rating_ties": int(tie.sum()),
        "preference_accuracy_excluding_ties": float(correct[~tie].mean()) if (~tie).any() else None,
        "n_margin_exactly_zero": int((m == 0).sum()),
    }


# --------------------------------------------------------------------------- held-out pairs

@torch.no_grad()
def evaluate_pairs(model, tokenizer, rows: list[dict], beta: float, cfg: dict) -> list[dict]:
    """Per-pair DPO margin m = [log pi(y+) - log ref(y+)] - [log pi(y-) - log ref(y-)]."""
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(model.parameters()).device
    bs = int(cfg["eval_batch_size"])
    out = []
    for start in range(0, len(rows), bs):
        chunk = rows[start:start + bs]
        chosen, rejected = collate(chunk)
        chosen = {k: v.to(device) for k, v in chosen.items()}
        rejected = {k: v.to(device) for k, v in rejected.items()}
        pol_c, pol_r, ref_c, ref_r, len_c, len_r = pair_logprobs(model, chosen, rejected, with_grad=False)
        margin = (pol_c - ref_c) - (pol_r - ref_r)
        loss = -F.logsigmoid(beta * margin)
        for i, row in enumerate(chunk):
            out.append({
                "source_index": str(row.get("source_index")),
                "prompt_id": row.get("prompt_id"),
                "length_stratum": row.get("length_stratum"),
                "score_chosen": float(row["score_chosen"]),
                "score_rejected": float(row["score_rejected"]),
                "rating_tie": float(row["score_chosen"]) == float(row["score_rejected"]),
                "policy_chosen_logp": float(pol_c[i]), "policy_rejected_logp": float(pol_r[i]),
                "ref_chosen_logp": float(ref_c[i]), "ref_rejected_logp": float(ref_r[i]),
                "chosen_logratio": float(pol_c[i] - ref_c[i]),
                "rejected_logratio": float(pol_r[i] - ref_r[i]),
                "margin": float(margin[i]),
                "dpo_loss": float(loss[i]),
                "correct": bool(margin[i] > 0),
                "chosen_tokens_scored": int(len_c[i]), "rejected_tokens_scored": int(len_r[i]),
            })
    return out


# --------------------------------------------------------------------------- generation + KL
# generate_with_kl / length_stats live in common.eval_utils (shared with Tasks 2-3).

def add_rewards(records: list[dict], items: list[dict], cfg: dict, rm=None) -> None:
    own = rm is None
    if own:
        rm = load_reward_model(cfg)
    rm_model, rm_tok = rm
    bs = int(cfg["eval_batch_size"])
    by_key = {it["meta"]["item_key"]: it["messages"] for it in items}
    for start in range(0, len(records), bs):
        chunk = records[start:start + bs]
        scores = score_reward_pairs(
            rm_model, rm_tok,
            [by_key[r["item_key"]] for r in chunk],
            [r["response"] for r in chunk],
            max_length=int(cfg["reward_max_length"]),
        )
        for r, s in zip(chunk, scores.tolist()):
            r["reward"] = float(s)
    if own:
        clear_gpu(rm_model)


def generation_summary(records: list[dict]) -> dict:
    tok_sum = sum(r["kl_token_sum"] for r in records)
    tok_cnt = sum(r["kl_token_count"] for r in records)
    rewards = [r["reward"] for r in records if r.get("reward") is not None]
    return {
        "n": len(records),
        "kl_token_mean_pooled": tok_sum / max(tok_cnt, 1.0),
        "kl_per_sequence_sum_mean": float(np.mean([r["kl_token_sum"] for r in records])),
        "kl_per_response_token_mean_mean": float(np.mean([r["kl_token_mean"] for r in records])),
        "reward_mean": float(np.mean(rewards)) if rewards else None,
        "reward_std": float(np.std(rewards)) if rewards else None,
        "response_tokens": length_stats([r["response_tokens"] for r in records]),
        "word_count": length_stats([r["word_count"] for r in records]),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
    }


def word_limit_summary(records: list[dict]) -> dict:
    comp = [r["compliant"] for r in records if r["compliant"] is not None]
    per_prompt = {}
    for r in records:
        per_prompt.setdefault(r["prompt_id"], []).append(r)
    prompt_rows = []
    for pid, rs in sorted(per_prompt.items()):
        prompt_rows.append({
            "prompt_id": pid, "word_limit": rs[0]["word_limit"],
            "compliance_rate": float(np.mean([x["compliant"] for x in rs])),
            "mean_word_count": float(np.mean([x["word_count"] for x in rs])),
            "mean_overshoot_words": float(np.mean([max(0, x["word_count"] - x["word_limit"]) for x in rs])),
        })
    return {
        "n_samples": len(records),
        "n_prompts": len(per_prompt),
        "compliance_rate": float(np.mean(comp)) if comp else None,
        "prompts_all_samples_compliant": int(sum(p["compliance_rate"] == 1.0 for p in prompt_rows)),
        "word_count": length_stats([r["word_count"] for r in records]),
        "word_count_over_limit_ratio_mean": float(np.mean([r["word_count"] / r["word_limit"] for r in records])),
        "response_tokens": length_stats([r["response_tokens"] for r in records]),
        "reward_mean": float(np.mean([r["reward"] for r in records])) if records and "reward" in records[0] else None,
        "per_prompt": prompt_rows,
    }


# --------------------------------------------------------------------------- orchestration

def eval_items_from_pairs(rows: list[dict]) -> list[dict]:
    return [{"messages": prompt_messages_from_preference(r),
             "meta": {"item_key": f"eval:{r.get('source_index')}", "source_index": str(r.get("source_index")),
                      "prompt_id": r.get("prompt_id")}} for r in rows]


def word_limit_items(rows: list[dict], n_samples: int) -> list[dict]:
    items = []
    for r in rows:
        msgs = prompt_messages(r)
        user_text = msgs[-1]["content"]
        for k in range(n_samples):
            items.append({"messages": msgs, "meta": {
                "item_key": f"wl:{r['prompt_id']}", "prompt_id": r["prompt_id"], "sample_index": k,
                "word_limit": parse_word_limit(user_text), "prompt_text": user_text}})
    return items


def resolve_beta(cfg: dict, adapter: str | None, beta: float | None) -> float:
    if beta is not None:
        return float(beta)
    if adapter:
        info = repo_path(adapter) / "run_info.json"
        if info.exists():
            return float(load_json(info)["beta"])
    return float(cfg["beta"])


def evaluate_model(config_path: str, adapter: str | None, name: str, beta: float | None = None,
                   skip_generation: bool = False, limit: int | None = None) -> dict:
    """Run the fixed Task 1 evaluation protocol on one policy and save everything to disk."""
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    beta = resolve_beta(cfg, adapter, beta)
    out_dir = repo_path(cfg["results_dir"]) / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = wall_timer()

    tokenizer = load_tokenizer(cfg["base_model"])
    max_len = int(cfg["max_sequence_length"])
    policy_rule = cfg.get("overlong_prompt_policy", "error")
    std_rows, std_dropped = filter_overlong_prompts(
        tokenizer, read_jsonl(cfg["paths"]["dpo_standard_eval"]), max_len, policy_rule)
    len_rows, len_dropped = filter_overlong_prompts(
        tokenizer, read_jsonl(cfg["paths"]["dpo_length_eval"]), max_len, policy_rule)
    wl_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    if limit is not None:  # smoke tests only
        std_rows, len_rows, wl_rows = std_rows[:limit], len_rows[:limit], wl_rows[:2]

    set_seed(seed)
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    print(f"[eval:{name}] adapter={adapter} beta={beta} pairs={len(std_rows)} strat={len(len_rows)}")

    std_pairs = evaluate_pairs(model, tokenizer, std_rows, beta, cfg)
    len_pairs = evaluate_pairs(model, tokenizer, len_rows, beta, cfg)
    write_jsonl(out_dir / "pairs_standard_eval.jsonl", std_pairs)
    write_jsonl(out_dir / "pairs_length_stratified_eval.jsonl", len_pairs)

    strata = {}
    for s in ["preferred_longer", "length_matched", "rejected_longer"]:
        strata[s] = preference_summary([r for r in len_pairs if r["length_stratum"] == s])

    metrics = {
        "name": name,
        "adapter": adapter,
        "beta_for_loss": beta,
        "seed": seed,
        "config_path": config_path,
        "decoding": {**cfg["generation"], "max_new_tokens": int(cfg["max_generation_tokens"]),
                     "max_prompt_length": int(cfg["max_prompt_length"]),
                     "word_limit_samples_per_prompt": int(cfg["word_limit_samples_per_prompt"])},
        "overlong_prompt_policy": policy_rule,
        "dropped_overlong": {"standard_eval": std_dropped, "length_stratified_eval": len_dropped},
        "heldout_preference": preference_summary(std_pairs),
        "length_stratified_preference": {"all": preference_summary(len_pairs), **strata},
        "generation_eval_prompts": None,
        "word_limit": None,
        "status": "COMPLETED" if limit is None else "SMOKE_TEST",
    }

    if skip_generation:
        metrics["generation_eval_prompts"] = {"status": "NOT RUN", "note": "--skip-generation"}
        metrics["word_limit"] = {"status": "NOT RUN", "note": "--skip-generation"}
    else:
        max_new = int(cfg["max_generation_tokens"])
        eval_items = eval_items_from_pairs(std_rows)
        wl_items = word_limit_items(wl_rows, int(cfg["word_limit_samples_per_prompt"]))
        gen_eval = generate_with_kl(model, tokenizer, eval_items, cfg, max_new, seed)
        gen_wl = generate_with_kl(model, tokenizer, wl_items, cfg, max_new, seed)
        for r in gen_wl:
            r["compliant"] = word_limit_compliance(r["prompt_text"], r["response"])
        del model
        clear_gpu()
        rm = load_reward_model(cfg)
        add_rewards(gen_eval, eval_items, cfg, rm)
        add_rewards(gen_wl, wl_items, cfg, rm)
        clear_gpu(rm[0])
        write_jsonl(out_dir / "generations_eval_prompts.jsonl", gen_eval)
        write_jsonl(out_dir / "generations_word_limit.jsonl", gen_wl)
        metrics["generation_eval_prompts"] = generation_summary(gen_eval)
        metrics["word_limit"] = word_limit_summary(gen_wl)

    metrics["wall_clock_s"] = timer()
    metrics["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    save_json(out_dir / "metrics.json", metrics)
    hp = metrics["heldout_preference"]
    g = metrics["generation_eval_prompts"]
    print(f"[eval:{name}] acc={hp['preference_accuracy']:.4f} loss={hp['dpo_loss']:.4f} "
          + ("" if skip_generation else
             f"kl={g['kl_token_mean_pooled']:.4f} reward={g['reward_mean']:.4f} "
             f"len={g['response_tokens']['mean']:.1f} wl_comp={metrics['word_limit']['compliance_rate']:.2f}")
          + f" -> {out_dir}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", default=None, help="omit to evaluate the untouched SFT policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta for the reported DPO loss; default: the adapter's training beta")
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--limit", type=int, help="smoke tests only")
    args = ap.parse_args()
    evaluate_model(args.config, args.adapter, args.name, args.beta, args.skip_generation, args.limit)


if __name__ == "__main__":
    main()
