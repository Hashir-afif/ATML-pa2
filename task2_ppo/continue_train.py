from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, load_json, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    count_parameters,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


# --------------------------------------------------------------------------- setup helpers

def disable_dropout(model) -> int:
    """Set every dropout probability to 0 (TRL's PPO default). Returns the number of modules changed.

    With dropout active, the epoch-1 ratio pi_new/pi_old would differ from 1 by dropout noise alone, so
    the logged clip fraction would partly measure noise rather than policy change.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout) and m.p > 0:
            m.p = 0.0
            n += 1
    return n


def fp32_master_weights(model) -> list[str]:
    """Cast trainable fp16 parameters (the critic's modules_to_save score head) to fp32.

    Adam on raw fp16 weights underflows (eps, second moments). Forward passes run under fp16 autocast
    (see critic_values), so compute stays fp16 while the optimizer updates fp32 master weights.
    """
    cast = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.dtype == torch.float16:
            p.data = p.data.float()
            cast.append(name)
    return cast


def prompt_length(tokenizer, row: dict) -> int:
    return len(tokenizer.apply_chat_template(prompt_messages(row), tokenize=True, add_generation_prompt=True))


def filter_long_prompts(tokenizer, rows: list[dict], max_prompt_length: int):
    """Drop prompts longer than max_prompt_length chat-template tokens (same rule for every run).

    common.generation.batch_generate truncates from the right, which would cut the assistant header off
    an over-long prompt, so such prompts cannot be used as-is. Returns (kept_rows, dropped_source_indices).
    """
    kept, dropped = [], []
    for r in rows:
        (kept if prompt_length(tokenizer, r) <= max_prompt_length else dropped).append(r)
    return kept, [str(r["source_index"]) for r in dropped]


def rollout_prompt_order(cfg: dict, tokenizer) -> tuple[list[dict], list[str]]:
    """Fixed training-prompt order: filtered pool permuted once with the global seed.

    Every run (standard and all forks) consumes this list from the start, so an 8-update fork sees exactly
    the first 8 prompts of the 20-update standard run.
    """
    rows, dropped = filter_long_prompts(tokenizer, read_jsonl(cfg["paths"]["rl_prompt_train"]),
                                        int(cfg["max_prompt_length"]))
    order = np.random.default_rng(int(cfg["seed"])).permutation(len(rows))
    return [rows[i] for i in order], dropped


def critic_values(critic, sequences, attention_mask, prompt_width: int, n_resp: int):
    """V(s_t) for each response token t = critic value at the position just before token t.

    Verified against cached/ppo_rollout.pt: this alignment reproduces the cached values (mean abs diff
    ~0.05, fp16 vs the staff's bf16), while the next-position alignment is off by ~2.
    """
    with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        v = token_values(critic, sequences, attention_mask)
    return v[:, prompt_width - 1: prompt_width - 1 + n_resp].float()


def explained_variance(pred, target, mask) -> float:
    m = mask.bool()
    y, p = target[m].float(), pred[m].float()
    var_y = y.var(unbiased=False)
    if var_y <= 0:
        return float("nan")
    return float(1.0 - (y - p).var(unbiased=False) / var_y)


# --------------------------------------------------------------------------- rollout

@torch.no_grad()
def collect_rollout(policy, critic, reward_model, reward_tokenizer, tokenizer, row: dict, cfg: dict) -> dict:
    """Sample one response, then record old/ref log-probs, values and the learned terminal reward."""
    gcfg = cfg["generation"]
    msgs = prompt_messages(row)
    gen = batch_generate(policy, tokenizer, [msgs], max_prompt_length=int(cfg["max_prompt_length"]),
                         max_new_tokens=int(cfg["max_response_length"]),
                         temperature=float(gcfg["temperature"]), top_p=float(gcfg["top_p"]),
                         do_sample=bool(gcfg["do_sample"]))
    # batch_generate runs under torch.inference_mode; clone so these tensors can enter the autograd graph
    # of the PPO epochs (inference tensors cannot be saved for backward).
    seq, attn, pw = gen["sequences"].clone(), gen["attention_mask"].clone(), gen["prompt_width"]
    rid, mask = gen["response_ids"].clone(), gen["response_mask"].clone()
    n = rid.shape[1]
    old_logp, _ = response_token_logprobs(policy, seq, attn, pw, rid)
    with reference_mode(policy):
        ref_logp, _ = response_token_logprobs(policy, seq, attn, pw, rid)
    values = critic_values(critic, seq, attn, pw, n)
    raw = float(score_reward_pairs(reward_model, reward_tokenizer, [msgs], gen["responses"],
                                   max_length=int(cfg["reward_max_length"]))[0])
    terminated = bool(gen["terminated_with_eos"][0])
    # Same rule as the staff cache: effective = raw - penalty when the response never emits EOS.
    effective = raw - (0.0 if terminated else float(cfg["missing_eos_penalty"]))
    return {"row": row, "sequences": seq, "attention_mask": attn, "prompt_width": pw, "response_ids": rid,
            "mask": mask, "old_logp": old_logp, "ref_logp": ref_logp, "values": values,
            "raw_reward": raw, "effective_reward": effective, "terminated": terminated,
            "truncated": bool(gen["truncated"][0]), "response": gen["responses"][0],
            "response_tokens": int(gen["response_lengths"][0])}


def affected_fraction(ratio, advantage, mask, eps: float):
    """Tokens where clipping actually binds in min(): the clipped term is selected and its gradient is 0."""
    hit = ((advantage > 0) & (ratio > 1.0 + eps)) | ((advantage < 0) & (ratio < 1.0 - eps))
    return masked_mean(hit.float(), mask)


# --------------------------------------------------------------------------- one PPO update

def ppo_update(policy, critic, policy_opt, value_opt, rollouts: list[dict], cfg: dict, eps: float, kl_beta: float) -> dict:
    """Shaped rewards -> GAE -> ppo_epochs passes of clipped policy loss + value MSE on the rollouts."""
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
    value_coef, max_gn = float(cfg["value_coef"]), float(cfg["max_grad_norm"])
    pol_params, val_params = trainable_parameters(policy), trainable_parameters(critic)
    for r in rollouts:
        task_r = torch.tensor([r["effective_reward"]], device=r["old_logp"].device)
        r["rewards"] = shaped_rewards(task_r, r["old_logp"], r["ref_logp"], r["mask"], kl_beta)
        adv, ret = compute_gae(r["rewards"], r["values"], r["mask"], gamma, lam)
        r["advantages"] = normalize_advantages(adv, r["mask"]) if cfg.get("normalize_advantages", False) else adv
        r["returns"] = ret

    epochs = []
    for epoch in range(int(cfg["ppo_epochs"])):
        policy_opt.zero_grad(set_to_none=True)
        value_opt.zero_grad(set_to_none=True)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "clip_fraction": 0.0, "affected_fraction": 0.0,
                 "ratio_mean": 0.0, "ratio_min": float("inf"), "ratio_max": 0.0}
        for r in rollouts:
            new_logp, _ = response_token_logprobs(policy, r["sequences"], r["attention_mask"],
                                                  r["prompt_width"], r["response_ids"])
            p_loss, ratio, clip_frac = ppo_policy_loss(new_logp, r["old_logp"], r["advantages"], r["mask"], eps)
            v_pred = critic_values(critic, r["sequences"], r["attention_mask"], r["prompt_width"],
                                   r["response_ids"].shape[1])
            v_loss = value_mse_loss(v_pred, r["returns"], r["mask"])
            ((p_loss + value_coef * v_loss) / len(rollouts)).backward()
            k = len(rollouts)
            valid = r["mask"].bool()
            stats["policy_loss"] += float(p_loss.detach()) / k
            stats["value_loss"] += float(v_loss.detach()) / k
            stats["clip_fraction"] += float(clip_frac) / k
            stats["affected_fraction"] += float(affected_fraction(ratio, r["advantages"], r["mask"], eps)) / k
            stats["ratio_mean"] += float(masked_mean(ratio, r["mask"])) / k
            stats["ratio_min"] = min(stats["ratio_min"], float(ratio[valid].min()))
            stats["ratio_max"] = max(stats["ratio_max"], float(ratio[valid].max()))
            del new_logp, v_pred
        stats["policy_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(pol_params, max_gn))
        stats["value_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(val_params, max_gn))
        policy_opt.step()
        value_opt.step()
        epochs.append(stats)

    # Stability statistic: how far this update moved the policy on its own rollout tokens,
    # estimated as the sampled KL(pi_old || pi_new) = E[log pi_old - log pi_new].
    with torch.no_grad():
        shift, max_abs_logratio = 0.0, 0.0
        for r in rollouts:
            after, _ = response_token_logprobs(policy, r["sequences"], r["attention_mask"],
                                               r["prompt_width"], r["response_ids"])
            d = r["old_logp"] - after
            shift += float(masked_mean(d, r["mask"])) / len(rollouts)
            max_abs_logratio = max(max_abs_logratio, float(d[r["mask"].bool()].abs().max()))
    return {"epochs": epochs, "policy_shift_kl": shift, "max_abs_logratio_after_update": max_abs_logratio}


# --------------------------------------------------------------------------- continuation run

def fork_name(eps: float, kl_beta: float) -> str:
    return f"fork_eps{eps:g}_kl{kl_beta:g}"


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None,
            kl_beta: float | None = None, run_name: str = "standard", smoke_test: bool = False):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, policy, critic = bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    popt, vopt = bundle["policy_optimizer"], bundle["value_optimizer"]
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    n_updates, per_update = int(cfg["updates"]), int(cfg["prompts_per_update"])
    seed = int(cfg["seed"])

    dropped_dropout = 0
    if cfg.get("disable_dropout", True):
        dropped_dropout = disable_dropout(policy) + disable_dropout(critic)
    master_cast = fp32_master_weights(critic)
    prompts, dropped_prompts = rollout_prompt_order(cfg, tok)
    needed = n_updates * per_update
    schedule = prompts[:needed]

    run_dir = repo_path(cfg["results_dir"]) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()  # a rerun of the same run name replaces its own training log
    print(f"[{run_name}] eps={eps} kl_beta={beta} updates={n_updates} prompts/update={per_update} "
          f"dropout_off={dropped_dropout} fp32_master={master_cast} long_prompts_dropped={len(dropped_prompts)}")

    set_seed(seed)  # identical sampling stream at the start of every run
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    history = []
    for u in range(n_updates):
        t_update = wall_timer()
        rows = schedule[u * per_update:(u + 1) * per_update]
        policy.eval()
        critic.eval()
        rollouts = [collect_rollout(policy, critic, rm, rm_tok, tok, row, cfg) for row in rows]
        t_rollout = t_update()
        policy.train()
        critic.train()
        upd = ppo_update(policy, critic, popt, vopt, rollouts, cfg, eps, beta)
        last = upd["epochs"][-1]
        k = len(rollouts)
        rec = {
            "update": u + 1,
            "source_indices": [str(r["row"]["source_index"]) for r in rollouts],
            "reward_raw": sum(r["raw_reward"] for r in rollouts) / k,
            "reward_effective": sum(r["effective_reward"] for r in rollouts) / k,
            "shaped_return": sum(float(r["rewards"].sum()) for r in rollouts) / k,
            "kl": sum(float(sampled_kl(r["old_logp"], r["ref_logp"], r["mask"])) for r in rollouts) / k,
            "kl_seq_sum": sum(float(((r["old_logp"] - r["ref_logp"]) * r["mask"]).sum()) for r in rollouts) / k,
            "entropy": sum(float(sample_entropy(r["old_logp"], r["mask"])) for r in rollouts) / k,
            "response_tokens": sum(r["response_tokens"] for r in rollouts) / k,
            "terminated_with_eos": sum(r["terminated"] for r in rollouts) / k,
            "value_mean": sum(float(masked_mean(r["values"], r["mask"])) for r in rollouts) / k,
            "value_explained_variance": float(np.nanmean([explained_variance(r["values"], r["returns"], r["mask"]) for r in rollouts])),
            "advantage_mean": sum(float(masked_mean(r["advantages"], r["mask"])) for r in rollouts) / k,
            "return_mean": sum(float(masked_mean(r["returns"], r["mask"])) for r in rollouts) / k,
            "policy_loss": float(np.mean([e["policy_loss"] for e in upd["epochs"]])),
            "value_loss": float(np.mean([e["value_loss"] for e in upd["epochs"]])),
            "clip_fraction": last["clip_fraction"],
            "affected_fraction": last["affected_fraction"],
            "grad_norm": float(np.mean([e["policy_grad_norm"] for e in upd["epochs"]])),
            "value_grad_norm": float(np.mean([e["value_grad_norm"] for e in upd["epochs"]])),
            "policy_shift_kl": upd["policy_shift_kl"],
            "max_abs_logratio_after_update": upd["max_abs_logratio_after_update"],
            "epochs": upd["epochs"],
            "rollout_s": t_rollout,
            "update_s": t_update(),
            "elapsed_s": elapsed(),
        }
        if torch.cuda.is_available():
            rec["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
        append_jsonl(log_path, rec)
        history.append(rec)
        print(f"[{run_name}] update {u + 1}/{n_updates} reward={rec['reward_effective']:.3f} kl={rec['kl']:.4f} "
              f"len={rec['response_tokens']:.0f} pl={rec['policy_loss']:.4f} vl={rec['value_loss']:.4f} "
              f"clip={rec['clip_fraction']:.4f} gn={rec['grad_norm']:.3f} t={rec['elapsed_s']:.0f}s", flush=True)
        del rollouts

    policy.save_pretrained(str(out))
    critic.save_pretrained(str(out / "value"))
    run_info = {
        "run_name": run_name,
        "status": "SMOKE_TEST" if smoke_test else "COMPLETED",
        "config_path": config_path,
        "config": cfg,
        "seed": seed,
        "start_checkpoint": {"policy": cfg["paths"]["ppo_midpoint_policy"], "value": cfg["paths"]["ppo_midpoint_value"]},
        "clip_epsilon": eps,
        "kl_beta": beta,
        "updates": len(history),
        "prompts_per_update": per_update,
        "ppo_epochs": int(cfg["ppo_epochs"]),
        "prompt_source_indices": [str(r["source_index"]) for r in schedule],
        "prompt_ids": [r["prompt_id"] for r in schedule],
        "long_prompts_dropped_from_pool": dropped_prompts,
        "optimizer": {"policy": {"name": "AdamW", "lr": float(cfg["policy_learning_rate"]), "schedule": "constant"},
                      "value": {"name": "AdamW", "lora_lr": float(cfg["value_lora_learning_rate"]),
                                "head_lr": float(cfg["value_head_learning_rate"]), "weight_decay": 0.0,
                                "schedule": "constant"}},
        "value_train_mode": cfg.get("value_train_mode"),
        "disable_dropout": bool(cfg.get("disable_dropout", True)),
        "normalize_advantages": bool(cfg.get("normalize_advantages", False)),
        "fp32_master_weights": master_cast,
        "rollout_decoding": {**cfg["generation"], "max_new_tokens": int(cfg["max_response_length"]),
                             "max_prompt_length": int(cfg["max_prompt_length"])},
        "generated_tokens_total": int(sum(h["response_tokens"] * per_update for h in history)),
        "parameters_policy": dict(zip(["total", "trainable"], count_parameters(policy))),
        "parameters_value": dict(zip(["total", "trainable"], count_parameters(critic))),
        "adapter_path": str(out.relative_to(repo_path("."))).replace("\\", "/"),
        "value_adapter_path": str((out / "value").relative_to(repo_path("."))).replace("\\", "/"),
        "wall_clock_s": elapsed(),
        "peak_vram_gib": (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else None,
        "peak_vram_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30) if torch.cuda.is_available() else None,
        "train_log": str(log_path.relative_to(repo_path("."))).replace("\\", "/"),
    }
    save_json(run_dir / "run_info.json", run_info)
    save_json(out / "run_info.json", run_info)
    print(f"[{run_name}] saved policy -> {out} (critic -> {out / 'value'}); "
          f"{run_info['wall_clock_s']:.0f}s, peak {run_info['peak_vram_gib'] or 0:.2f} GiB", flush=True)
    return {"policy": policy, "value_model": critic, "reward_model": rm, "run_info": run_info, "history": history}


def run_or_reuse_fork(config_path: str, eps: float, kl_beta: float, force: bool = False) -> str:
    """Train the short fork (eps, kl_beta) from the midpoint unless an identical completed fork exists.

    The (eps=0.20, kl_beta=0.10) fork is the reference point of both the clipping and the KL study; it is
    trained once and reused by both. Returns the adapter path.
    """
    cfg = load_yaml(config_path)
    name = fork_name(eps, kl_beta)
    adapter = f"{Path(cfg['output']).parent.as_posix()}/{name}"
    info_path = repo_path(adapter) / "run_info.json"
    if not force and info_path.exists():
        info = load_json(info_path)
        if (info.get("status") == "COMPLETED" and info.get("updates") == int(cfg["fork_updates"])
                and abs(info["clip_epsilon"] - eps) < 1e-12 and abs(info["kl_beta"] - kl_beta) < 1e-12):
            print(f"[{name}] reusing completed fork at {adapter}")
            return adapter
    result = run_ppo(config_path, output=adapter, updates=int(cfg["fork_updates"]),
                     clip_epsilon=eps, kl_beta=kl_beta, run_name=name)
    for key in ("policy", "value_model", "reward_model"):
        result.pop(key, None)
    clear_gpu()
    return adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--smoke-test", action="store_true", help="mark the run as a smoke test in run_info.json")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name, args.smoke_test)


if __name__ == "__main__":
    main()
