from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode, trainable_parameters
from task2_ppo.continue_train import disable_dropout, fork_name, run_or_reuse_fork
from task2_ppo.evaluate import ensure_evaluated
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


# --------------------------------------------------------------------------- cached-batch study

def build_cached_batch(cfg: dict, tokenizer, rows: list[dict]) -> list[dict]:
    """Reconstruct token ids for each cached rollout (the cache stores text, not ids).

    Prompt = the eval-pool row's messages through the chat template (no truncation); response = the
    cached text re-tokenized, plus EOS when the cached rollout terminated. The resulting length is checked
    against the cached `response_tokens`.
    """
    ev = {str(r["source_index"]): r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    batch = []
    for r in rows:
        msgs = prompt_messages(ev[str(r["source_index"])])
        p_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        resp = tokenizer(r["response"], add_special_tokens=False)["input_ids"]
        if r["terminated_with_eos"]:
            resp = resp + [tokenizer.eos_token_id]
        n_cached = int(r["response_tokens"])
        n = min(len(resp), n_cached)
        batch.append({
            "source_index": str(r["source_index"]),
            "ids": torch.tensor([p_ids + resp[:n]]),
            "prompt_width": len(p_ids),
            "n": n,
            "retokenized_matches_cache": len(resp) == n_cached,
            "long_prompt": len(p_ids) > int(cfg["max_prompt_length"]),
            "values": r["values"][:n].float()[None],
            "cached_old": r["old_logprobs"][:n].float()[None],
            "cached_ref": r["ref_logprobs"][:n].float()[None],
            "effective_reward": float(r["effective_terminal_reward"]),
        })
    return batch


def _logp(policy, item, device, with_grad=False):
    ids = item["ids"].to(device)
    attn = torch.ones_like(ids)
    rid = ids[:, item["prompt_width"]:]
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        lp, _ = response_token_logprobs(policy, ids, attn, item["prompt_width"], rid)
    return lp


def clip_stats(ratio: torch.Tensor, adv: torch.Tensor, eps: float) -> dict:
    """Pooled over all valid response tokens (same convention as ppo_policy_loss's masked_mean)."""
    outside = (ratio < 1 - eps) | (ratio > 1 + eps)
    affected = ((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps))
    unclipped = ratio * adv
    clipped = torch.minimum(unclipped, ratio.clamp(1 - eps, 1 + eps) * adv)
    return {"clip_fraction": float(outside.float().mean()), "affected_fraction": float(affected.float().mean()),
            "surrogate_unclipped": float(unclipped.mean()), "surrogate_clipped": float(clipped.mean()),
            "surrogate_gap": float((unclipped - clipped).mean()), "n_tokens": int(ratio.numel())}


def ratio_quantiles(ratio: torch.Tensor) -> dict:
    q = torch.quantile(ratio.double(), torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0], dtype=torch.float64))
    keys = ["min", "p01", "p05", "median", "p95", "p99", "max"]
    out = {k: float(v) for k, v in zip(keys, q)}
    out["max_abs_deviation_from_1"] = float((ratio - 1).abs().max())
    return out


def multi_epoch_sweep(policy, batch, device, eps: float, n_epochs: int, lr: float, max_grad_norm: float,
                      init_state: dict) -> list[dict]:
    """SUPPLEMENTARY (beyond the release ppo_epochs): repeated PPO epochs on the same cached batch.

    Restarts from the midpoint weights, then for each epoch measures the clip statistics on the ratios
    *before* that epoch's step and takes the step with this eps. Shows after how many passes over one batch
    each eps starts to bind; it is not part of the required protocol.
    """
    for name, p in policy.named_parameters():
        if name in init_state:
            p.data.copy_(init_state[name])
    opt = AdamW(trainable_parameters(policy), lr=lr)
    total = sum(it["n"] for it in batch)
    out = []
    for epoch in range(1, n_epochs + 1):
        opt.zero_grad(set_to_none=True)
        ratios, advs = [], []
        for it in batch:
            new = _logp(policy, it, device, with_grad=True)
            loss, ratio, _ = ppo_policy_loss(new, it["old"].to(device), it["adv"].to(device), torch.ones_like(new), eps)
            (loss * it["n"] / total).backward()
            ratios.append(ratio[0].float().cpu())
            advs.append(it["adv"][0])
            del new
        stats = clip_stats(torch.cat(ratios), torch.cat(advs), eps)
        stats["epoch"] = epoch
        stats["max_abs_ratio_dev"] = float((torch.cat(ratios) - 1).abs().max())
        stats["policy_grad_norm_preclip"] = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm))
        opt.step()
        out.append(stats)
    return out


def cached_batch_study(config_path: str = "configs/ppo.yaml") -> dict:
    """Immediate geometric effect of eps on the supplied cached rollout batch.

    Protocol (ppo_epochs = 2, the release setting):
      * pi_old / pi_ref are recomputed with this pipeline from the midpoint, so the epoch-1 ratio is exactly 1.
        (The cached log-probs cannot be reproduced exactly: ~0.015 nats/token numerical gap; reported below.)
      * Advantages: cached effective terminal reward + KL shaping (kl_beta) + cached critic values -> GAE.
      * Epoch 1: one AdamW step on the clipped loss over the whole batch. Because the ratio is 1 everywhere,
        clipping cannot bind, so this step is identical for every eps.
      * Epoch 2: the ratios after that step are measured once; each eps is applied to the same ratios.
    """
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    timer = wall_timer()
    tok = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    batch = build_cached_batch(cfg, tok, rows)
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    disable_dropout(policy)
    device = next(policy.parameters()).device
    beta, gamma, lam = float(cfg["kl_beta"]), float(cfg["gamma"]), float(cfg["gae_lambda"])

    for it in batch:
        it["old"] = _logp(policy, it, device).float().cpu()
        with reference_mode(policy):
            it["ref"] = _logp(policy, it, device).float().cpu()
        mask = torch.ones(1, it["n"])
        rewards = shaped_rewards(torch.tensor([it["effective_reward"]]), it["old"], it["ref"], mask, beta)
        it["adv"], _ = compute_gae(rewards, it["values"], mask, gamma, lam)

    total_tokens = sum(it["n"] for it in batch)
    init_state = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
    opt = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
    opt.zero_grad(set_to_none=True)
    epoch1_clip = 0.0
    for it in batch:
        new = _logp(policy, it, device, with_grad=True)
        loss, _, cf = ppo_policy_loss(new, it["old"].to(device), it["adv"].to(device),
                                      torch.ones_like(new), float(cfg["clip_epsilon"]))
        # weight by token count -> identical to masked_mean pooled over the padded batch
        (loss * it["n"] / total_tokens).backward()
        epoch1_clip += float(cf) * it["n"] / total_tokens
        del new
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"])))
    opt.step()
    for it in batch:
        it["new"] = _logp(policy, it, device).float().cpu()
    sweep_epochs = int(cfg.get("clip_sensitivity_epochs", 0))
    sweep = {f"{float(e):g}": multi_epoch_sweep(policy, batch, device, float(e), sweep_epochs,
                                                float(cfg["policy_learning_rate"]), float(cfg["max_grad_norm"]),
                                                init_state)
             for e in cfg["clip_values"]} if sweep_epochs > 0 else {}
    del policy, opt
    clear_gpu()

    def pooled(sel, key_a, key_b):
        r = torch.cat([torch.exp(it[key_a] - it[key_b])[0] for it in batch if sel(it)])
        a = torch.cat([it["adv"][0] for it in batch if sel(it)])
        return r, a

    every, short = (lambda it: True), (lambda it: not it["long_prompt"])
    ratio_all, adv_all = pooled(every, "new", "old")
    ratio_short, adv_short = pooled(short, "new", "old")
    ratio_cache, adv_cache = pooled(short, "old", "cached_old")
    ratio_ref_ctrl, adv_ref_ctrl = pooled(short, "ref", "cached_ref")
    eps_values = [float(e) for e in cfg["clip_values"]]
    result = {
        "status": "COMPLETED",
        "config_path": config_path,
        "seed": int(cfg["seed"]),
        "cache": cfg["cached_rollouts"],
        "n_rollouts": len(batch),
        "n_tokens": total_tokens,
        "long_prompt_rollouts": [it["source_index"] for it in batch if it["long_prompt"]],
        "retokenization_mismatches": [it["source_index"] for it in batch if not it["retokenized_matches_cache"]],
        "protocol": cached_batch_study.__doc__,
        "kl_beta_for_advantages": beta,
        "epoch1": {"clip_fraction": epoch1_clip, "policy_grad_norm_preclip": grad_norm,
                   "note": "ratio == 1 on every token in epoch 1, so the step is the same for every eps"},
        "advantage": {"mean": float(adv_all.mean()), "std": float(adv_all.std()),
                      "frac_positive": float((adv_all > 0).float().mean())},
        "epoch2_ratio_quantiles": ratio_quantiles(ratio_all),
        "by_eps": {f"{e:g}": clip_stats(ratio_all, adv_all, e) for e in eps_values},
        "by_eps_excluding_long_prompts": {f"{e:g}": clip_stats(ratio_short, adv_short, e) for e in eps_values},
        "diagnostic_cached_vs_recomputed_old": {
            "note": "ratio exp(recomputed old - cached old) at the unchanged midpoint (rows with prompts <= "
                    "max_prompt_length). Pure numerical/precision mismatch; shown to justify recomputing pi_old.",
            "abs_logratio_mean": float(torch.log(ratio_cache).abs().mean()),
            "ratio_quantiles": ratio_quantiles(ratio_cache),
            "by_eps": {f"{e:g}": clip_stats(ratio_cache, adv_cache, e) for e in eps_values},
            "reference_control": {
                "note": "same comparison for the frozen reference model, which never changes between the staff "
                        "run and ours: any gap here is numerical by construction",
                "abs_logratio_mean": float(torch.log(ratio_ref_ctrl).abs().mean()),
                "by_eps": {f"{e:g}": clip_stats(ratio_ref_ctrl, adv_ref_ctrl, e) for e in eps_values},
            },
        },
        "supplementary_multi_epoch_sweep": {
            "note": "NOT part of the required protocol (release ppo_epochs=2). Per eps, restart from the midpoint and "
                    "take repeated PPO epochs on the same cached batch; stats are measured before each epoch's step.",
            "epochs": sweep_epochs,
            "by_eps": sweep,
        },
        "per_rollout": [{"source_index": it["source_index"], "n_tokens": it["n"], "long_prompt": it["long_prompt"],
                         "effective_reward": it["effective_reward"],
                         "max_abs_ratio_dev": float((torch.exp(it["new"] - it["old"]) - 1).abs().max()),
                         **{f"clip_fraction_eps{e:g}": clip_stats(torch.exp(it["new"] - it["old"])[0], it["adv"][0], e)["clip_fraction"]
                            for e in eps_values}} for it in batch],
        "wall_clock_s": timer(),
    }
    out = repo_path(cfg["results_dir"]) / "clipping" / "cached_batch_study.json"
    save_json(out, result)
    print(f"[clipping:cached] tokens={total_tokens} ratio range "
          f"[{result['epoch2_ratio_quantiles']['min']:.4f}, {result['epoch2_ratio_quantiles']['max']:.4f}]")
    for e, s in result["by_eps"].items():
        print(f"[clipping:cached] eps={e}: clip_fraction={s['clip_fraction']:.5f} affected={s['affected_fraction']:.5f} "
              f"surrogate unclipped={s['surrogate_unclipped']:.5f} clipped={s['surrogate_clipped']:.5f}")
    print(f"[clipping:cached] saved -> {out}")
    return result


def run_clipping_study(config_path: str = "configs/ppo.yaml", skip_cached: bool = False, skip_forks: bool = False,
                       force: bool = False) -> None:
    cfg = load_yaml(config_path)
    print("Required epsilon values:", cfg["clip_values"])
    if not skip_cached:
        cached_batch_study(config_path)
    if not skip_forks:
        # Matched short forks from the identical midpoint: only eps changes (kl_beta, prompts, seed, budget fixed).
        for eps in [float(e) for e in cfg["clip_values"]]:
            adapter = run_or_reuse_fork(config_path, eps, float(cfg["kl_beta"]), force=force)
            ensure_evaluated(config_path, adapter, fork_name(eps, float(cfg["kl_beta"])), force=force)
            clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-cached", action="store_true")
    ap.add_argument("--skip-forks", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain/re-evaluate even if completed outputs exist")
    args = ap.parse_args()
    run_clipping_study(args.config, args.skip_cached, args.skip_forks, args.force)


if __name__ == "__main__":
    main()
