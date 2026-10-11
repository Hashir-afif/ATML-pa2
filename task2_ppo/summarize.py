"""Consolidate Task 2 results: results.json, CSV tables, figures, qualitative candidates.

Reads only what continue_train.py / evaluate.py / analyze_clipping.py saved to disk; never recomputes or
invents numbers. Missing runs are recorded as NOT RUN.
"""
from __future__ import annotations

import argparse
import csv
import json

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task2_ppo.continue_train import fork_name

# Fixed entity -> colour mapping (never re-assigned by rank or by which runs exist).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948", "#008300"]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.8, "lines.linewidth": 2})
TRAJ_KEYS = [("reward_effective", "Learned reward (effective)"), ("kl", "KL from reference (nats/token)"),
             ("policy_loss", "Policy loss"), ("value_loss", "Value loss"), ("entropy", "Entropy (nats/token)"),
             ("grad_norm", "Policy grad norm (pre-clip)"), ("clip_fraction", "Clip fraction (final epoch)"),
             ("response_tokens", "Response length (tokens)")]


def _save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def collect(cfg, names):
    root = repo_path(cfg["results_dir"])
    runs, curves, evals = {}, {}, {}
    for name in names:
        rp = root / "runs" / name / "run_info.json"
        if name == "midpoint":
            runs[name] = {"status": "NOT APPLICABLE", "note": "supplied midpoint, no continuation"}
        elif rp.exists():
            runs[name] = load_json(rp)
            curves[name] = read_jsonl(root / "runs" / name / "train_log.jsonl")
        else:
            runs[name] = {"status": "NOT RUN", "note": "run_info.json missing"}
        mp = root / "eval" / name / "metrics.json"
        evals[name] = load_json(mp) if mp.exists() else {"status": "NOT RUN", "note": "metrics.json missing"}
    return runs, curves, evals


def stability(curve):
    """Stability statistics of a run's update trajectory (defined once, applied to every run)."""
    if not curve:
        return {}
    shift = np.array([c["policy_shift_kl"] for c in curve])
    gn = np.array([c["grad_norm"] for c in curve])
    rew = np.array([c["reward_effective"] for c in curve])
    return {"policy_shift_kl_mean": float(shift.mean()), "policy_shift_kl_max": float(shift.max()),
            "max_abs_logratio_after_update_max": float(max(c["max_abs_logratio_after_update"] for c in curve)),
            "grad_norm_cv": float(gn.std() / gn.mean()) if gn.mean() > 0 else None,
            "clip_fraction_mean": float(np.mean([c["clip_fraction"] for c in curve])),
            "reward_std_over_updates": float(rew.std())}


def summary_row(name, run, curve, ev):
    row = {"model": name, "status": ev.get("status"), "updates": run.get("updates"), "clip_epsilon": run.get("clip_epsilon"),
           "kl_beta": run.get("kl_beta"), "generated_tokens_total": run.get("generated_tokens_total"),
           "wall_clock_s": run.get("wall_clock_s"), "peak_vram_gib": run.get("peak_vram_gib")}
    if ev.get("status") == "COMPLETED":
        h = ev["heldout"]
        row.update({"heldout_n": h["n"], "reward_effective": h["reward_effective_mean"], "reward_raw": h["reward_raw_mean"],
                    "reward_effective_std": h["reward_effective_std"], "kl_token_mean": h["kl_token_mean_pooled"],
                    "kl_seq_sum_mean": h["kl_per_sequence_sum_mean"], "entropy": h["entropy_token_mean_pooled"],
                    "resp_tokens_mean": h["response_tokens"]["mean"], "resp_tokens_std": h["response_tokens"]["std"],
                    "resp_tokens_iqr": h["response_tokens"]["iqr"], "eos_rate": h["eos_rate"],
                    "truncation_rate": h["truncation_rate"]})
    row.update(stability(curve))
    return row


def fig_standard(curve, out):
    if not curve:
        return
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    x = [c["update"] for c in curve]
    for ax, (k, title) in zip(axes.flat, TRAJ_KEYS):
        ax.plot(x, [c[k] for c in curve], color=PALETTE[0], marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("PPO update")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.suptitle("Task 2 standard PPO continuation (20 updates from the supplied midpoint)")
    fig.tight_layout()
    _save(fig, out / "task2_standard_trajectories")


def fig_forks(curves, names, labels, title, stem, out):
    present = [n for n in names if n in curves]
    if not present:
        return
    keys = [("reward_effective", "Learned reward (effective)"), ("kl", "KL from reference (nats/token)"),
            ("entropy", "Entropy (nats/token)"), ("response_tokens", "Response length (tokens)")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, (k, t) in zip(axes, keys):
        for i, n in enumerate(names):
            if n in curves:
                c = curves[n]
                ax.plot([r["update"] for r in c], [r[k] for r in c], color=PALETTE[i], marker="o", markersize=3, label=labels[i])
        ax.set_title(t)
        ax.set_xlabel("PPO update")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, out / stem)


def fig_cached(cached, out):
    if not cached or cached.get("status") != "COMPLETED":
        return
    sweep = cached["supplementary_multi_epoch_sweep"]["by_eps"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (e, rows) in enumerate(sweep.items()):
        rows = [r for r in rows if r["max_abs_ratio_dev"] > 0]  # epoch 1 is exactly 0 (ratio == 1); log axis
        axes[0].plot([r["epoch"] for r in rows], [r["max_abs_ratio_dev"] for r in rows], color=PALETTE[i], marker="o",
                     markersize=4, label=f"ε={e}")
        axes[0].axhline(float(e), color=PALETTE[i], linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("PPO epoch on the same cached batch")
    axes[0].set_ylabel("max |ρ − 1| over tokens")
    axes[0].set_yscale("log")
    axes[0].set_title("Ratio excursion vs clip band (dashed = ε)")
    axes[0].legend(frameon=False)
    eps = list(cached["by_eps"].keys())
    real = [cached["by_eps"][e]["clip_fraction"] for e in eps]
    noise = [cached["diagnostic_cached_vs_recomputed_old"]["by_eps"][e]["clip_fraction"] for e in eps]
    xpos = np.arange(len(eps))
    b1 = axes[1].bar(xpos - 0.2, real, width=0.38, color=PALETTE[0], label="Epoch-2 ratios (recomputed π_old)")
    b2 = axes[1].bar(xpos + 0.2, noise, width=0.38, color=PALETTE[1], label="Cached π_old (numerical noise only)")
    axes[1].bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    axes[1].bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([f"ε={e}" for e in eps])
    axes[1].set_ylabel("Clip fraction (tokens outside [1−ε, 1+ε])")
    axes[1].set_title("Cached-batch clip fraction")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save(fig, out / "task2_cached_clipping")


def qualitative(cfg, evals, top_k):
    """Rank (not judge) candidates for manual reading; selection/annotation is done by a human."""
    root = repo_path(cfg["results_dir"]) / "eval"

    def load(n):
        p = root / n / "generations.jsonl"
        return {r["item_key"]: r for r in read_jsonl(p)} if p.exists() else None

    base = load("midpoint")
    if base is None:
        return {"status": "NOT RUN", "note": "midpoint evaluation missing"}
    out = {}
    eps, b = float(cfg["clip_epsilon"]), [float(x) for x in cfg["kl_values"]]
    for name in ["standard", fork_name(eps, min(b)), fork_name(eps, max(b))]:
        pol = load(name)
        if pol is None:
            continue
        pairs = []
        for k, r in pol.items():
            if k in base:
                m = base[k]
                pairs.append({"item_key": k, "reward_gain": r["reward_effective"] - m["reward_effective"],
                              "token_gain": r["response_tokens"] - m["response_tokens"],
                              "policy": {x: r[x] for x in ("reward_effective", "response_tokens", "terminated_with_eos", "response")},
                              "midpoint": {x: m[x] for x in ("reward_effective", "response_tokens", "terminated_with_eos", "response")}})
        pairs.sort(key=lambda d: d["reward_gain"], reverse=True)
        out[f"{name}_largest_reward_gain_vs_midpoint"] = pairs[:top_k]
        out[f"{name}_largest_reward_drop_vs_midpoint"] = pairs[::-1][:top_k]
        out[f"{name}_reward_gain_with_length_increase"] = [p for p in pairs if p["token_gain"] > 0][:top_k]
        hi = sorted(pol.values(), key=lambda r: r["reward_raw"], reverse=True)
        out[f"{name}_highest_reward_responses"] = hi[:top_k]
        out[f"{name}_highest_reward_truncated"] = [r for r in hi if not r["terminated_with_eos"]][:top_k]
    return out


def paired_vs_midpoint(cfg, names, n_boot: int = 5000):
    """Per-prompt paired effective-reward / length difference vs the midpoint on the same held-out prompts,
    with a percentile bootstrap 95% CI (seeded). Pairs share the prompt, which removes prompt difficulty
    from the comparison; single-sample generation noise remains."""
    root = repo_path(cfg["results_dir"]) / "eval"

    def load(n):
        p = root / n / "generations.jsonl"
        return {r["item_key"]: r for r in read_jsonl(p)} if p.exists() else None

    base = load("midpoint")
    if base is None:
        return {"status": "NOT RUN", "note": "midpoint evaluation missing"}
    rng = np.random.default_rng(int(cfg["seed"]))
    out = {"n_bootstrap": n_boot, "seed": int(cfg["seed"]), "by_model": {}}
    for n in names:
        g = load(n)
        if g is None or n == "midpoint":
            continue
        keys = sorted(k for k in base if k in g)
        d = np.array([g[k]["reward_effective"] - base[k]["reward_effective"] for k in keys])
        dl = np.array([g[k]["response_tokens"] - base[k]["response_tokens"] for k in keys])
        boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["by_model"][n] = {"n": len(keys), "reward_diff_mean": float(d.mean()), "reward_diff_ci95": [float(lo), float(hi)],
                              "length_diff_mean": float(dl.mean())}
    return out


def reward_length_correlation(cfg, names):
    root = repo_path(cfg["results_dir"]) / "eval"
    out = {}
    for n in names:
        p = root / n / "generations.jsonl"
        if p.exists():
            g = read_jsonl(p)
            rho, pval = spearmanr([r["response_tokens"] for r in g], [r["reward_raw"] for r in g])
            out[n] = {"spearman_length_vs_raw_reward": float(rho), "p_value": float(pval), "n": len(g)}
    return out


def build_summary(config_path: str = "configs/ppo.yaml") -> dict:
    cfg = load_yaml(config_path)
    eps0, beta0 = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    clip_names = [fork_name(float(e), beta0) for e in cfg["clip_values"]]
    kl_names = [fork_name(eps0, float(b)) for b in cfg["kl_values"]]
    names = ["midpoint", "standard"] + list(dict.fromkeys(clip_names + kl_names))
    runs, curves, evals = collect(cfg, names)
    root = repo_path(cfg["results_dir"])
    fig_dir = root / "figures"
    cached_path = root / "clipping" / "cached_batch_study.json"
    cached = load_json(cached_path) if cached_path.exists() else {"status": "NOT RUN"}

    rows = [summary_row(n, runs[n], curves.get(n), evals[n]) for n in names]
    paired = paired_vs_midpoint(cfg, names)
    for row in rows:
        pm = paired.get("by_model", {}).get(row["model"])
        if pm:
            row.update({"paired_reward_diff_vs_midpoint": pm["reward_diff_mean"],
                        "paired_ci95_low": pm["reward_diff_ci95"][0], "paired_ci95_high": pm["reward_diff_ci95"][1],
                        "paired_length_diff_vs_midpoint": pm["length_diff_mean"]})
    write_csv(root / "summary_table.csv", rows)
    write_csv(root / "clipping_forks.csv", [r for r in rows if r["model"] in clip_names])
    write_csv(root / "kl_forks.csv", [r for r in rows if r["model"] in kl_names])
    if "standard" in curves:
        write_csv(root / "standard_trajectory.csv",
                  [{k: v for k, v in c.items() if k not in ("epochs", "source_indices")} for c in curves["standard"]])
    if cached.get("status") == "COMPLETED":
        write_csv(root / "clipping_cached.csv",
                  [{"eps": e, **s, **{f"noise_only_{k}": v for k, v in cached["diagnostic_cached_vs_recomputed_old"]["by_eps"][e].items()}}
                   for e, s in cached["by_eps"].items()])

    fig_standard(curves.get("standard"), fig_dir)
    fig_forks(curves, clip_names, [f"ε={float(e):g}" for e in cfg["clip_values"]],
              f"Clipping forks ({cfg['fork_updates']} updates, β_KL={beta0:g})", "task2_clip_forks", fig_dir)
    fig_forks(curves, kl_names, [f"β_KL={float(b):g}" for b in cfg["kl_values"]],
              f"KL-pressure forks ({cfg['fork_updates']} updates, ε={eps0:g})", "task2_kl_forks", fig_dir)
    fig_cached(cached, fig_dir)
    save_json(root / "qualitative_candidates.json", qualitative(cfg, evals, int(cfg["qualitative_top_k"])))

    results = {
        "task": "task2_ppo",
        "note": f"Standard continuation: {cfg['updates']} updates; forks: {cfg['fork_updates']} updates from the same "
                f"midpoint. The eps={eps0:g}/kl_beta={beta0:g} fork is shared by the clipping and KL studies.",
        "standard_compute": {k: runs["standard"].get(k) for k in ("wall_clock_s", "peak_vram_gib", "peak_vram_reserved_gib", "updates")},
        "training_runs": runs,
        "training_curves": curves,
        "evaluation": evals,
        "clipping_cached_batch": cached,
        "studies": {"clipping_forks": clip_names, "kl_forks": kl_names},
        "paired_reward_vs_midpoint": paired,
        "reward_length_correlation": reward_length_correlation(cfg, names),
        "tables": {"summary": rows},
    }
    save_json(root / "results.json", results)
    print(f"Wrote {root / 'results.json'}")
    for r in rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                          if k in ("model", "status", "reward_effective", "kl_token_mean", "entropy", "resp_tokens_mean",
                                   "policy_shift_kl_max")}))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    build_summary(args.config)


if __name__ == "__main__":
    main()
