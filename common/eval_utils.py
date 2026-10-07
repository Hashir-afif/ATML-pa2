"""Evaluation helpers shared by Tasks 1-3 (added by the student; not part of the course release).

Moved here from task1_dpo/evaluate.py unchanged in behaviour so that Tasks 2 and 3 use the identical
generation + sampled-KL protocol instead of importing across task boundaries.
"""
from __future__ import annotations

import numpy as np
import torch

from common.generation import batch_generate, response_token_logprobs
from common.logging_utils import set_seed
from common.metrics import sampled_kl, word_count
from common.models import reference_mode


def length_stats(lengths) -> dict:
    a = np.asarray(lengths, dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None, "iqr": None}
    q25, med, q75 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "median": float(med), "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25),
            "min": float(a.min()), "max": float(a.max())}


def generate_with_kl(model, tokenizer, items: list[dict], cfg: dict, max_new_tokens: int, seed: int) -> list[dict]:
    """Sample one response per item and score it under policy and reference.

    KL uses the released sampled-response estimator (common.metrics.sampled_kl): response-token
    log pi - log ref on the policy's own samples. We store the per-response token sum and count so
    the pooled token-level mean (what sampled_kl returns on the whole set) and a per-sequence
    variant can both be recovered from disk. `policy_logp_sum` lets the course entropy helper
    (common.metrics.sample_entropy = -mean sampled log-prob) be recovered the same way.
    """
    gcfg = cfg["generation"]
    gen_bs, lp_bs = int(cfg["generation_batch_size"]), int(cfg["eval_batch_size"])
    set_seed(seed)
    out = []
    for start in range(0, len(items), gen_bs):
        chunk = items[start:start + gen_bs]
        gen = batch_generate(
            model, tokenizer, [it["messages"] for it in chunk],
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=max_new_tokens,
            temperature=float(gcfg["temperature"]), top_p=float(gcfg["top_p"]),
            do_sample=bool(gcfg["do_sample"]),
        )
        pw = gen["prompt_width"]
        for s in range(0, len(chunk), lp_bs):
            sl = slice(s, s + lp_bs)
            seq, attn = gen["sequences"][sl], gen["attention_mask"][sl]
            rid, rmask = gen["response_ids"][sl], gen["response_mask"][sl]
            with torch.no_grad():
                pol, _ = response_token_logprobs(model, seq, attn, pw, rid)
                with reference_mode(model):
                    ref, _ = response_token_logprobs(model, seq, attn, pw, rid)
            diff = (pol - ref) * rmask
            for j in range(seq.shape[0]):
                n_tok = float(rmask[j].sum())
                it = chunk[s + j]
                text = gen["responses"][s + j]
                rec = dict(it["meta"])
                rec.update({
                    "response": text,
                    "response_tokens": int(gen["response_lengths"][s + j]),
                    "terminated_with_eos": bool(gen["terminated_with_eos"][s + j]),
                    "truncated": bool(gen["truncated"][s + j]),
                    "word_count": word_count(text),
                    "kl_token_sum": float(diff[j].sum()),
                    "kl_token_count": n_tok,
                    "kl_token_mean": float(sampled_kl(pol[j], ref[j], rmask[j])),
                    "policy_logp_sum": float((pol[j] * rmask[j]).sum()),
                })
                out.append(rec)
            del pol, ref, diff
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out
