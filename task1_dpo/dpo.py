from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    Validate this implementation against the equation in the assignment manual
    before using it.
    """

    policy_margin = (
        policy_chosen_logp
        - policy_rejected_logp
    )

    ref_margin = (
        ref_chosen_logp
        - ref_rejected_logp
    )

    # DPO logit: beta * [(log pi(y+) - log ref(y+)) - (log pi(y-) - log ref(y-))]
    #          = beta * (policy_margin - ref_margin).
    # FIX: the released starter used `policy_margin + ref_margin`, which rewards the policy
    # for pairs the reference already prefers instead of measuring the policy's change
    # relative to the reference (at initialisation pi == ref, so the correct logit is 0
    # and the loss is log 2; the defective version gives 2 * beta * ref_margin).
    logits = beta * (
        policy_margin - ref_margin
    )

    loss = -F.logsigmoid(
        logits
    ).mean()

    return loss, {
        "logit_mean":
            logits.detach().mean(),

        "policy_margin_mean":
            policy_margin.detach().mean(),

        "preference_accuracy": (
            (policy_margin - ref_margin) > 0
        ).float().mean().detach(),
    }
