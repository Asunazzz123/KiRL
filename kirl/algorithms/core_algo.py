# algorithm/core_algo.py
import torch
import torch.nn.functional as F




def compute_grpo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    reduction: str = "sequence_mean"
):
    assert 0 <= clip_eps < 1,"clip_eps must be in [0, 1)"
    assert reduction in {"sequence_mean", "token_mean"},"reduction must be 'sequence_mean' or 'token_mean'"

    mask = response_mask.bool()
    token_counts = mask.sum(dim=-1)
    valid_sequences = token_counts > 0

    log_ratio = torch.where(
        mask,
        log_probs.float() - old_log_probs.detach().float(),
        0.0,
    )
    ratio = log_ratio.exp()

    advantage = torch.where(
        mask,
        advantages.detach().float().unsqueeze(-1),
        0.0
    )

    unclipped = ratio * advantage
    clipped = ratio.clamp(1-clip_eps,1+clip_eps) * advantage
    token_loss = -torch.minimum(unclipped, clipped)
    token_loss = torch.where(mask,token_loss,0.0)

    if reduction == "sequence_mean":
        sequence_loss = (
            token_loss.sum(dim=-1) / token_counts.clamp_min(1)
        )
        loss = (
            sequence_loss.sum()
            / valid_sequences.sum().clamp_min(1)
        )
    else:
        loss = token_loss.sum() / mask.sum().clamp_min(1)

    with torch.no_grad():
        count = mask.sum().clamp_min(1)
        clip_active = clipped < unclipped
        metadata = {
            "policy/loss": loss.detach(),
            "policy/ratio_mean": torch.where(
                mask,ratio,0.0
            ).sum() / count,
            "policy/clip_fraction": (clip_active&mask).float().sum() / count
        }
    return loss, metadata


def compute_dpo_loss(
    chosen_log_probs: torch.Tensor,
    rejected_log_probs: torch.Tensor,
    reference_chosen_log_probs: torch.Tensor,
    reference_rejected_log_probs: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute standard sigmoid DPO loss over preference pairs.
    """
    tensors = (chosen_log_probs, rejected_log_probs,
               reference_chosen_log_probs, reference_rejected_log_probs)
    if chosen_log_probs.ndim != 1 or chosen_log_probs.numel() == 0:
        raise ValueError("chosen_log_probs must have shape [B] with B > 0")
    for tensor in tensors:
        if tensor.shape != chosen_log_probs.shape or tensor.device != chosen_log_probs.device:
            raise ValueError("all DPO inputs must share shape [B] and device")
        if not tensor.is_floating_point():
            raise TypeError("DPO log probabilities must be floating-point tensors")
    if not 0 < beta < float("inf"):
        raise ValueError("beta must be finite and positive")

    chosen_reward = beta * (
        chosen_log_probs.float() - reference_chosen_log_probs.detach().float()
    )
    rejected_reward = beta * (
        rejected_log_probs.float() - reference_rejected_log_probs.detach().float()
    )
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin).mean()

    with torch.no_grad():
        metadata = {
            "dpo/loss": loss.detach(),
            "reward/chosen": chosen_reward.mean(),
            "reward/rejected": rejected_reward.mean(),
            "reward/margin": margin.mean(),
            "reward/accuracy": (margin > 0).float().mean(),
        }
    return loss, metadata


def _validate_ppo_inputs(
    response_mask: torch.Tensor, **tensors: torch.Tensor,
) -> None:
    """
    Validate shape/device contracts without synchronizing accelerator values.
    """
    if response_mask.ndim != 2 or response_mask.shape[0] == 0:
        raise ValueError("response_mask must have shape [B, T] with B > 0")
    for name, tensor in tensors.items():
        if tensor.shape != response_mask.shape:
            raise ValueError(f"{name} must match response_mask shape [B, T]")
        if tensor.device != response_mask.device:
            raise ValueError(f"{name} must be on the same device as response_mask")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")


def _reduce_ppo_tokens(
    values: torch.Tensor, mask: torch.Tensor, reduction: str,
) -> torch.Tensor:
    """Reduce valid tokens; an entirely masked batch yields connected zero."""
    masked = torch.where(mask, values, 0.0)
    if reduction == "token_mean":
        return masked.sum() / mask.sum().clamp_min(1)
    if reduction == "sequence_mean":
        counts = mask.sum(dim=-1)
        means = masked.sum(dim=-1) / counts.clamp_min(1)
        return means.sum() / (counts > 0).sum().clamp_min(1)
    raise ValueError("reduction must be 'token_mean' or 'sequence_mean'")


def compute_ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    reduction: str = "token_mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute PPO-Clip actor loss, excluding entropy and reference KL terms.
    """
    _validate_ppo_inputs(
        response_mask, log_probs=log_probs, old_log_probs=old_log_probs,
        advantages=advantages,
    )
    if not 0 <= clip_eps < 1:
        raise ValueError("clip_eps must be in [0, 1)")

    mask = response_mask.bool()
    log_ratio = torch.where(
        mask, log_probs.float() - old_log_probs.detach().float(), 0.0,
    )
    ratio = log_ratio.exp()
    advantage = torch.where(mask, advantages.detach().float(), 0.0)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage
    loss = _reduce_ppo_tokens(-torch.minimum(unclipped, clipped), mask, reduction)

    with torch.no_grad():
        metadata = {
            "policy/loss": loss.detach(),
            "policy/ratio_mean": _reduce_ppo_tokens(ratio, mask, "token_mean"),
            "policy/clip_fraction": _reduce_ppo_tokens(
                (clipped < unclipped).float(), mask, "token_mean",
            ),
            # Rollout-policy KL diagnostic, not a reference-model KL penalty.
            "policy/approx_kl": _reduce_ppo_tokens(
                ratio - 1 - log_ratio, mask, "token_mean",
            ),
        }
    return loss, metadata


def compute_ppo_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    reduction: str = "token_mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute PPO clipped critic loss: 0.5 * max(raw MSE, clipped MSE).
    """
    _validate_ppo_inputs(
        response_mask, values=values, old_values=old_values, returns=returns,
    )
    if not 0 <= clip_eps < float("inf"):
        raise ValueError("clip_eps must be finite and non-negative")

    mask = response_mask.bool()
    current = torch.where(mask, values.float(), 0.0)
    previous = torch.where(mask, old_values.detach().float(), 0.0)
    target = torch.where(mask, returns.detach().float(), 0.0)
    clipped = previous + (current - previous).clamp(-clip_eps, clip_eps)
    raw_error = (current - target).square()
    clipped_error = (clipped - target).square()
    loss = 0.5 * _reduce_ppo_tokens(
        torch.maximum(raw_error, clipped_error), mask, reduction,
    )

    with torch.no_grad():
        metadata = {
            "value/loss": loss.detach(),
            "value/clip_fraction": _reduce_ppo_tokens(
                (clipped_error > raw_error).float(), mask, "token_mean",
            ),
            "value/mean": _reduce_ppo_tokens(current, mask, "token_mean"),
            "return/mean": _reduce_ppo_tokens(target, mask, "token_mean"),
        }
    return loss, metadata


def compute_dr_grpo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    max_completion_length: int,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute Dr. GRPO clipped policy loss without a KL penalty.
    """

    assert 0 <= clip_eps < 1,"clip_eps must be in [0, 1)"


    mask = response_mask.bool()
    log_ratio = torch.where(
        mask,
        log_probs.float() - old_log_probs.detach().float(),
        0.0,
    )
    ratio = log_ratio.exp()
    advantage = torch.where(mask, advantages.detach().float().unsqueeze(-1), 0.0)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage
    token_loss = torch.where(mask, -torch.minimum(unclipped, clipped), 0.0)

    loss = token_loss.sum() / (log_probs.shape[0] * max_completion_length)

    with torch.no_grad():
        count = mask.sum().clamp_min(1)
        metadata = {
            "policy/loss": loss.detach(),
            "policy/ratio_mean": torch.where(mask, ratio, 0.0).sum() / count,
            "policy/clip_fraction": ((clipped < unclipped) & mask).float().sum() / count,
        }
    return loss, metadata
