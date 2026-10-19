"""Per-token PPO loss of the RL update (dual-clip PPO + truncated importance sampling), kept free of any
distributed state so it can be unit-tested.

Dual clip (Ye et al., AAAI 2020): for A < 0 the clipped loss is
additionally capped at -A * dual_clip. TIS weight w = min(pi_old_train / pi_vllm, tis_cap), detached.
"""
import torch


def ppo_token_loss(lp, lp_old, lp_vllm, A, eps_low=0.2, eps_high=0.28, dual_clip=3.0, tis_cap=2.0):
    """lp: [T] log pi_theta (requires grad); lp_old: [T] log pi_old (trainer, round start);
    lp_vllm: [T] sampler logprobs; A: float advantage of the episode. Returns ([T] loss, stats)."""
    ratio = torch.exp(torch.clamp(lp - lp_old, -20.0, 20.0))
    pg1 = -A * ratio
    pg2 = -A * torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
    l = torch.maximum(pg1, pg2)
    if A < 0:
        l = torch.minimum(l, torch.full_like(l, -A * dual_clip))
    w = torch.clamp(torch.exp(lp_old - lp_vllm), max=tis_cap).detach()
    stats = {"clipfrac": float((pg2 > pg1).float().mean()), "approx_kl": float((lp_old - lp).detach().mean()),
             "ratio_mean": float(ratio.detach().mean()), "tis_mean": float(w.mean())}
    return l * w, stats
