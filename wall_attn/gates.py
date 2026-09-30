"""Gate preparation shared by training, prefill, and decode."""

import math

import torch


def soft_clamp_log_gates(log_gates: torch.Tensor, g_max: float = 0.86) -> torch.Tensor:
    """Soft-bound non-positive natural-log gates to [-g_max, 0].

    Apply once to ``F.logsigmoid(logits.float())`` before ``wall_attn`` or building decode prefixes. 
    Inputs must be log-retentions (<= 0), not logits!

    Follows Wall Attention blog post (https://blog.tilderesearch.com/blog/wall-attn).
    
    The blog derives a 0.87 per-token log-gate bound from the fp32 exponent of a 64-token tile (64 * 0.87 * log2(e) ~= 80 < 110 clamp).
    
    The default is 0.86 to be safe.
    """
    if not math.isfinite(g_max) or g_max <= 0:
        raise ValueError("g_max must be finite and positive")
    gates = log_gates if log_gates.dtype == torch.float64 else log_gates.float()
    return g_max * torch.expm1(gates / g_max)
