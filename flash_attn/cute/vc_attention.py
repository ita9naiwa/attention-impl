"""VC-Attention inference on FA4 Blackwell, arXiv:2609.15810.

Use one permutation/centroid pair per layer. Refresh grouping every four steps
in the first ceil(total_steps/4) steps, then retain the permutation and disable
smoothing. ExpCast remains enabled. RoPE belongs before this API.
"""

import torch

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.vc_preprocess import grouping, prepare


def attention_prepared(p, *, expcast=True, smooth=True, out=None):
    """Run the FA4 CuTe kernel; p comes from native CUDA prepare()."""
    return _flash_attn_fwd(
        p["q"],
        p["k"],
        p["v"],
        q_descale=p["qs"],
        k_descale=p["ks"],
        vc_vscale=p["vs"],
        vc_mean=p["means"] if smooth else None,
        vc_expcast=expcast,
        out=out,
    )[0].transpose(1, 2)


def vc_attention(q, k, v, *, permutation=None, smooth=True, expcast=True, hadamard=True):
    """BHND post-RoPE inputs -> BHND BF16 output, forward-only.

    Pass a cached permutation to amortize grouping. Without a permutation,
    grouping runs when smooth=True. Inputs must be finite; permutation must
    contain every token exactly once. All inputs must share device and dtype.
    """
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise ValueError("VC-Attention is inference-only")
    if permutation is None and smooth:
        permutation, _ = grouping(v)
    p = prepare(q, k, v, permutation=permutation, smooth=smooth, hadamard=hadamard)
    return attention_prepared(p, expcast=expcast, smooth=smooth)
