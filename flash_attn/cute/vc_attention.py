"""VC-Attention inference on FA4 Blackwell, arXiv:2609.15810.

Q/K Hadamard rotation, K centering and FP8 quantization (prepare), then ExpCast attention.
V-Smooth and token grouping are not implemented. RoPE belongs before this API.
"""

import torch

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.vc_preprocess import prepare


def attention_prepared(p, *, expcast=True, out=None):
    """Run the FA4 CuTe kernel; p comes from native CUDA prepare()."""
    return _flash_attn_fwd(
        p["q"],
        p["k"],
        p["v"],
        q_descale=p["qs"],
        k_descale=p["ks"],
        vc_vscale=p["vs"],
        vc_expcast=expcast,
        out=out,
    )[0].transpose(1, 2)


def vc_attention(q, k, v, *, expcast=True):
    """BHND post-RoPE inputs -> BHND BF16 output, forward-only.

    Inputs must be finite and share device and dtype.
    """
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise ValueError("VC-Attention is inference-only")
    return attention_prepared(prepare(q, k, v, smooth=False), expcast=expcast)
