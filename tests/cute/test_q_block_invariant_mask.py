"""A Q-block promise must preserve Q-dependent masks, bounds and cache identity."""

import cutlass
import pytest
import torch
from cutlass import cute

from flash_attn.cute import utils


@cute.jit
def q_block_mask(batch, head, q, kv, seqlen_info, aux):
    return utils.scalar_to_ssa(
        aux[0][batch[0], head[0], q[0] // 256, kv[0]], cutlass.Boolean
    )


# The promise covers auxiliary reads as well as the returned value.
q_block_mask.__q_block_invariant__ = 256


@pytest.mark.parametrize("q_len,k_len", [(384, 256), (385, 257)])
def test_q_block_invariant_backward(q_len, k_len, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM10x backward mask specialization")
    from flash_attn.cute import interface
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    torch.manual_seed(817)
    b, h, d = 2, 2, 128
    q = torch.randn(b, q_len, h, d, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(b, k_len, h, d, device="cuda", dtype=q.dtype) for _ in range(2)]
    # Distinct masks for batch/head and Q blocks: using representative q=0 fails.
    positions = torch.arange(k_len, device="cuda")
    shifts = torch.arange(b * h * 2, device="cuda").reshape(b, h, 2, 1)
    keep = ((positions + shifts) % 3 != 0).contiguous()
    out, lse = interface._flash_attn_fwd(
        q,
        k,
        v,
        mask_mod=q_block_mask,
        aux_tensors=[keep],
        return_lse=True,
    )[:2]
    dout = torch.randn_like(out)
    dlse = torch.randn_like(lse) * 0.1
    qm, kn = (q_len + 127) // 128, (k_len + 127) // 128
    counts = torch.full((b, h, kn), qm, device="cuda", dtype=torch.int32)
    indices = (
        torch.arange(qm, device="cuda", dtype=torch.int32)
        .expand(b, h, kn, qm)
        .contiguous()
    )
    sparse = BlockSparseTensorsTorch(
        full_block_cnt=torch.zeros_like(counts),
        full_block_idx=torch.full_like(indices, -1),
        mask_block_cnt=counts,
        mask_block_idx=indices,
        block_size=(128, 128),
    )
    refs = [x.float().detach().requires_grad_() for x in (q, k, v)]
    score = (
        refs[0].transpose(1, 2) @ refs[1].transpose(1, 2).transpose(-1, -2) * d**-0.5
    )
    valid = keep[:, :, torch.arange(q_len, device="cuda") // 256, :]
    score = score.masked_fill(~valid, -torch.inf)
    ref_out = (score.softmax(-1) @ refs[2].transpose(1, 2)).transpose(1, 2)
    expected = torch.autograd.grad(
        (ref_out, score.logsumexp(-1)), refs, (dout.float(), dlse)
    )

    cache = {}
    monkeypatch.setattr(interface._flash_attn_bwd, "compile_cache", cache)
    cache_sizes = []
    for marker in (256, 0, 256):
        monkeypatch.setattr(q_block_mask, "__q_block_invariant__", marker)
        actual = interface._flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            dlse=dlse,
            mask_mod=q_block_mask,
            aux_tensors=[keep],
            block_sparse_tensors=sparse,
        )
        for ref, grad in zip(expected, actual):
            error = (grad.float() - ref).abs()
            assert torch.isfinite(grad).all()
            assert error.mean() < 1e-3
            assert error.max() / (ref.abs().mean() + 1e-6) < 0.25
        cache_sizes.append(len(cache))
    assert cache_sizes == [1, 2, 2]
