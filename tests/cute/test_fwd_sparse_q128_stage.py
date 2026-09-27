"""SM100 forward with a 128-row block-sparse Q block runs one Q stage and matches a dense masked reference."""

import pytest
import torch


def _sparse(keep, q_block):
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    kn = keep.shape[-1]
    idx = torch.where(keep, torch.arange(kn, device=keep.device, dtype=torch.int32), kn)
    idx = idx.sort(-1).values.clamp_max(kn - 1).to(torch.int32).contiguous()
    cnt = keep.sum(-1, dtype=torch.int32).contiguous()
    return BlockSparseTensorsTorch(
        full_block_cnt=cnt,
        full_block_idx=idx,
        mask_block_cnt=torch.zeros_like(cnt),
        mask_block_idx=torch.zeros_like(idx),
        block_size=(q_block, 128),
    )


def _q_stage(seqlen, sparse):
    from flash_attn.cute import interface

    return interface._get_fwd_config(
        arch=100, head_dim=128, head_dim_v=128, max_seqlen_q=seqlen, max_seqlen_k=seqlen, num_head_kv=2,
        qhead_per_kvhead=1, pack_gqa=False, batch_size=1, causal=False, local=False, window_size_left=None,
        window_size_right=None, num_splits=1, device=torch.device("cuda"), block_sparse_tensors=sparse,
    ).q_stage


@pytest.fixture
def sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM10x forward")


def test_q_stage_selection(sm100):
    n = 1024
    keep = torch.ones(1, 2, n // 128, n // 128, dtype=torch.bool, device="cuda")
    assert _q_stage(n, _sparse(keep, 128)) == 1
    assert _q_stage(n, _sparse(keep[:, :, ::2], 256)) == 2  # 256-row sparse Q blocks keep two stages
    assert _q_stage(n, None) == 2


def test_sparse_q128_forward_matches_dense_mask(sm100):
    from flash_attn.cute.interface import flash_attn_func

    torch.manual_seed(128)
    b, h, n, d = 1, 2, 1024, 128
    q, k, v = [torch.randn(b, n, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    keep = torch.rand(b, h, n // 128, n // 128, device="cuda") > 0.5
    keep[..., 0] = True
    out = flash_attn_func(q, k, v, block_sparse_tensors=_sparse(keep, 128))
    out = out[0] if isinstance(out, tuple) else out
    tok = keep.repeat_interleave(128, 2).repeat_interleave(128, 3)
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / d**0.5
    ref = torch.einsum("bhqk,bkhd->bqhd", s.masked_fill(~tok, float("-inf")).softmax(-1), v.float())
    # Neighbour-row selection (the 256-row doubling this replaces) gives errors of order 0.1-0.3.
    assert (out.float() - ref).abs().max().item() < 1e-2
