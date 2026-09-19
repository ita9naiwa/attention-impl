"""Disjoint KV partitions must use the original global softmax normalization."""
import pytest
import torch


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_backward_workspace_disjoint_partitions(dim, dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("workspace protocol currently supports SM10x")
    from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    torch.manual_seed(741 + dim)
    q, k, v = [torch.randn(1, 256, 2, dim, device="cuda", dtype=dtype) for _ in range(3)]
    out, lse = _flash_attn_fwd(q, k, v, return_lse=True)[:2]
    dout = torch.randn_like(out)
    dlse = torch.randn_like(lse) * 0.1

    def sparse(kv_tiles):
        # KV-owned metadata: every physical KV tile attends to both Q tiles.
        counts = torch.full((1, 2, kv_tiles), 2, device="cuda", dtype=torch.int32)
        indices = torch.arange(2, device="cuda", dtype=torch.int32).expand(1, 2, kv_tiles, 2).contiguous()
        return BlockSparseTensorsTorch(
            full_block_cnt=counts, full_block_idx=indices,
            mask_block_cnt=torch.zeros_like(counts), mask_block_idx=torch.full_like(indices, -1),
            block_size=(128, 128),
        )

    full = _flash_attn_bwd(q, k, v, out, dout, lse, dlse=dlse, block_sparse_tensors=sparse(2))
    parts = sparse(1)
    dq, dk0, dv0, workspace = _flash_attn_bwd(
        q, k[:, :128], v[:, :128], out, dout, lse,
        dlse=dlse, block_sparse_tensors=parts, _return_workspace=True,
    )
    with pytest.raises(AssertionError):
        _flash_attn_bwd(q, k[:, 128:], v[:, 128:], out, dout, lse.clone(),
                       dlse=dlse, block_sparse_tensors=parts, dq=dq, _workspace=workspace)
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(AssertionError):
            _flash_attn_bwd(q, k[:, 128:], v[:, 128:], out, dout, lse,
                           dlse=dlse, block_sparse_tensors=parts, dq=dq, _workspace=workspace)
    result, dk1, dv1 = _flash_attn_bwd(
        q, k[:, 128:], v[:, 128:], out, dout, lse,
        dlse=dlse, block_sparse_tensors=parts, dq=dq, _workspace=workspace,
    )
    assert result is dq
    actual = (dq, torch.cat((dk0, dk1), 1), torch.cat((dv0, dv1), 1))

    refs = [x.detach().float().requires_grad_() for x in (q, k, v)]
    scores = refs[0].transpose(1, 2) @ refs[1].transpose(1, 2).transpose(-1, -2) * dim**-0.5
    reference_out = (scores.softmax(-1) @ refs[2].transpose(1, 2)).transpose(1, 2)
    expected = torch.autograd.grad((reference_out, scores.logsumexp(-1)), refs, (dout.float(), dlse))
    for reference, ordinary, partitioned in zip(expected, full, actual):
        for gradient in (ordinary, partitioned):
            error = (gradient.float() - reference).abs()
            assert torch.isfinite(gradient).all()
            assert error.mean() < 1e-3
            assert error.max() / (reference.abs().mean() + 1e-6) < 0.25
