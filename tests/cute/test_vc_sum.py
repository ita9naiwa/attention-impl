"""Large dense ExpCast denominator with nonconstant values and a ragged tail."""

import pytest
import torch

from flash_attn.cute import interface


@torch.no_grad()
def test_large_zero_query_uniform_value_mean(monkeypatch):
    if not torch.cuda.is_available() or interface._get_device_arch() != 103:
        pytest.skip("B300 large TC-sum specialization")
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", {})

    for n in (16384, 32384, 32416, 65536, 65537):
        shape = (1, n, 2, 128)
        q = torch.zeros(shape, device="cuda", dtype=torch.float8_e4m3fn)
        # With Q=0 every valid P code is identical: output is the decoded V mean.
        # The last value makes silently dropping the ragged final token detectable.
        values = ((torch.arange(n, device="cuda") % 2) * 2 - 1).float()
        values[-1] = 64.0
        v = (
            values[None, :, None, None]
            .expand(shape)
            .contiguous()
            .to(torch.float8_e4m3fn)
        )
        vs = torch.exp2((torch.arange(128, device="cuda") % 3 - 1).float())
        vs = vs[None, None, :].expand(1, 2, 128).contiguous()
        qs = ks = torch.ones((1, 2), device="cuda")
        out = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
        interface._flash_attn_fwd(
            q,
            q,
            v,
            q_descale=qs,
            k_descale=ks,
            vc_vscale=vs,
            vc_expcast=True,
            num_splits=1,
            out=out,
        )
        # Sum representable FP8 values in FP64, independent of online block traversal.
        expected = (v.double().mean(dim=1) * vs.double()).float()[:, None, :, :]
        value_bound = v.float().abs().amax(dim=1) * vs
        ulp = (
            torch.exp2(
                torch.floor(
                    torch.log2(
                        expected.abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
                    )
                )
            )
            * torch.finfo(torch.bfloat16).eps
        )
        tolerance = ulp + 8e-6 * (1 + value_bound[:, None, :, :])
        actual = out.float()
        maximum_ratio = ((actual - expected).abs() / tolerance).amax().item()
        assert torch.isfinite(actual).all() and maximum_ratio <= 1, (n, maximum_ratio)


@torch.no_grad()
def test_tc_sum_threshold_and_equal_length_cache(monkeypatch):
    if not torch.cuda.is_available() or interface._get_device_arch() != 103:
        pytest.skip("B300 TC-sum eligibility")
    from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100 as Kernel

    original = Kernel.__init__
    observed = []

    def spy(self, *args, **kwargs):
        original(self, *args, **kwargs)
        observed.append(self)

    monkeypatch.setattr(Kernel, "__init__", spy)
    short, equal, unequal = (16383, 16383), (16384, 16384), (16384, 16385)
    for order in ((short, equal, unequal), (equal, unequal, short)):
        monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", {})
        observed.clear()
        sizes = []
        for nq, nk in order * 2:
            q = torch.zeros((1, nq, 1, 128), device="cuda", dtype=torch.float8_e4m3fn)
            k = torch.zeros((1, nk, 1, 128), device="cuda", dtype=torch.float8_e4m3fn)
            v = torch.full((1, nk, 1, 128), 0.5, device="cuda").to(torch.float8_e4m3fn)
            out = torch.full(q.shape, float("nan"), device="cuda", dtype=torch.bfloat16)
            interface._flash_attn_fwd(q, k, v, vc_expcast=True, num_splits=1, out=out)
            assert torch.isfinite(out).all() and torch.all(out == 0.5), (nq, nk)
            sizes.append(len(interface._flash_attn_fwd.compile_cache))
        expected = [nq == nk and nq >= 16384 for nq, nk in order[:2]]
        assert [kernel.vc_tc_sum for kernel in observed] == expected
        assert sizes == [1, 2, 2, 2, 2, 2], sizes
