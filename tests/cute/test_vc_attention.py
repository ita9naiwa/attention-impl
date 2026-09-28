"""Run on a Blackwell GPU: python tests/cute/test_vc_attention.py."""

import math

import torch

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.vc_attention import attention_prepared, vc_attention
from flash_attn.cute.vc_preprocess import prepare


def _hadamard(d, device):
    """Orthonormal Sylvester Hadamard matrix (the order of prepare()'s butterfly)."""
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < d:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(d)


@torch.no_grad()
def test_native_preparation():
    torch.manual_seed(91)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for shape in [(2, 3, 129, 64), (1, 2, 513, 128)]:
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for bshd in (False, True):
                q, k, v = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
                h = _hadamard(shape[-1], "cuda")
                rq, rk = q.float() @ h, k.float() @ h
                rk = rk - rk.mean(2, keepdim=True)
                vf = v.float()
                inputs = [x.transpose(1, 2).contiguous() for x in (q, k, v)] if bshd else (q, k, v)
                p = prepare(*inputs, smooth=False, bshd=bshd)
                assert set(p) == {"q", "k", "v", "qs", "ks", "vs"}
                torch.testing.assert_close(p["qs"], rq.abs().amax((2, 3)) / 448, rtol=2e-6, atol=1e-8)
                torch.testing.assert_close(p["ks"], rk.abs().amax((2, 3)) / 448, rtol=2e-6, atol=1e-8)
                vs = vf.abs().amax(2) / 448
                torch.testing.assert_close(p["vs"], torch.where(vs > 0, vs, 1.0), rtol=2e-6, atol=1e-8)
                for name, original, scale in [
                    ("q", rq, p["qs"][:, :, None, None]),
                    ("k", rk, p["ks"][:, :, None, None]),
                    ("v", vf, p["vs"][:, :, None, :]),
                ]:
                    decoded = p[name].permute(0, 2, 1, 3).float() * scale
                    error = (decoded - original).norm() / original.norm().clamp_min(1e-20)
                    assert error < 0.04, (shape, dtype, bshd, name, error.item())
    q, k = [torch.randn(1, 2, 257, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    zero = prepare(q, k, torch.zeros_like(q), smooth=False)
    assert zero["v"].float().count_nonzero().item() == 0 and bool((zero["vs"] == 1).all())
    const = prepare(q, k, torch.full_like(q, 2.5), smooth=False)
    assert bool((const["v"].float() == 448).all())
    print("PASS native CUDA preprocessing: 12 shape/dtype/layout cases, constants")


def test_prepare_removed_modes_raise():
    q = torch.randn(1, 2, 129, 64, device="cuda", dtype=torch.bfloat16)
    permutation = torch.arange(129, device="cuda").expand(1, 2, 129).contiguous()
    default, explicit = prepare(q, q, q), prepare(q, q, q, smooth=False)
    assert set(default) == set(explicit) and all(
        torch.equal(default[k].view(torch.uint8) if default[k].dtype == torch.float8_e4m3fn else default[k],
                    explicit[k].view(torch.uint8) if explicit[k].dtype == torch.float8_e4m3fn else explicit[k])
        for k in default
    ), "the default call must equal smooth=False"
    for kwargs in ({"smooth": True}, {"permutation": permutation}, {"hadamard": False}):
        try:
            prepare(q, q, q, **kwargs)
        except NotImplementedError:
            continue
        raise AssertionError(f"prepare{kwargs} did not raise")
    big = torch.zeros(65536, 1, 1, 64, device="cuda", dtype=torch.bfloat16)
    try:
        prepare(big, big, big, smooth=False)
    except ValueError:
        pass
    else:
        raise AssertionError("b > 65535 did not raise")
    print("PASS default prepare() == smooth=False; removed modes raise")


@torch.no_grad()
def reference(q, k, v, qs, ks, vs, softmax_scale, code_mode="double_fma"):
    q, k, v = [x.float().transpose(1, 2) for x in (q, k, v)]
    b, h, n, _d = q.shape
    maximum = torch.full((b, h, n), -torch.inf, device=q.device)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(q)
    # Match the kernel's FP32 scale multiplication order. The probability
    # code path fuses BOTH the bias subtraction and the score multiply/add.
    # A separately rounded bias can cross an ExpCast byte boundary (seed 821).
    base_scale = torch.tensor(
        softmax_scale, device=q.device, dtype=torch.float32
    ) * math.log2(math.e)
    factor = (base_scale * (qs * ks))[..., None]
    for block in range((n + 127) // 128 - 1, -1, -1):
        start = block * 128
        raw = q @ k[:, :, start : start + 128].transpose(-1, -2)
        new_maximum = torch.maximum(maximum, raw.amax(-1))
        alpha = torch.exp2((maximum - new_maximum) * factor)
        u = (raw - new_maximum[..., None]) * factor[..., None]
        if code_mode == "literal":
            c = 8 * u + 119.65
        else:
            factor8 = factor * 8
            bias = 119.65 - new_maximum * factor8
            if code_mode == "double_fma":
                bias = (
                    torch.tensor(119.65, device=q.device, dtype=torch.float32).double()
                    - new_maximum.double() * factor8.double()
                ).float()
            # A Float32 product/add fits exactly in Float64 before the one
            # final Float32 rounding, emulating FMA independently of CuTe.
            c = (
                raw.double() * factor8[..., None].double() + bias[..., None].double()
            ).float()
        codes = torch.round(c).clamp(0, 120).to(torch.uint8)
        probabilities = codes.view(torch.float8_e4m3fn).float()
        row_sum = probabilities.sum(-1)
        numerator = numerator * alpha[..., None] + probabilities @ v[:, :, start : start + 128]
        denominator = denominator * alpha + row_sum
        maximum = new_maximum
    return numerator / denominator[..., None] * vs[:, :, None, :]


@torch.no_grad()
def check(q, k, v, qs, ks, vs, scale, label):
    expected = reference(q, k, v, qs, ks, vs, scale)
    actual = (
        _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=scale,
            q_descale=qs,
            k_descale=ks,
            vc_expcast=True,
            vc_vscale=vs,
        )[0]
        .transpose(1, 2)
        .float()
    )
    # One BF16 ULP accommodates adjacent rounding at FP32 accumulation boundaries.
    # An explicit small FP32 accumulation allowance handles outputs near zero.
    exponent = torch.floor(
        torch.log2(expected.abs().clamp_min(torch.finfo(torch.bfloat16).tiny))
    )
    ulp = torch.exp2(exponent) * torch.finfo(torch.bfloat16).eps
    value_bound = (v.float().transpose(1, 2).abs().amax(2) * vs).amax(-1)
    tolerance = ulp + 8e-6 * (1 + value_bound[:, :, None, None])
    error = (actual - expected).abs()
    maximum_ratio = (error / tolerance).amax().item()
    assert torch.isfinite(actual).all() and maximum_ratio <= 1, (
        label,
        maximum_ratio,
        error.amax().item(),
    )
    print(
        label,
        "max_bound_ratio",
        maximum_ratio,
        "max_abs",
        error.amax().item(),
        flush=True,
    )


@torch.no_grad()
def test_expcast_reference():
    torch.manual_seed(2301)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for b, h, n, d in ((2, 2, 129, 64), (1, 2, 257, 128), (1, 1, 513, 128)):
        q, k, v = [
            torch.randn((b, n, h, d), device="cuda").to(torch.float8_e4m3fn)
            for _ in range(3)
        ]
        qs = torch.full((b, h), 0.25, device="cuda")
        ks = torch.full_like(qs, 0.5)
        vs = torch.exp2(torch.randint(-1, 2, (b, h, d), device="cuda").float())
        check(q, k, v, qs, ks, vs, 1 / math.sqrt(d), f"random {b, h, n, d}")
    # Place score probabilities on either side of byte-rounding boundaries,
    # including zero/subnormal transition and upper normal-code cells.
    b, h, n, d = 1, 1, 257, 128
    q = torch.zeros(b, n, h, d, device="cuda")
    q[:, :, :, 0] = 1
    k = torch.zeros_like(q)
    k[:, :, :, 0] = -1
    k[:, 0, :, 0] = 0
    q, k = [x.to(torch.float8_e4m3fn) for x in (q, k)]
    v = torch.randn(b, n, h, d, device="cuda").to(torch.float8_e4m3fn)
    qs = ks = torch.ones(b, h, device="cuda")
    vs = torch.ones(b, h, d, device="cuda")
    for code_target in (0.49, 0.51, 7.49, 7.51, 63.49, 63.51, 119.49, 119.51):
        scale = (119.65 - code_target) / (8 * math.log2(math.e))
        check(q, k, v, qs, ks, vs, scale, f"code boundary {code_target}")
    check(q, k, torch.zeros_like(v.float()).to(torch.float8_e4m3fn), qs, ks, vs, 1.0, "zero V")
    print(
        "PASS descending Eq.6/7 reference within BF16 ULP + bounded FP32 accumulation error",
        flush=True,
    )


@torch.no_grad()
def test_expcast_seed821_byte_boundary():
    # Seed 821 / N257 / D64 regresses the fused-bias byte-boundary case.
    # Keep arbitrary FP8 Q/K: no grid restriction or relaxed output tolerance.
    torch.manual_seed(821)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for d in (64, 128):
        for n in (1, 128, 129, 257, 384, 513, 1025):
            b, h = 1, 2
            q, k, v = [
                torch.randn(b, n, h, d, device="cuda").to(torch.float8_e4m3fn)
                for _ in range(3)
            ]
            qs = torch.full((b, h), 0.25, device="cuda")
            ks = torch.full_like(qs, 0.5)
            vs = torch.exp2(torch.randint(-1, 2, (b, h, d), device="cuda").float())
            check(q, k, v, qs, ks, vs, 1 / math.sqrt(d), f"seed821 {n=} {d=}")


@torch.no_grad()
def test_native_pipeline():
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for shape in ((2, 2, 129, 64), (2, 2, 257, 128)):
        q, k, v = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        )
        p = prepare(q, k, v, smooth=False)
        for expcast in (False, True):
            actual = attention_prepared(p, expcast=expcast)
            relative = (actual.float() - expected).norm() / expected.norm()
            assert relative.item() < 0.09, (shape, expcast, relative.item())
            assert torch.isfinite(actual).all()
            print("pipeline", shape, expcast, relative.item(), flush=True)
    shape = (1, 2, 384, 128)
    q, k = [torch.zeros(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    zero = torch.zeros_like(v)
    for expcast in (False, True):
        torch.testing.assert_close(vc_attention(q, k, zero, expcast=expcast), zero, rtol=0, atol=0)
    # Capture only after both NVRTC and CuTe compilation have been warmed.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            expected = vc_attention(q, k, v)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = vc_attention(q, k, v)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("PASS native CUDA + FA4 CuTe pipeline, zero V, CUDA Graph", flush=True)


def _vbs_vector_mask():
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute.mask import r2p_bitmask_below

    @cute.jit
    def mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        base = n_idx[0]
        limit = aux_tensors[0][base // 128] - base % 128
        packed = cute.make_rmem_tensor(4, cutlass.Uint32)
        for word in cutlass.range_constexpr(4):
            packed[word] = r2p_bitmask_below(limit, word)
        return packed.load()

    mask.__vec_size__ = 128
    return mask


@torch.no_grad()
def test_full_inner_hwmax_exact():
    """FA_VC_FULL_INNER_HWMAX (default on) must reproduce the software-row-max VC Q256 sparse kernel (=0) bit for bit."""
    if torch.cuda.get_device_capability() != (10, 3):
        print("SKIP full-inner hwmax: requires SM103")
        return
    from flash_attn.cute import interface, utils
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    records = []
    original_kernel, original_flag = interface.FlashAttentionForwardSm100, utils._fa_vc_full_inner_hwmax_enabled

    class Traced(original_kernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            records.append(self)

    interface.FlashAttentionForwardSm100 = Traced
    original_cache, interface._flash_attn_fwd.compile_cache = interface._flash_attn_fwd.compile_cache, {}
    mask = _vbs_vector_mask()
    heads, nk = 2, 10
    torch.manual_seed(7)

    def lists(sizes, sq, pattern):
        # pattern[h][m] -> selected physical KV blocks; classify by size like the VC wrapper.
        mb = (sq + 255) // 256
        idx = {n: torch.zeros(1, heads, mb, nk, dtype=torch.int32) for n in ("full", "mask")}
        cnt = {n: torch.zeros(1, heads, mb, dtype=torch.int32) for n in ("full", "mask")}
        for h in range(heads):
            for m in range(mb):
                for b in sorted(pattern[h][m]):
                    kind = "full" if sizes[b] == 128 else "mask" if sizes[b] > 0 else None
                    if kind:
                        idx[kind][0, h, m, cnt[kind][0, h, m]] = b
                        cnt[kind][0, h, m] += 1
        return [t.cuda() for t in (cnt["mask"], idx["mask"], cnt["full"], idx["full"])]

    def payload(kind, sq, sk):
        if kind == "random":
            q = torch.randn(1, sq, heads, 128, device="cuda") * 120
            k = torch.randn(1, sk, heads, 128, device="cuda") * 120
        else:  # extrema, signed zeros, ties, all-negative rows, per-tile changing maxima
            q = torch.full((1, sq, heads, 128), 448.0, device="cuda")
            q[:, 1::3] = -0.0
            q[:, 2::5, :, ::2] = -448.0
            level = -(torch.arange(sk, device="cuda") // 128 % 4).float() * 112 - 1
            k = level[None, :, None, None].expand(1, sk, heads, 128).clone()
            k[:, 3::7] = 448.0
            k[:, 5::11] = 0.0
        v = torch.randn(1, sk, heads, 128, device="cuda") * 200
        return [t.clamp(-448, 448).to(torch.float8_e4m3fn).contiguous() for t in (q, k, v)]

    patterns = {  # per head, per Q256 block; includes empty, masked-only and >=3-full lists
        "a": [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 4, 6, 8], []], [[0, 2, 4, 6, 8], [0, 1, 2, 3, 9], [5]]],
        "b": [[[9, 7, 5, 3, 1, 0], [2], [0, 2, 3, 4, 5, 6]], [[1, 4], [], [0, 3, 6, 7, 8]]],
    }
    cases = [  # (payload, sq, sk, sizes, pattern)
        ("random", 768, 1280, [128] * 10, "a"),
        ("random", 768, 1280, [128, 77, 128, 128, 0, 128, 128, 1, 128, 128], "a"),
        ("extreme", 768, 1280, [128, 128, 128, 5, 128, 128, 128, 128, 64, 128], "b"),
        ("random", 728, 1280, [128, 128, 128, 128, 128, 90, 128, 128, 128, 128], "b"),  # Q edge
        ("random", 768, 1230, [128] * 9 + [78], "a"),  # ragged K
    ]
    qs = torch.full((1, heads), 1e-3, device="cuda")
    ks = torch.tensor([[1e-3, 3e-4]], device="cuda")
    vs = torch.rand(1, heads, 128, device="cuda") + 0.5
    try:
        for kind, sq, sk, sizes, pat in cases:
            q, k, v = payload(kind, sq, sk)
            aux = torch.tensor(sizes, device="cuda", dtype=torch.int32)
            mc, mi, fc, fi = lists(sizes, sq, patterns[pat])
            sparse = BlockSparseTensorsTorch(mc, mi, fc, fi, block_size=(256, 128))

            def run():
                return interface._flash_attn_fwd(
                    q, k, v, q_descale=qs, k_descale=ks, vc_vscale=vs, vc_expcast=True,
                    tile_mn=(128, 128), max_seqlen_q=sq, mask_mod=mask,
                    block_sparse_tensors=sparse, aux_tensors=[aux], return_lse=True,
                )[:2]

            results = []
            for flag in (False, True):
                utils._fa_vc_full_inner_hwmax_enabled = flag
                before = len(records)
                results.append([t.clone() for t in run()])
                if len(records) > before:  # freshly compiled kernel object
                    assert records[-1].vc_sparse_stats_overlap and records[-1].vc_full_inner_hwmax == flag
            (out0, lse0), (out1, lse1) = results
            assert torch.equal(out0, out1) and torch.equal(lse0, lse1), (kind, sq, sk, sizes)
            assert torch.isfinite(out1.float()).all()
            empty = (mc + fc)[0, :, :].repeat_interleave(256, -1)[:, :sq] == 0  # (H, Sq)
            assert torch.count_nonzero(out1[0].transpose(0, 1)[empty]) == 0
            print("full-inner hwmax exact", kind, sq, sk, flush=True)
        assert {r.vc_full_inner_hwmax for r in records} == {False, True}, len(records)
        # Changed auxiliary sizes, lists and V under CUDA Graph replay of the opted-in kernel.
        utils._fa_vc_full_inner_hwmax_enabled = True
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for sizes2, pat2 in (([128] * 10, "a"), ([128, 128, 3, 128, 128, 128, 0, 128, 128, 128], "b")):
            aux.copy_(torch.tensor(sizes2, device="cuda", dtype=torch.int32))
            for dst, src in zip((mc, mi, fc, fi), lists(sizes2, sq, patterns[pat2])):
                dst.copy_(src)
            v.copy_((-v.float()).to(v.dtype))
            for t in captured:
                t.fill_(float("nan"))
            graph.replay()
            utils._fa_vc_full_inner_hwmax_enabled = False
            ref = run()
            utils._fa_vc_full_inner_hwmax_enabled = True
            assert all(torch.equal(a, b) for a, b in zip(captured, ref)), sizes2
        # NaN-poisoned K payload (E4M3FN has no inf encoding): unsupported input, but the opt-in must
        # match the software path bit for bit (NaN positions included) so behavior equals baseline.
        sizes = [128] * 10
        aux.copy_(torch.tensor(sizes, device="cuda", dtype=torch.int32))
        for dst, src in zip((mc, mi, fc, fi), lists(sizes, sq, patterns["a"])):
            dst.copy_(src)
        q, k, v = payload("random", sq, sk)
        results_clean = [t.clone() for t in run()]
        k.view(torch.uint8)[0, 300, 0, 5] = 0x7F  # inner full tile of head 0
        k.view(torch.uint8)[0, 1000:1003, 1, :] = 0xFF  # whole rows, head 1
        results = []
        for flag in (False, True):
            utils._fa_vc_full_inner_hwmax_enabled = flag
            results.append([t.clone() for t in run()])
        (out0, lse0), (out1, lse1) = results
        print("NaN poison: baseline out NaNs", out0.float().isnan().sum().item(), "LSE NaNs", lse0.isnan().sum().item(),
              "changed vs clean", not torch.equal(out0, results_clean[0]), flush=True)
        for a, b in ((out0.float(), out1.float()), (lse0, lse1)):
            assert torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(), b.nan_to_num())
        print("PASS full-inner hwmax: exact out/LSE across full/masked/empty, Q/K edges, graph replay, NaN poison", flush=True)
    finally:
        interface.FlashAttentionForwardSm100 = original_kernel
        interface._flash_attn_fwd.compile_cache = original_cache
        utils._fa_vc_full_inner_hwmax_enabled = original_flag


def test_alias_guard_hint_exact():
    """alias_guard hint (None/True/False) on the block-sparse Q256 forward and the single-stage Q128 forward (128-row
    sparse Q blocks), BF16 and VC ExpCast: out/LSE bit-identical.

    The guard only shrinks the persistent grid (148 -> 146 CTAs on B300) when the Q-block count and the SM count divide
    each other. Only True engages it (None/False keep the SM-count grid). Q = 37/74/148 alias, Q = 150 is the control.
    """
    if torch.cuda.get_device_capability() != (10, 3):
        print("SKIP alias guard hint: requires SM103")
        return
    import json, os
    from torch.profiler import ProfilerActivity, profile
    from flash_attn.cute import interface
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    sm = torch.cuda.get_device_properties(0).multi_processor_count
    mask = _vbs_vector_mask()
    heads, nk, prefix = 8, 6, 3  # heads x Q > 146 so the capped grid is reached; dense prefix rows as on H3
    sk = nk * 128
    aux = torch.tensor([128] * (nk - 1) + [77], device="cuda", dtype=torch.int32)
    vc_args = dict(q_descale=torch.full((1, heads), 1e-3, device="cuda"), k_descale=torch.full((1, heads), 1e-3, device="cuda"),
                   vc_vscale=torch.rand(1, heads, 128, device="cuda") + 0.5, vc_expcast=True)

    def fwd_grid(fn):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        path = f"/tmp/alias-guard-hint-{os.getpid()}.json"
        prof.export_chrome_trace(path)
        events = json.load(open(path))["traceEvents"]
        os.remove(path)
        grids = {e["args"].get("grid", [0])[0] for e in events
                 if e.get("cat") == "kernel" and "forward" in e["name"].lower()}
        assert len(grids) == 1, grids
        return grids.pop()

    for q_tile, vc in [(t, c) for t in (256, 128) for c in (False, True)]:
        for q_blocks in (37, 74, 148, 150):
            sq = q_blocks * q_tile
            torch.manual_seed(q_blocks)
            if vc:
                q, k, v = [(torch.randn(1, n, heads, 128, device="cuda") * 120).clamp(-448, 448).to(torch.float8_e4m3fn)
                           for n in (sq, sk, sk)]
            else:
                q, k, v = [torch.randn(1, n, heads, 128, device="cuda", dtype=torch.bfloat16) for n in (sq, sk, sk)]
            selected = torch.rand(heads, q_blocks, nk, device="cuda") < 0.4
            selected[:, :prefix] = True
            order = torch.arange(nk, device="cuda", dtype=torch.int32).expand(heads, q_blocks, nk)

            def packed(sel):
                idx = torch.where(sel, order, nk).sort(-1).values
                return sel.sum(-1, dtype=torch.int32)[None].contiguous(), idx.masked_fill(idx == nk, 0).to(torch.int32)[None].contiguous()

            (mc, mi), (fc, fi) = packed(selected & (aux < 128) & (aux > 0)), packed(selected & (aux == 128))
            sparse = BlockSparseTensorsTorch(mc, mi, fc, fi, block_size=(q_tile, 128))
            aliased = q_blocks % sm == 0 or sm % q_blocks == 0
            case = (q_tile, "vc" if vc else "bf16", q_blocks)
            results = {}
            for hint in (None, True, False):
                def run():
                    return interface._flash_attn_fwd(
                        q, k, v, tile_mn=(128, 128), max_seqlen_q=sq, mask_mod=mask, block_sparse_tensors=sparse,
                        aux_tensors=[aux], return_lse=True, alias_guard=hint, **(vc_args if vc else {}),
                    )[:2]
                results[hint] = [t.clone() for t in run()]
                engaged = aliased and hint is True
                grid = fwd_grid(run)
                assert grid == (sm - 2 if engaged else sm), (case, hint, grid)
            for hint in (True, False):
                pairs = zip(results[None], results[hint])
                assert all(torch.equal(a, b) for a, b in pairs), (case, hint)
            assert torch.isfinite(results[None][0].float()).all()
            print("alias guard hint exact", *case, flush=True)
    print("PASS alias guard hint: BF16 and VC out/LSE bit-identical across None/True/False", flush=True)


if __name__ == "__main__":
    test_native_preparation()
    test_prepare_removed_modes_raise()
    test_expcast_reference()
    test_expcast_seed821_byte_boundary()
    test_native_pipeline()
    test_full_inner_hwmax_exact()
    test_alias_guard_hint_exact()
