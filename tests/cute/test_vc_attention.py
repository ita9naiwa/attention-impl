"""Run on a Blackwell GPU: python tests/cute/test_vc_attention.py."""

import math

import torch

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.vc_attention import attention_prepared, vc_attention
from flash_attn.cute.vc_preprocess import _rotate_qk, grouping, prepare


@torch.no_grad()
def test_native_preparation():
    torch.manual_seed(91)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for shape in [(2, 3, 129, 64), (1, 2, 513, 128)]:
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            q, k, v = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
            _b, _h, n, d = shape
            permutation = torch.rand(shape[:-1], device="cuda").argsort(-1)
            for smooth in (False, True):
                p = prepare(
                    q, k, v, permutation=permutation, smooth=smooth, hadamard=False
                )
                gather = permutation[..., None].expand_as(q)
                vf = v.float().gather(2, gather)
                kf = (k.float() - k.float().mean(2, keepdim=True)).gather(2, gather)
                means = (
                    torch.stack(
                        [vf[:, :, s : s + 128].mean(2) for s in range(0, n, 128)], dim=2
                    )
                    if smooth
                    else torch.zeros_like(p["means"], dtype=torch.float32)
                )
                residual = vf - means.repeat_interleave(128, dim=2)[:, :, :n]
                vs = residual.abs().amax(2) / 448
                vs = torch.where(vs > 0, vs, 1.0)
                torch.testing.assert_close(
                    p["qs"], q.float().abs().amax((2, 3)) / 448, rtol=1e-6, atol=1e-8
                )
                torch.testing.assert_close(
                    p["ks"], kf.abs().amax((2, 3)) / 448, rtol=1e-6, atol=1e-8
                )
                torch.testing.assert_close(p["vs"], vs, rtol=2e-6, atol=1e-8)
                # Validate FP32 normalization independently, then exact BF16 storage.
                high_precision = prepare(
                    q,
                    k,
                    v,
                    permutation=permutation,
                    smooth=smooth,
                    hadamard=False,
                    mean_dtype=torch.float32,
                )
                torch.testing.assert_close(
                    high_precision["means"] * high_precision["vs"][:, :, None, :],
                    means,
                    rtol=2e-5,
                    atol=1e-6,
                )
                assert p["means"].dtype == torch.bfloat16
                torch.testing.assert_close(
                    p["means"], high_precision["means"].bfloat16(), rtol=0, atol=0
                )
                for name, original, scale in [
                    ("q", q.float(), p["qs"][:, :, None, None]),
                    ("k", kf, p["ks"][:, :, None, None]),
                    ("v", residual, p["vs"][:, :, None, :]),
                ]:
                    decoded = p[name].permute(0, 2, 1, 3).float() * scale
                    error = (decoded - original).norm() / original.norm().clamp_min(
                        1e-20
                    )
                    assert error < 0.04, (shape, dtype, smooth, name, error.item())
    for d in (64, 128):
        q, k, v = [
            torch.randn(1, 2, 129, d, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        rq, rk = _rotate_qk(q, k)
        original = q.float() @ k.float().transpose(-1, -2)
        rotated = rq @ rk.transpose(-1, -2)
        torch.testing.assert_close(rotated, original, rtol=2e-5, atol=3e-5)
        p = prepare(q, k, v)
        torch.testing.assert_close(
            p["qs"], rq.abs().amax((2, 3)) / 448, rtol=1e-6, atol=1e-8
        )
        centered = rk - rk.mean(2, keepdim=True)
        torch.testing.assert_close(
            p["ks"], centered.abs().amax((2, 3)) / 448, rtol=1e-6, atol=1e-8
        )
    q, k = [
        torch.randn(1, 2, 257, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    for constant in (0.0, 2.5):
        p = prepare(q, k, torch.full_like(q, constant))
        assert p["v"].float().count_nonzero().item() == 0
        torch.testing.assert_close(
            p["means"] * p["vs"][:, :, None, :],
            torch.full_like(p["means"], constant).float(),
            rtol=0,
            atol=0,
        )
    # Cancellation-sensitive residuals must use the original block mean.
    q = torch.zeros(1, 1, 128, 64, device="cuda", dtype=torch.float32)
    center = torch.arange(64, device="cuda", dtype=torch.float32) * 8 + 4096
    delta = (torch.arange(128, device="cuda") % 2 * 2 - 1).float()[:, None]
    amplitude = torch.arange(1, 65, device="cuda", dtype=torch.float32)[None, :] / 1024
    v = (center + delta * amplitude)[None, None].contiguous()
    p = prepare(q, q, v, hadamard=False)
    expected = ((v - center) / p["vs"][:, :, None, :]).to(torch.float8_e4m3fn)
    torch.testing.assert_close(
        p["v"].permute(0, 2, 1, 3).float(), expected.float(), rtol=0, atol=0
    )
    q = torch.randn(1, 2, 257, 128, device="cuda", dtype=torch.bfloat16)
    permutation, centroids = grouping(q, clusters=4)
    torch.testing.assert_close(
        permutation.sort(-1).values, torch.arange(257, device="cuda").expand(1, 2, 257)
    )
    assert centroids.shape == (1, 2, 4, 128)
    print(
        "PASS native CUDA preprocessing: 12 shape/dtype/smoothing cases, constants, grouping, Hadamard invariance"
    )


@torch.no_grad()
def reference(q, k, v, qs, ks, vs, means, softmax_scale, code_mode="double_fma"):
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
        numerator = (
            numerator * alpha[..., None]
            + probabilities @ v[:, :, start : start + 128]
            + row_sum[..., None] * means[:, :, block, None, :]
        )
        denominator = denominator * alpha + row_sum
        maximum = new_maximum
    return numerator / denominator[..., None] * vs[:, :, None, :]


@torch.no_grad()
def check(q, k, v, qs, ks, vs, means, scale, label):
    expected = reference(q, k, v, qs, ks, vs, means, scale)
    actual = (
        _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=scale,
            q_descale=qs,
            k_descale=ks,
            vc_expcast=True,
            vc_mean=means,
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
    value_bound = (
        (v.float().transpose(1, 2).abs().amax(2) + means.abs().amax(2)) * vs
    ).amax(-1)
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
        means = torch.randn(b, h, (n + 127) // 128, d, device="cuda") * 0.3
        check(q, k, v, qs, ks, vs, means, 1 / math.sqrt(d), f"random {b, h, n, d}")
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
    means = torch.randn(b, h, 3, d, device="cuda") * 0.25
    for code_target in (0.49, 0.51, 7.49, 7.51, 63.49, 63.51, 119.49, 119.51):
        scale = (119.65 - code_target) / (8 * math.log2(math.e))
        check(q, k, v, qs, ks, vs, means, scale, f"code boundary {code_target}")
    zeros = torch.zeros_like(v.float()).to(torch.float8_e4m3fn)
    for constant in (0.0, 2.5):
        check(
            q,
            k,
            zeros,
            qs,
            ks,
            vs,
            torch.full_like(means, constant),
            1.0,
            f"constant {constant}",
        )
    print(
        "PASS descending Eq.6/7 reference within BF16 ULP + bounded FP32 accumulation error",
        flush=True,
    )


@torch.no_grad()
def test_bf16_mean_reference():
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
            means = (
                torch.randn(b, h, (n + 127) // 128, d, device="cuda") * 0.3
            ).bfloat16()
            check(q, k, v, qs, ks, vs, means, 1 / math.sqrt(d), f"BF16 means {n=} {d=}")


@torch.no_grad()
def test_persistent_mean_synchronization():
    # 16 * ceil(8193 / 512) work tiles exceed 148 B300 SMs, forcing CTA reuse.
    # Compare identical quantized inputs; no quadratic dense reference needed.
    torch.manual_seed(9821)
    b, n, h, d = 1, 8193, 16, 128
    cases = []
    for _ in range(2):
        q, k, v = [
            torch.randn(b, n, h, d, device="cuda").to(torch.float8_e4m3fn)
            for _ in range(3)
        ]
        qs = torch.full((b, h), 0.25, device="cuda")
        ks = torch.full_like(qs, 0.5)
        vs = torch.exp2(torch.randint(-1, 2, (b, h, d), device="cuda").float())
        means = (torch.randn(b, h, (n + 127) // 128, d, device="cuda") * 0.3).bfloat16()
        kwargs = {
            "softmax_scale": 1 / math.sqrt(d),
            "q_descale": qs,
            "k_descale": ks,
            "vc_expcast": True,
            "vc_vscale": vs,
        }
        expected = _flash_attn_fwd(q, k, v, vc_mean=means.float(), **kwargs)[0].float()
        exponent = torch.floor(
            torch.log2(expected.abs().clamp_min(torch.finfo(torch.bfloat16).tiny))
        )
        ulp = torch.exp2(exponent) * torch.finfo(torch.bfloat16).eps
        value_bound = (
            (v.float().transpose(1, 2).abs().amax(2) + means.float().abs().amax(2)) * vs
        ).amax(-1)
        tolerance = ulp + 8e-6 * (1 + value_bound[:, None, :, None])
        out = torch.empty((b, n, h, d), device="cuda", dtype=torch.bfloat16)
        cases.append((q, k, v, means, kwargs, expected, tolerance, out))

    def launch(case):
        q, k, v, means, kwargs, _, _, out = case
        _flash_attn_fwd(q, k, v, vc_mean=means, out=out, **kwargs)

    def verify(case):
        *_, expected, tolerance, out = case
        ratio = ((out.float() - expected).abs() / tolerance).amax().item()
        assert torch.isfinite(out).all() and ratio <= 1, ratio

    for iteration in range(3):
        case = cases[iteration % 2]
        case[-1].fill_(float("nan"))
        launch(case)
        verify(case)

    # All kernels and output buffers are warmed before graph capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for case in cases:
            launch(case)
    for _ in range(3):
        for case in cases:
            case[-1].fill_(float("nan"))
        graph.replay()
        for case in cases:
            verify(case)
    print(
        "PASS persistent CTA reuse: alternating buffers and CUDA Graph, BF16 versus FP32 means",
        flush=True,
    )


@torch.no_grad()
def test_vsmooth_reference():
    """Exact-exp V-Smooth oracle: FP32 row mass, E4M3 PV, FA4 stale maximum."""
    torch.manual_seed(7319)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    def oracle(q, k, v, qs, ks, vs, means):
        q, k, v = [x.float().transpose(1, 2) for x in (q, k, v)]
        b, h, n, d = q.shape
        base = torch.tensor(
            1 / math.sqrt(d), device=q.device, dtype=torch.float32
        ) * math.log2(math.e)
        factor = (base * (qs * ks))[..., None]
        maximum = torch.full((b, h, n), -torch.inf, device=q.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(q)
        for block in range((n + 127) // 128 - 1, -1, -1):
            start = block * 128
            raw = q @ k[:, :, start : start + 128].transpose(-1, -2)
            proposed = torch.maximum(maximum, raw.amax(-1))
            change = (maximum - proposed) * factor
            # FA4 keeps its old maximum if rescaling would change it <=4 bits.
            keep = change >= -4
            maximum = torch.where(keep, maximum, proposed)
            alpha = torch.where(keep, 1.0, torch.exp2(change))
            # Both bias and score affine operations compile as FP32 FMAs.
            bias = (4.0 - maximum.double() * factor.double()).float()
            z = (
                raw.double() * factor[..., None].double() + bias[..., None].double()
            ).float()
            probability = torch.exp2(z)
            mass = probability.sum(-1)
            packed = probability.to(torch.float8_e4m3fn).float()
            numerator = (
                numerator * alpha[..., None]
                + packed @ v[:, :, start : start + 128]
                + mass[..., None] * means[:, :, block, None, :].float()
            )
            denominator = denominator * alpha + mass
        return numerator / denominator[..., None] * vs[:, :, None, :]

    for dtype in (torch.bfloat16, torch.float32):
        for d in (64, 128):
            for n, force_rescale in (
                (1, False),
                (128, False),
                (129, False),
                (257, False),
                (513, False),
                (257, True),
            ):
                b, h = 1, 2
                q, k, v = [
                    torch.randn(b, n, h, d, device="cuda").to(torch.float8_e4m3fn)
                    for _ in range(3)
                ]
                qs = torch.full((b, h), 0.25, device="cuda")
                ks = torch.full_like(qs, 0.5)
                if force_rescale:
                    qf, kf = torch.zeros_like(q.float()), torch.zeros_like(k.float())
                    qf[..., 0] = 1
                    kf[..., 0] = torch.linspace(32, -32, n, device="cuda")[
                        None, :, None
                    ]
                    q, k = qf.to(torch.float8_e4m3fn), kf.to(torch.float8_e4m3fn)
                    qs.fill_(1)
                    ks.fill_(1)
                vs = torch.exp2(torch.randint(-1, 2, (b, h, d), device="cuda").float())
                means = (
                    torch.randn(b, h, (n + 127) // 128, d, device="cuda") * 0.3
                ).to(dtype)
                expected = oracle(q, k, v, qs, ks, vs, means)
                actual = (
                    _flash_attn_fwd(
                        q,
                        k,
                        v,
                        q_descale=qs,
                        k_descale=ks,
                        vc_expcast=False,
                        vc_mean=means,
                        vc_vscale=vs,
                    )[0]
                    .transpose(1, 2)
                    .float()
                )
                exponent = torch.floor(
                    torch.log2(
                        expected.abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
                    )
                )
                ulp = torch.exp2(exponent) * torch.finfo(torch.bfloat16).eps
                value_bound = (
                    (
                        v.float().transpose(1, 2).abs().amax(2)
                        + means.float().abs().amax(2)
                    )
                    * vs
                ).amax(-1)
                tolerance = ulp + 8e-6 * (1 + value_bound[:, :, None, None])
                ratio = ((actual - expected).abs() / tolerance).amax().item()
                label = f"V-Smooth {dtype} {n=} {d=} {force_rescale=}"
                assert torch.isfinite(actual).all() and ratio <= 1, (label, ratio)
                print(label, "max_bound_ratio", ratio, flush=True)


@torch.no_grad()
def test_native_pipeline():
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for shape in ((2, 2, 129, 64), (2, 2, 257, 128)):
        q, k, v = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        permutation = torch.rand(shape[:-1], device="cuda").argsort(-1)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        )
        for smooth in (False, True):
            p = prepare(q, k, v, permutation=permutation, smooth=smooth)
            for expcast in (False, True):
                actual = attention_prepared(p, expcast=expcast, smooth=smooth)
                relative = (actual.float() - expected).norm() / expected.norm()
                assert relative.item() < 0.09, (shape, smooth, expcast, relative.item())
                assert torch.isfinite(actual).all()
                print("pipeline", shape, smooth, expcast, relative.item(), flush=True)
    # Uniform P isolates V reconstruction from QK and probability approximation.
    # Three exactly sized groups have large opposing centers and small residuals.
    shape = (1, 2, 384, 128)
    q, k = [torch.zeros(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    centers = torch.randn((1, 2, 2, 128), device="cuda") * 12
    centers = torch.cat((centers, -centers.sum(2, keepdim=True)), dim=2)
    labels = torch.arange(384, device="cuda") % 3
    v = (centers[:, :, labels] + 0.05 * torch.randn(shape, device="cuda")).bfloat16()
    permutation = labels.argsort(stable=True).expand(1, 2, 384).contiguous()
    plain = prepare(q, k, v, permutation=permutation, smooth=False)
    smoothed = prepare(q, k, v, permutation=permutation, smooth=True)
    torch.testing.assert_close(plain["qs"], smoothed["qs"], rtol=0, atol=0)
    torch.testing.assert_close(plain["ks"], smoothed["ks"], rtol=0, atol=0)
    target = v.float().mean(2, keepdim=True).expand_as(v)
    for expcast in (False, True):
        raw = attention_prepared(plain, expcast=expcast, smooth=False).float()
        restored = attention_prepared(smoothed, expcast=expcast, smooth=True).float()
        raw_error, restored_error = [
            (x - target).square().mean().sqrt().item() for x in (raw, restored)
        ]
        assert restored_error < raw_error * 0.25, (expcast, raw_error, restored_error)
        print("smoothing_RMSE", expcast, raw_error, restored_error, flush=True)
    for value in (0.0, 2.5):
        const = torch.full_like(v, value)
        for expcast in (False, True):
            out = vc_attention(q, k, const, permutation=permutation, expcast=expcast)
            torch.testing.assert_close(out, const, rtol=0, atol=0)
    # Capture only after both NVRTC and CuTe compilation have been warmed.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            expected = vc_attention(q, k, v, permutation=permutation)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = vc_attention(q, k, v, permutation=permutation)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(
        "PASS native CUDA + FA4 CuTe pipeline, smoothing improvement, constants, CUDA Graph",
        flush=True,
    )


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
    test_expcast_reference()
    test_bf16_mean_reference()
    test_persistent_mean_synchronization()
    test_vsmooth_reference()
    test_native_pipeline()
    test_full_inner_hwmax_exact()
    test_alias_guard_hint_exact()
