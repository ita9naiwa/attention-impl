"""BHND and BNHD inputs must describe identical attention values; prepare() matches an exact CPU emulation."""

import torch
from flash_attn.cute.vc_preprocess import prepare


def _fp8(x):
    """cvt.rn.satfinite.e4m3: round to nearest even, saturate finite overflow and infinities to +-448, NaN -> canonical 0x7F."""
    return torch.where(x.isnan(), torch.full_like(x, float("nan")).abs(), x.clamp(-448, 448)).to(torch.float8_e4m3fn)


def _serial_sum(x, dim):
    total = torch.zeros_like(x.select(dim, 0))
    for i in range(x.shape[dim]):
        total = total + x.select(dim, i)
    return total


def _fmax_all(x, dims):
    """fmaxf reduction (NaN operands ignored; order-independent otherwise)."""
    for d in sorted(dims, reverse=True):
        x = torch.stack([x.select(d, i) for i in range(x.shape[d])]).nan_to_num(nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")).amax(0)
    return x


@torch.no_grad()
def _reference(q, k, v, bshd=False):
    """Exact CPU float32 emulation of prepare(smooth=False): Hadamard butterfly from the lowest channel bit up (lower
    index a + b, upper a - b) times the device rsqrtf(d); K sums serial over rows within each 128-row block, then serial
    over blocks; fmaxf/fminf statistics; IEEE divisions; satfinite E4M3 codes in BNHD."""
    q, k, v = [(x.transpose(1, 2) if bshd else x).float().cpu() for x in (q, k, v)]
    b, h, n, d = q.shape
    scale = torch.rsqrt(torch.tensor(float(d), device="cuda")).cpu()

    def rotate(x):
        bit = 1
        while bit < d:
            x = x.reshape(b, h, n, d // (2 * bit), 2, bit)
            lo, hi = x[..., 0, :], x[..., 1, :]
            x = torch.stack((lo + hi, lo - hi), -2).reshape(b, h, n, d)
            bit *= 2
        return x * scale

    rq, rk = rotate(q), rotate(k)
    nb = (n + 127) // 128
    blocks = [_serial_sum(rk[:, :, i * 128 : (i + 1) * 128], 2) for i in range(nb)]
    km = _serial_sum(torch.stack(blocks, 2), 2) / n
    lo = torch.fmin(torch.full_like(km, float("inf")), rk.nan_to_num(nan=float("inf"), posinf=float("inf"), neginf=-float("inf")).amin(2))
    hi = torch.fmax(torch.full_like(km, -float("inf")), rk.nan_to_num(nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")).amax(2))
    qm = torch.fmax(torch.zeros(b, h), _fmax_all(rq.abs(), (2, 3)))
    kr = torch.fmax((lo - km).abs(), (hi - km).abs())
    kr = torch.where(kr.isnan().all(-1, keepdim=True), float("nan"), kr.nan_to_num(nan=-float("inf"), posinf=float("inf"), neginf=-float("inf"))).amax(-1)
    vm = torch.fmax(torch.zeros(b, h, d), _fmax_all(v.abs(), (2,)))
    qs = torch.where(qm > 0, qm / 448, 1.0)
    ks = torch.where(kr > 0, kr / 448, 1.0)
    vs = torch.where(vm > 0, vm / 448, 1.0)
    codes = [_fp8(x).transpose(1, 2).contiguous() for x in (rq / qs[..., None, None], (rk - km[:, :, None]) / ks[..., None, None], v / vs[:, :, None])]
    return {"q": codes[0], "k": codes[1], "v": codes[2], "qs": qs, "ks": ks, "vs": vs}


def _assert_equal(actual, expected, label):
    for key, value in expected.items():
        got = actual[key].cpu()
        if value.dtype == torch.float8_e4m3fn:
            got, value = got.view(torch.uint8), value.view(torch.uint8)
        assert torch.equal(got, value), (label, key)


@torch.no_grad()
def test_fused_stats_preserve_row_order():
    torch.manual_seed(91)
    for d in (64, 128):
        for n in (31, 32, 33, 127, 129, 257):
            q, k, v = [
                torch.randn((1, 2, n, d), device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            if d == 128 and n in (33, 129, 257):
                # Contiguous views may still be misaligned for quartet loads.
                views = []
                for offset, value in enumerate((q, k, v), start=1):
                    storage = torch.empty(
                        value.numel() + offset, device=value.device, dtype=value.dtype
                    )
                    view = storage[offset:].view_as(value)
                    view.copy_(value)
                    views.append(view)
                q, k, v = views
            _assert_equal(prepare(q, k, v, smooth=False), _reference(q, k, v), (d, n))


def _exceptional(kind, shape, dtype, g):
    info = torch.finfo(dtype)
    x = torch.randn(shape, generator=g)
    if kind == "large":  # finite inputs whose Hadamard sums overflow
        x = x.sign() * info.max / (1 + 3 * torch.rand(shape, generator=g))
    elif kind == "zeros":
        x = torch.where(
            torch.rand(shape, generator=g) < 0.5, torch.tensor(0.0), torch.tensor(-0.0)
        )
    elif kind == "subnormal":
        x = x * info.smallest_normal * 0.25
    elif kind == "spikes":
        x = torch.where(
            torch.rand(shape, generator=g) < 1e-3, x.sign() * info.max / 2, x
        )
    x = x.to(dtype)
    if kind == "nonfinite":
        flat = x.view(-1)
        for n, i in enumerate(torch.randperm(flat.numel(), generator=g)[:24].tolist()):
            flat[i] = (float("nan"), float("inf"), float("-inf"))[n % 3]
    return x.cuda()


@torch.no_grad()
def test_fused_quantizer_exceptional_inputs_match_reference():
    # The fused butterflies (signed-unit FMA) must round exactly like plain add/sub, including overflow to
    # Inf/NaN, signed zeros and subnormals.
    g = torch.Generator().manual_seed(7)
    for kind in ("randn", "large", "zeros", "subnormal", "spikes", "nonfinite"):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for d in (64, 128):
                for bshd in (False, True):
                    shape = (2, 300, 3, d) if bshd else (2, 3, 300, d)
                    q, k, v = (_exceptional(kind, shape, dtype, g) for _ in range(3))
                    _assert_equal(prepare(q, k, v, smooth=False, bshd=bshd), _reference(q, k, v, bshd),
                                  (kind, dtype, d, bshd))


@torch.no_grad()
def test_preprocess_layout():
    torch.manual_seed(42)
    for d in (64, 128):
        inputs = [
            torch.randn((2, 129, 3, d), device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        expected = prepare(*(x.transpose(1, 2).contiguous() for x in inputs), smooth=False)
        actual = prepare(*inputs, smooth=False, bshd=True)
        for key in expected:
            assert torch.equal(actual[key].view(torch.uint8), expected[key].view(torch.uint8)), (d, key)
    # CUDA grid y/z cap: batch or heads > 65535 are rejected (the unfused 1-D grid fallback was removed).
    for shape in ((1, 1, 65536, 64), (65536, 1, 1, 64)):
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        try:
            prepare(x, x, x, smooth=False, bshd=True)
        except ValueError:
            continue
        raise AssertionError(f"{shape} did not raise")
    print("PASS native preprocessing: layouts, scales, batch/head grid limits")


@torch.no_grad()
def test_fused_dynamic_graph():
    torch.manual_seed(151)
    for d in (64, 128):
        for bshd in (False, True):
            shape = (2, 257, 3, d) if bshd else (2, 3, 257, d)
            values = [
                torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            options = {"smooth": False, "bshd": bshd}
            for _ in range(3):
                prepare(*values, **options)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = prepare(*values, **options)
            for _ in range(3):
                for value in values:
                    value.add_(0.02)
                for value in captured.values():
                    value.fill_(float("nan"))
                graph.replay()
                expected = prepare(*values, **options)
                for key in expected:
                    assert torch.equal(
                        captured[key].view(torch.uint8), expected[key].view(torch.uint8)
                    ), key
    print("PASS fused dynamic CUDA Graph: D64/128, BHND/BSHD, poisoned outputs")


if __name__ == "__main__":
    test_fused_stats_preserve_row_order()
    test_fused_quantizer_exceptional_inputs_match_reference()
    test_preprocess_layout()
    test_fused_dynamic_graph()
