"""Delayed-scaling VC preparation (caller-owned VCScaleState): cold/fallback exactness, steady-state accuracy, Graph."""

import pytest
import torch

from flash_attn.cute.vc_preprocess import VCScaleState, prepare

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="SM100 VC preprocessing",
)
KEYS = ("q", "k", "v", "qs", "ks", "vs")


def _inputs(n=1000, h=4, d=128, gain=None, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = [torch.randn(1, n, h, d, device="cuda", generator=g) for _ in range(3)]
    if gain is not None:
        x = [t * gain.view(1, 1, h, 1) for t in x]
    x[1] = x[1] + 0.5  # nonzero K mean exercises centering
    return [t.to(torch.bfloat16).contiguous() for t in x]


def _equal(a, b):
    return all(
        torch.equal(
            a[key].view(torch.uint8) if a[key].dtype == torch.float8_e4m3fn else a[key],
            b[key].view(torch.uint8) if b[key].dtype == torch.float8_e4m3fn else b[key],
        )
        for key in KEYS
    )


def _centered_score_error(p, q, k):
    # Hadamard rotation is orthonormal and K centering is a per-row shift: compare row-centered Q K^T.
    ref = torch.einsum("bnhd,bmhd->bhnm", q.float(), k.float())
    got = torch.einsum(
        "bnhd,bmhd->bhnm",
        p["q"].float() * p["qs"][:, None, :, None],
        p["k"].float() * p["ks"][:, None, :, None],
    )
    ref, got = ref - ref.mean(-1, keepdim=True), got - got.mean(-1, keepdim=True)
    return ((got - ref).norm() / ref.norm()).item()


def _v_error(p, v):
    return (
        (p["v"].float() * p["vs"][:, None]).sub(v.float()).norm() / v.float().norm()
    ).item()


@pytest.mark.parametrize("d", [64, 128])
def test_cold_path_and_stats_are_exact(d):
    x = _inputs(d=d)
    state = VCScaleState()
    assert _equal(
        prepare(*x, smooth=False, bshd=True, scale_state=state),
        prepare(*x, smooth=False, bshd=True),
    )  # T1
    y = _inputs(d=d, seed=1)
    prepare(
        *y, smooth=False, bshd=True, scale_state=state
    )  # steady state recomputes this call's statistics
    fresh = VCScaleState()
    prepare(
        *y, smooth=False, bshd=True, scale_state=fresh
    )  # cold init on the same input
    for name in (
        "qs",
        "ks",
        "vs",
        "kmean",
    ):  # T6: fused_quantize_stats slabs == fused_stats slabs
        assert torch.equal(getattr(state, name), getattr(fresh, name)), name


@pytest.mark.parametrize("d", [64, 128])
def test_steady_state_accuracy_without_fallback(d):  # T2
    state = VCScaleState()
    prepare(*_inputs(d=d, seed=10), smooth=False, bshd=True, scale_state=state)
    for step in range(6):
        x = _inputs(d=d, seed=11 + step)
        delayed = prepare(*x, smooth=False, bshd=True, scale_state=state)
        fresh = prepare(*x, smooth=False, bshd=True)
        assert _centered_score_error(
            delayed, x[0], x[1]
        ) <= 1.1 * _centered_score_error(fresh, x[0], x[1])
        assert _v_error(delayed, x[2]) <= 1.1 * _v_error(fresh, x[2]) + 1e-4
        assert all(torch.isfinite(delayed[key]).all() for key in ("qs", "ks", "vs"))
    assert state.fallbacks.item() == 0 and state.saturations.item() == 0


def test_gross_change_falls_back_bit_exact():  # T3
    state = VCScaleState()
    prepare(*_inputs(seed=20), smooth=False, bshd=True, scale_state=state)
    gain = torch.tensor(
        [4.0, 1.0, 0.2, 1.0], device="cuda"
    )  # heads 0 (clips) and 2 (wastes bits) must fall back
    x = _inputs(seed=21, gain=gain)
    delayed = prepare(*x, smooth=False, bshd=True, scale_state=state)
    fresh = prepare(*x, smooth=False, bshd=True)
    assert (
        state.fallbacks.item() >= 2 and state.saturations.item() > 0
    )  # head 0 saturates before its fallback
    for head in (0, 2):
        assert torch.equal(
            delayed["q"][:, :, head].view(torch.uint8),
            fresh["q"][:, :, head].view(torch.uint8),
        )
        assert torch.equal(
            delayed["k"][:, :, head].view(torch.uint8),
            fresh["k"][:, :, head].view(torch.uint8),
        )
        assert torch.equal(
            delayed["v"][:, :, head].view(torch.uint8),
            fresh["v"][:, :, head].view(torch.uint8),
        )
        assert torch.equal(delayed["qs"][:, head], fresh["qs"][:, head])
        assert torch.equal(delayed["vs"][:, head], fresh["vs"][:, head])


def test_bypass_and_shape_change():  # T4
    state = VCScaleState()
    x = _inputs()
    prepare(*x, smooth=False, bshd=True, scale_state=state)
    smooth = prepare(*x, smooth=True, bshd=True, scale_state=state)
    ref = prepare(*x, smooth=True, bshd=True)
    assert _equal(smooth, ref) and torch.equal(smooth["means"], ref["means"])
    y = _inputs(n=777)
    assert _equal(
        prepare(*y, smooth=False, bshd=True, scale_state=state),
        prepare(*y, smooth=False, bshd=True),
    )
    assert state.signature[4] == 777


def test_shared_state_misuse_falls_back():  # T8: alternating distributions through one state
    state = VCScaleState()
    a = torch.ones(4, device="cuda")
    for step in range(4):
        x = _inputs(seed=30 + step, gain=a * (8.0 if step % 2 else 1.0))
        out = prepare(*x, smooth=False, bshd=True, scale_state=state)
        if step:
            assert _equal(out, prepare(*x, smooth=False, bshd=True))
    assert state.fallbacks.item() == 3 * 4


def test_graph_replay_matches_eager():  # T5
    state = VCScaleState()
    x = _inputs(seed=40)
    prepare(*x, smooth=False, bshd=True, scale_state=state)
    prepare(*_inputs(seed=41), smooth=False, bshd=True, scale_state=state)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = prepare(*x, smooth=False, bshd=True, scale_state=state)
    for step in range(3):
        y = _inputs(seed=42 + step)
        for dst, src in zip(x, y):
            dst.copy_(src)
        mirror = VCScaleState()
        mirror._init(
            state.signature,
            state.kmean.clone(),
            state.qs.clone(),
            state.ks.clone(),
            state.vs.clone(),
        )
        expected = prepare(*x, smooth=False, bshd=True, scale_state=mirror)
        for key in ("q", "k", "v"):
            captured[key].view(torch.uint8).fill_(0x7F)
        graph.replay()
        assert _equal(captured, expected)
    assert state.fallbacks.item() == 0


@pytest.mark.parametrize("d", [64, 128])
def test_power_of_two_margin_matches_cold_dequant(d):
    # Unchanged statistics (same input: same amax and K mean): the 2x margins only shift the E4M3 exponent, so the
    # dequantized warm values equal the cold ones wherever the cold code is normal-safe (|code| >= 2^-5). Below that
    # the warm payload can round differently in the subnormal tail; those elements are counted, not asserted away.
    x = _inputs(d=d, seed=50)
    state = VCScaleState()
    cold = prepare(*x, smooth=False, bshd=True, scale_state=state)
    warm = prepare(*x, smooth=False, bshd=True, scale_state=state)
    assert state.fallbacks.item() == 0 and state.saturations.item() == 0
    tail = {}
    for key, scale in (("q", "qs"), ("k", "ks"), ("v", "vs")):
        assert torch.equal(warm[scale], 2 * cold[scale])
        shape = (1, 1, -1, 1) if scale != "vs" else (1, 1, *cold[scale].shape[1:])
        c = cold[key].float() * cold[scale].view(shape)
        w = warm[key].float() * warm[scale].view(shape)
        safe = cold[key].float().abs() >= 2.0**-5
        assert torch.equal(w[safe], c[safe]), key
        tail[key] = int((w != c).sum())
    print("subnormal-tail decoded differences", tail)


def test_state_reuse_across_head_dim_and_heads():
    # A state reused across D128 <-> D64 or a (B, H) change re-initializes every buffer and matches a fresh state.
    state = VCScaleState()
    for h, d, seed in (
        (4, 128, 60),
        (4, 64, 61),
        (4, 128, 62),
        (2, 128, 63),
        (2, 64, 64),
    ):
        first, second = _inputs(h=h, d=d, seed=seed), _inputs(h=h, d=d, seed=seed + 100)
        assert _equal(
            prepare(*first, smooth=False, bshd=True, scale_state=state),
            prepare(*first, smooth=False, bshd=True),
        )
        assert (
            state.kmean.shape == (1, h, d)
            and state.vs.shape == (1, h, d)
            and state.qs.shape == (1, h)
        )
        mirror = VCScaleState()
        prepare(*first, smooth=False, bshd=True, scale_state=mirror)
        assert _equal(
            prepare(*second, smooth=False, bshd=True, scale_state=state),
            prepare(*second, smooth=False, bshd=True, scale_state=mirror),
        )
