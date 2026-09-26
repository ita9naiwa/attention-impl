import itertools
import json

import torch
from flash_attn.cute.vc_preprocess import prepare
from flash_attn.cute.vc_vsa_preprocess import VCScaleState, prepare_vsa


def pad_pool_reference(values, source_map, sizes, block, query_map, query_tokens):
    padded, pools = [], []
    valid = source_map >= 0
    for index, value in enumerate(values):
        full = value.new_zeros(value.shape[0], source_map.numel(), *value.shape[2:])
        full[:, valid] = value[:, source_map[valid].long()]
        pooled = (
            full.float().reshape(value.shape[0], -1, block, *value.shape[2:]).sum(2)
        )
        pools.append((pooled / sizes.clamp_min(1)[None, :, None, None]).transpose(1, 2))
        if index == 0:
            compact = value.new_zeros(value.shape[0], query_tokens, *value.shape[2:])
            compact[:, query_map[query_map >= 0].long()] = full[:, query_map >= 0]
            padded.append(compact)
        else:
            padded.append(full)
    return padded, pools


def make_inputs(block, d, dtype, b=2, h=3, metadata_dtypes=(torch.int64,) * 3):
    sizes = torch.tensor(
        [37, block, block - 35, block - 1], device="cuda", dtype=torch.int64
    )
    source_n = int(sizes.sum())
    source_map = torch.full((4 * block,), -1, device="cuda", dtype=torch.int64)
    permutation = torch.randperm(source_n, device="cuda")
    start = 0
    for i, size in enumerate(sizes.tolist()):
        source_map[i * block : i * block + size] = permutation[start : start + size]
        start += size
    query_map = torch.arange(4 * block, device="cuda") - block + 17
    query_map[:block] = -1
    source_map, query_map, sizes = (
        x.to(t) for x, t in zip((source_map, query_map, sizes), metadata_dtypes)
    )
    values = [
        torch.randn((b, source_n, h, d), device="cuda", dtype=dtype) for _ in range(3)
    ]
    return values, source_map, sizes, query_map


def case(block, d, dtype, b=2, h=3, metadata_dtypes=(torch.int64,) * 3):
    values, source_map, sizes, query_map = make_inputs(block, d, dtype, b, h, metadata_dtypes)
    p, pools = prepare_vsa(
        *values,
        source_map,
        sizes,
        block,
        padded_to_query=query_map,
        query_tokens=3 * block,
        query_offset=17,
    )
    local_map = torch.where(query_map >= 0, query_map - 17, -1)
    padded, expected_pools = pad_pool_reference(
        values, source_map, sizes, block, local_map, 3 * block
    )
    q, k, v = padded
    q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, block))
    expected = prepare(q, k, v, smooth=False, bshd=True)
    expected["q"] = expected["q"][:, : 3 * block]
    pool_error = []
    quant_error = {}
    for actual, ref in zip(pools, expected_pools):
        torch.testing.assert_close(actual, ref, rtol=3e-5, atol=3e-6)
        pool_error.append((actual - ref).abs().max().item())
    for key in ["qs", "ks", "vs"]:
        torch.testing.assert_close(p[key], expected[key], rtol=2e-6, atol=1e-7)
    for key in ["q", "k", "v"]:
        x, y = p[key].float(), expected[key].float()
        error = (x - y).norm() / y.norm().clamp_min(1e-20)
        assert error < 0.002, (block, d, dtype, key, error)
        quant_error[key] = error.item()
    print(
        json.dumps(
            {
                "case": [block, d, str(dtype), b, h],
                "metadata_dtypes": [str(x) for x in metadata_dtypes],
                "pool_max_abs": pool_error,
                "fp8_relative_l2_vs_bridge": quant_error,
            }
        ),
        flush=True,
    )
    return values, source_map, sizes, query_map, p, expected


@torch.no_grad()
def test_vsa_preparation():
    torch.manual_seed(20260918)
    with torch.no_grad():
        for block in (128, 256):
            for d in (64, 128):
                for dtype in (torch.bfloat16, torch.float16, torch.float32):
                    case(block, d, dtype)
        for metadata_dtypes in itertools.product((torch.int32, torch.int64), repeat=3):
            case(128, 64, torch.bfloat16, 1, 2, metadata_dtypes)
        values, m, s, qm, _, _ = case(
            128, 128, torch.bfloat16, 1, 2, (torch.int32, torch.int32, torch.int64)
        )
        prefix, _ = prepare_vsa(
            *values, m, s, 128, padded_to_query=torch.full_like(qm, -1), query_tokens=0
        )
        assert prefix["q"].shape == (1, 0, 2, 128) and torch.equal(
            prefix["qs"], torch.ones_like(prefix["qs"])
        )
        print("prefix-only document PASS", flush=True)
        for bad in (0.5, True, -1, 2147483648):
            try:
                prepare_vsa(
                    *values,
                    m,
                    s,
                    128,
                    padded_to_query=qm,
                    query_tokens=384,
                    query_offset=bad,
                )
            except ValueError:
                pass
            else:
                raise AssertionError(("bad query_offset accepted", bad))
        for _ in range(3):
            prepare_vsa(
                *values,
                m,
                s,
                128,
                padded_to_query=qm,
                query_tokens=384,
                query_offset=17,
            )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, pools = prepare_vsa(
                *values,
                m,
                s,
                128,
                padded_to_query=qm,
                query_tokens=384,
                query_offset=17,
            )
        for _ in range(3):
            for x in values:
                x.copy_(torch.randn_like(x))
            graph.replay()
            expected, ep = prepare_vsa(
                *values,
                m,
                s,
                128,
                padded_to_query=qm,
                query_tokens=384,
                query_offset=17,
            )
            for key in captured:
                assert torch.equal(
                    captured[key].view(torch.uint8), expected[key].view(torch.uint8)
                ), key
            for x, y in zip(pools, ep):
                assert torch.equal(x, y)
        # Positive out-of-range indices must remain masked before vector loads.
        before = [x.clone() for x in (*captured.values(), *pools)]
        m[m < 0] = values[0].shape[1]
        graph.replay()
        for actual, expected in zip((*captured.values(), *pools), before):
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        print("CUDA Graph dynamic inputs and invalid source bounds PASS", flush=True)


def run(values, m, s, qm, block, state=None, query_tokens=None):
    return prepare_vsa(
        *values,
        m,
        s,
        block,
        padded_to_query=qm,
        query_tokens=3 * block if query_tokens is None else query_tokens,
        query_offset=17,
        scale_state=state,
    )


def assert_same(a, b):
    (pa, poola), (pb, poolb) = a, b
    assert pa.keys() == pb.keys()
    for key in pa:
        assert torch.equal(pa[key].view(torch.uint8), pb[key].view(torch.uint8)), key
    for x, y in zip(poola, poolb):
        assert torch.equal(x, y)


def clone_state(state):
    other = VCScaleState(state.qk_margin, state.v_margin)
    other.signature = state.signature
    for name in ("kmean", "qs", "ks", "vs", "fallbacks", "saturations"):
        setattr(other, name, getattr(state, name).clone())
    return other


def canonical_like(tiles=16, h=8, d=128):
    """Identity token order, tiles alternating 256/131 valid rows, prefix of one tile without queries."""
    sizes = torch.tensor([256 if i % 2 == 0 else 131 for i in range(tiles)], device="cuda")
    slot = torch.arange(256, device="cuda")
    valid = (slot[None] < sizes[:, None]).flatten()
    source_map = torch.where(valid, valid.cumsum(0) - 1, -1)
    query_map = torch.where(valid, source_map - 256 + 17, -1)
    query_map[:256] = -1
    source_n = int(sizes.sum())
    values = [torch.randn((1, source_n, h, d), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    return values, source_map, sizes, query_map, source_n - 256


@torch.no_grad()
def test_vsa_delayed_state():
    torch.manual_seed(20260926)
    # T1: state primed on the same input with margins 1.0 -> warm path equals the cold path bytes and pools.
    geometries = [
        (block, d, dtype, metadata)
        for block in (128, 256)
        for d in (64, 128)
        for dtype, metadata in (
            (torch.bfloat16, (torch.int64,) * 3),
            (torch.float32, (torch.int32, torch.int32, torch.int64)),
        )
    ]
    for block, d, dtype, metadata in geometries:
        values, m, s, qm = make_inputs(block, d, dtype, 2, 3, metadata)
        cold = run(values, m, s, qm, block)
        state = VCScaleState(1.0, 1.0)
        assert_same(run(values, m, s, qm, block, state), cold)  # empty state: cold path + init
        assert_same(run(values, m, s, qm, block, state), cold)  # warm path
        assert state.fallbacks.item() == 0, (block, d, dtype)
    values, m, s, qm, nq = canonical_like()
    cold = prepare_vsa(*values, m, s, 256, padded_to_query=qm, query_tokens=nq, query_offset=17)
    state = VCScaleState(1.0, 1.0)
    for _ in range(2):
        warm = prepare_vsa(
            *values, m, s, 256, padded_to_query=qm, query_tokens=nq, query_offset=17, scale_state=state
        )
        assert_same(warm, cold)
    assert state.fallbacks.item() == 0
    print("T1 primed state == cold path PASS", flush=True)

    # T5: metadata dtypes, prefix-only document, and the default margins on the same input.
    for metadata in itertools.product((torch.int32, torch.int64), repeat=3):
        values, m, s, qm = make_inputs(128, 64, torch.bfloat16, 1, 2, metadata)
        state = VCScaleState(1.0, 1.0)
        cold = run(values, m, s, qm, 128, state)
        assert_same(run(values, m, s, qm, 128, state), cold)
    empty = torch.full_like(qm, -1)
    state = VCScaleState(1.0, 1.0)
    cold = run(values, m, s, empty, 128, state, query_tokens=0)
    warm = run(values, m, s, empty, 128, state, query_tokens=0)
    assert_same(warm, cold)
    assert warm[0]["q"].shape[1] == 0
    values, m, s, qm = make_inputs(256, 128, torch.bfloat16)
    state = VCScaleState()
    cold = run(values, m, s, qm, 256, state)
    warm = run(values, m, s, qm, 256, state)
    assert state.fallbacks.item() == 0 and state.saturations.item() == 0
    for key, scale in (("q", "qs"), ("k", "ks"), ("v", "vs")):
        torch.testing.assert_close(warm[0][scale], cold[0][scale] * (1.5 if key != "v" else 2.0))
        shape = (2, 1, 3, -1)  # (b,h) or (b,h,d) descales over BSHD payloads
        x = warm[0][key].float() * warm[0][scale].reshape(shape)
        y = cold[0][key].float() * cold[0][scale].reshape(shape)
        assert (x - y).norm() / y.norm() < 0.05, key
    for x, y in zip(warm[1], cold[1]):
        assert torch.equal(x, y)
    print("T5 maps/prefix/margins PASS", flush=True)

    # T2 + T7: drift beyond the bound (growth and shrink) -> fallback, bytes equal cold; saturation counted.
    for gain in (4.0, 0.25):
        state = VCScaleState()
        run(values, m, s, qm, 256, state)
        drifted = [x * gain for x in values]
        warm = run(drifted, m, s, qm, 256, state)
        assert_same(warm, run(drifted, m, s, qm, 256))
        assert state.fallbacks.item() == 2 * 3, gain
        assert (state.saturations.item() > 0) == (gain > 1), gain
    # T7 steady state: fresh inputs of one distribution -> no fallbacks, no saturations.
    state = VCScaleState()
    run(values, m, s, qm, 256, state)
    for _ in range(4):
        fresh = [torch.randn_like(x) for x in values]
        run(fresh, m, s, qm, 256, state)
    assert state.fallbacks.item() == 0 and state.saturations.item() == 0
    print("T2 drift fallback == cold, T7 saturation counter PASS", flush=True)

    # Retention: returned descales survive a later call on the same state.
    first = run(values, m, s, qm, 256, state)[0]
    kept = {key: first[key].clone() for key in ("qs", "ks", "vs")}
    run([x * 4 for x in values], m, s, qm, 256, state)
    for key in kept:
        assert torch.equal(first[key], kept[key]) and first[key].data_ptr() != getattr(state, key).data_ptr()
    print("retention PASS", flush=True)

    # T6: two alternating distributions through one state -> every call falls back, bytes equal cold.
    state = VCScaleState()
    other = [x * 8 for x in values]
    run(values, m, s, qm, 256, state)
    for i in range(4):
        inputs = other if i % 2 == 0 else values
        assert_same(run(inputs, m, s, qm, 256, state), run(inputs, m, s, qm, 256))
        assert state.fallbacks.item() == 6 * (i + 1)
    print("T6 alternating distributions PASS", flush=True)

    # T4: signature mismatch (heads, block size, dtype) -> cold path and re-initialized state.
    state = VCScaleState()
    run(values, m, s, qm, 256, state)
    for block, d, dtype, h in ((256, 128, torch.bfloat16, 2), (128, 128, torch.bfloat16, 3), (256, 128, torch.float16, 3)):
        v2, m2, s2, q2 = make_inputs(block, d, dtype, 2, h)
        assert_same(run(v2, m2, s2, q2, block, state), run(v2, m2, s2, q2, block))
        assert state.signature == (v2[0].device, dtype, 2, h, d, block)
    assert state.fallbacks.item() == 0
    # Head-dim change with the same batch/heads (channel buffers change shape): cold path, then back.
    for d in (64, 128, 64):
        v2, m2, s2, q2 = make_inputs(256, d, torch.bfloat16, 2, 3)
        assert_same(run(v2, m2, s2, q2, 256, state), run(v2, m2, s2, q2, 256))
        assert state.signature[4] == d and state.vs.shape[-1] == d
    print("T4 signature mismatch -> cold PASS", flush=True)

    # T3: CUDA graph capture/replay with a state and mutated inputs == eager with a cloned state.
    state = VCScaleState()
    run(values, m, s, qm, 256, state)
    run(values, m, s, qm, 256, state)
    eager_state = clone_state(state)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(values, m, s, qm, 256, state)
    for gain in (1.0, 1.0, 5.0, 1.0):
        for x in values:
            x.copy_(torch.randn_like(x) * gain)
        graph.replay()
        expected = run(values, m, s, qm, 256, eager_state)
        assert_same(captured, expected)
        for name in ("kmean", "qs", "ks", "vs", "fallbacks", "saturations"):
            assert torch.equal(getattr(state, name), getattr(eager_state, name)), name
    assert state.fallbacks.item() > 0
    print("T3 CUDA graph with state PASS", flush=True)


if __name__ == "__main__":
    test_vsa_preparation()
    test_vsa_delayed_state()
