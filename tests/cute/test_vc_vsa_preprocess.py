import contextlib
import itertools
import json
import tempfile
from pathlib import Path

import torch
from flash_attn.cute import vc_vsa_preprocess as vsa
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
    # Batch change with the same heads/head-dim (per-head buffers change shape): cold path, then back.
    for bsz in (1, 2):
        v2, m2, s2, q2 = make_inputs(256, 128, torch.bfloat16, bsz, 3)
        assert_same(run(v2, m2, s2, q2, 256, state), run(v2, m2, s2, q2, 256))
        assert state.signature[2] == bsz and state.qs.shape[0] == bsz
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


def ungrouped_library():
    """vc_vsa_preprocess.cu with the grouped D128 stats loads undone: the three original read_contiguous calls."""
    src = Path(vsa.__file__).with_suffix(".cu").read_text()
    call = "read_qkv_contiguous<D>(q,k,v,base+lane*W,dtype,rawq,rawk,rawv);"
    assert src.count(call) == 1
    loads = "\n                ".join(
        f"read_contiguous<D>({x},base+lane*W,dtype,raw{x});" for x in "qkv"
    )
    helper = src[src.index("// Stats D128:") : src.index("__device__ int64_t vsa_index")]
    src = src.replace(call, loads).replace(helper, "")
    saved = vsa.__file__
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "vc_vsa_preprocess.cu").write_text(src)
        vsa.__file__ = str(Path(tmp) / "vc_vsa_preprocess.py")
        try:
            return vsa._library.__wrapped__(torch.cuda.current_device())
        finally:
            vsa.__file__ = saved


@contextlib.contextmanager
def using(library):
    saved = vsa._library
    vsa._library = lambda device: library
    try:
        yield
    finally:
        vsa._library = saved


def raw(x):
    return x.detach().contiguous().reshape(-1).view(torch.uint8)


def snapshot(result, state):
    p, pools = result
    out = [(key, p[key]) for key in sorted(p)] + [(f"pool{i}", x) for i, x in enumerate(pools)]
    names = ("kmean", "qs", "ks", "vs", "fallbacks", "saturations")
    return [(n, x.clone()) for n, x in out + [(f"state.{n}", getattr(state, n)) for n in names]]


def assert_raw(a, b, what):
    for (name, x), (other, y) in zip(a, b, strict=True):
        assert name == other and x.dtype == y.dtype and x.shape == y.shape, (what, name)
        assert torch.equal(raw(x), raw(y)), (what, name)


def placed(x, offset):
    """Contiguous copy of x at a storage offset of `offset` elements (breaks 8B alignment for BF16 1..3)."""
    y = torch.zeros(x.numel() + offset, dtype=x.dtype, device=x.device)[offset:].view(x.shape)
    return y.copy_(x)


def special(x, kind):
    if kind == "normal":
        return x
    info = torch.finfo(x.dtype)
    if kind == "finite":
        pool = [
            0.0,
            -0.0,
            info.smallest_normal / 4,
            -info.smallest_normal / 2,
            info.tiny,
            info.max / 2,
            -info.max / 4,
        ]
    else:
        pool = [float("inf"), float("-inf"), float("nan"), 1.0, -2.0]
    x = x.clone()
    flat = x.view(-1)
    index = torch.randperm(flat.numel(), device=x.device)[: flat.numel() // 8]
    choice = torch.tensor(pool, device=x.device, dtype=x.dtype)
    flat[index] = choice[torch.randint(len(pool), (index.numel(),), device=x.device)]
    if kind == "finite":
        x[0, :, 0] = -0.0  # a whole (b, h) of signed zeros
        x[-1, :, -1] = 0.0
    return x


@torch.no_grad()
def test_vsa_stats_grouped_loads_raw_bytes():
    """Raw-byte parity of the grouped D128 stats loads against the original per-input loads."""
    torch.manual_seed(20260927)
    torch.zeros(1, device="cuda")  # the driver-API module load needs the primary context
    base = ungrouped_library()
    new = vsa._library(torch.cuda.current_device())

    def both(values, m, s, qm, block, query_tokens=None, states=(None, None)):
        out = []
        for library, state in zip((base, new), states):
            state = VCScaleState() if state is None else state
            with using(library):
                out.append(snapshot(run(values, m, s, qm, block, state, query_tokens), state))
        return out

    offsets = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
        (4, 4, 4),
        (0, 1, 0),
        (0, 0, 2),
        (3, 0, 0),
    ]
    metadata = list(itertools.product((torch.int32, torch.int64), repeat=3))
    cases = 0
    for n, (block, d, dtype, kind) in enumerate(
        itertools.product(
            (128, 256),
            (64, 128),
            (torch.bfloat16, torch.float16, torch.float32),
            ("normal", "finite", "nonfinite"),
        )
    ):
        values, m, s, qm = make_inputs(block, d, dtype, 2, 3, metadata[n % 8])
        values = [special(x, kind) for x in values]
        for offset in offsets if dtype == torch.bfloat16 else offsets[:2] + offsets[5:6]:
            placed_values = [placed(x, o) for x, o in zip(values, offset)]
            if dtype == torch.bfloat16:
                assert [x.data_ptr() % 8 != 0 for x in placed_values] == [
                    o % 4 != 0 for o in offset
                ]
            a, b = both(placed_values, m, s, qm, block)
            assert_raw(a, b, (block, d, dtype, kind, offset))
            cases += 1
        a, b = both(
            values, m, s, torch.full_like(qm, -1), block, query_tokens=0
        )  # every query invalid
        assert_raw(a, b, (block, d, dtype, kind, "no queries"))
    values, m, s, qm, nq = canonical_like()
    for offset in offsets[:3]:
        v = [placed(x, o) for x, o in zip(values, offset)]
        assert_raw(*both(v, m, s, qm, 256, query_tokens=nq), ("canonical", offset))
    print(f"cold raw-byte parity PASS ({cases} cases)", flush=True)

    # Aligned D128 BF16 (the grouped branch) over all 8 int32/int64 metadata mixes, cold and delayed.
    for block, mix in itertools.product((128, 256), metadata):
        values, m, s, qm = make_inputs(block, 128, torch.bfloat16, 2, 3, mix)
        assert all(x.data_ptr() % 8 == 0 for x in values)
        assert_raw(*both(values, m, s, qm, block), ("aligned D128 metadata", block, mix))
        states = (VCScaleState(), VCScaleState())
        for step, gain in enumerate((1.0, 1.0, 4.0)):  # init (cold), warm, drift fallback
            inputs = values if step < 2 else [x * gain for x in values]
            a, b = both(inputs, m, s, qm, block, states=states)
            assert_raw(a, b, ("aligned D128 metadata delayed", block, mix, step))
    print("aligned D128 BF16 all metadata mixes cold + delayed raw-byte parity PASS", flush=True)

    # Changed inputs, eager and CUDA graph replay, cold and delayed; aligned and unaligned BF16 D128.
    for offset in ((0, 0, 0), (1, 0, 3)):
        values, m, s, qm = make_inputs(256, 128, torch.bfloat16)
        values = [placed(x, o) for x, o in zip(values, offset)]
        for step in range(3):
            for x in values:
                x.copy_(torch.randn_like(x) * (1 + step))
            assert_raw(*both(values, m, s, qm, 256), ("eager changed input", offset, step))
        states = (VCScaleState(), VCScaleState())
        graphs, captured = [], []
        for library, state in zip((base, new), states):
            with using(library):
                run(values, m, s, qm, 256, state)
                run(values, m, s, qm, 256, state)
                torch.cuda.synchronize()
                cold_graph, warm_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(cold_graph):
                    cold = run(values, m, s, qm, 256)
                with torch.cuda.graph(warm_graph):
                    warm = run(values, m, s, qm, 256, state)
                graphs.append((cold_graph, warm_graph))
                captured.append((cold, warm, state))
        for gain in (1.0, 1.0, 5.0, 1.0):
            for x in values:
                x.copy_(torch.randn_like(x) * gain)
            snaps = []
            for (cold_graph, warm_graph), (cold, warm, state) in zip(graphs, captured):
                cold_graph.replay()
                warm_graph.replay()
                snaps.append(snapshot(cold, state)[:-6] + snapshot(warm, state))
            assert_raw(*snaps, ("graph replay", offset, gain))
    print("changed-input eager and graph replay raw-byte parity PASS", flush=True)

    # Delayed histories from identically initialized states: warm same distribution, drift, shrink,
    # alternating, exceptional values, back to normal.
    configs = [
        (256, 128, torch.bfloat16, (0, 0, 0), (1.5, 2.0)),
        (256, 128, torch.bfloat16, (1, 1, 1), (1.5, 2.0)),
        (128, 128, torch.bfloat16, (0, 2, 0), (1.0, 1.0)),
        (128, 64, torch.bfloat16, (0, 0, 0), (1.5, 2.0)),
        (256, 128, torch.float16, (0, 0, 0), (1.5, 2.0)),
        (256, 128, torch.float32, (0, 1, 0), (1.5, 2.0)),
    ]
    for block, d, dtype, offset, margins in configs:
        values, m, s, qm = make_inputs(block, d, dtype, 2, 3, metadata[block // 128 + d // 64])
        history = [values] + [[torch.randn_like(x) for x in values] for _ in range(3)]
        history += [
            [x * 4 for x in values],
            [x * 0.25 for x in values],
            [x * 8 for x in values],
            values,
        ]
        history += [
            [special(x, "finite") for x in values],
            [special(x, "nonfinite") for x in values],
            values,
        ]
        inputs = [placed(x, o) for x, o in zip(values, offset)]
        states = (VCScaleState(*margins), VCScaleState(*margins))
        fallbacks = []
        for step, h in enumerate(history):
            for x, y in zip(inputs, h):
                x.copy_(y)
            a, b = both(inputs, m, s, qm, block, states=states)
            assert_raw(a, b, ("delayed", block, d, dtype, offset, step))
            fallbacks.append(int(dict(b)["state.fallbacks"]))
        increments = [y - x for x, y in zip(fallbacks[:-1], fallbacks[1:])]
        assert max(increments) > 0 and (margins == (1.0, 1.0) or 0 in increments[:3]), (
            block,
            d,
            fallbacks,
        )
    print("delayed warm/drift/fallback history raw-byte parity PASS", flush=True)


if __name__ == "__main__":
    test_vsa_preparation()
    test_vsa_delayed_state()
    test_vsa_stats_grouped_loads_raw_bytes()
