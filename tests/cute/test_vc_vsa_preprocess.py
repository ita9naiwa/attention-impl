import contextlib
import itertools
import json

import torch
from flash_attn.cute import vc_vsa_preprocess as vsa
from flash_attn.cute.vc_preprocess import prepare
from flash_attn.cute.vc_vsa_preprocess import prepare_vsa


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
                case(block, d, torch.bfloat16)
        for dtype in (torch.float16, torch.float32):  # BF16-only producer
            values, m, s_, qm = make_inputs(128, 128, dtype)
            try:
                run(values, m, s_, qm, 128)
            except ValueError:
                pass
            else:
                raise AssertionError(("non-BF16 input accepted", dtype))
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


def run(values, m, s, qm, block, query_tokens=None):
    return prepare_vsa(
        *values,
        m,
        s,
        block,
        padded_to_query=qm,
        query_tokens=3 * block if query_tokens is None else query_tokens,
        query_offset=17,
    )


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


def raw(x):
    return x.detach().contiguous().reshape(-1).view(torch.uint8)


def snapshot(result):
    p, pools = result
    out = [(key, p[key]) for key in sorted(p)] + [(f"pool{i}", x) for i, x in enumerate(pools)]
    return [(n, x.clone()) for n, x in out]


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


@contextlib.contextmanager
def sm_count(n):
    saved = vsa._sm_count
    vsa._sm_count = lambda device: n
    try:
        yield
    finally:
        vsa._sm_count = saved


@torch.no_grad()
def test_vsa_producer_prefetch_raw_bytes():
    """Aligned BF16 D128 (prefetching stats rows, grid-stride quantize with register scales) vs the same values with
    one misaligned tensor, which takes the per-row read_qkv_contiguous / per-token vsa_quantize_token code."""
    torch.manual_seed(20260927)
    torch.zeros(1, device="cuda")

    def pair(values, m, s, qm, block, query_tokens=None):
        assert all(x.data_ptr() % 8 == 0 for x in values)
        slow = [placed(x, o) for x, o in zip(values, (0, 0, 1))]
        return snapshot(run(values, m, s, qm, block, query_tokens)), snapshot(run(slow, m, s, qm, block, query_tokens))

    metadata = list(itertools.product((torch.int32, torch.int64), repeat=3))
    # SM counts 1/5/7 make the quantize grid 4/20/28 CTAs: many grid-stride steps, and with 5/7 a partial last
    # stride (padded_n % (16 * sms) != 0); None keeps the device count.
    for sms, block, kind in itertools.product((None, 1, 5, 7), (128, 256), ("normal", "finite", "nonfinite")):
        with sm_count(sms) if sms else contextlib.nullcontext():
            for mix in metadata[:: 3 if sms else 1]:
                values, m, s, qm = make_inputs(block, 128, torch.bfloat16, 2, 3, mix)
                values = [special(x, kind) for x in values]
                assert_raw(*pair(values, m, s, qm, block), (sms, block, kind, mix))
            assert_raw(*pair(values, m, s, torch.full_like(qm, -1), block, 0), (sms, block, kind, "no queries"))
    values, m, s, qm, nq = canonical_like(tiles=41)  # padded_n 10496: several strides at the device count
    for sms in (None, 3):
        with sm_count(sms) if sms else contextlib.nullcontext():
            assert_raw(*pair(values, m, s, qm, 256, nq), ("canonical", sms))
    print("producer prefetch raw-byte parity PASS", flush=True)


if __name__ == "__main__":
    test_vsa_preparation()
    test_vsa_producer_prefetch_raw_bytes()
