import itertools
import json

import torch
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


def case(block, d, dtype, b=2, h=3, metadata_dtypes=(torch.int64,) * 3):
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


if __name__ == "__main__":
    test_vsa_preparation()
