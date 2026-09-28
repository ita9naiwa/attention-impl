"""CPU: the forward compile key and kernel attributes see alias_guard only as (hint is True and block-sparse).

Runs _flash_attn_fwd under FakeTensorMode with a stubbed cute.compile and a recording compile cache. Each launch
yields (compile_key, the kernel object's scalar attributes -- vc_*, alias_guard_hint, use_ldred_rowmax,
q_stage, ... -- and the compile argument types). No GPU needed.
"""

import itertools

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

ALIAS_POS = 0  # compile_key position of the alias_guard element


def capture(arch, dtype, sparse_q, vc_expcast, head_dim, alias_guard, seqlen=1024, tile_mn=None,
            vc_vbs128=False, vc_vscale=False, vc_mean=False):
    """("rec", compile_key, attrs, arg_types) for one forward launch, or ("err", exc_type, first_line)."""
    from flash_attn.cute import interface
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    recs = []

    class Cache(dict):
        def __setitem__(self, key, val):
            recs[-1] = (key,) + recs[-1]
            super().__setitem__(key, val)

    def stub(fa_fwd, *args, **kwargs):
        attrs = tuple(sorted((a, v) for a, v in vars(fa_fwd).items()
                             if v is None or isinstance(v, (bool, int, float, str))))
        recs.append((attrs, tuple(type(a).__name__ for a in args)))
        return lambda *a, **k: None

    old = interface.cute.compile, interface._flash_attn_fwd.compile_cache
    interface.cute.compile, interface._flash_attn_fwd.compile_cache = stub, Cache()
    try:
        with FakeTensorMode():
            dev, b, h, n = "cuda", 1, 2, seqlen
            q, k, v = [torch.empty(b, n, h, head_dim, device=dev, dtype=dtype) for _ in range(3)]
            kw = {}
            if sparse_q is not None:
                qn, kn = n // sparse_q, n // 128
                cnt = torch.empty(b, h, qn, device=dev, dtype=torch.int32)
                idx = torch.empty(b, h, qn, kn, device=dev, dtype=torch.int32)
                kw["block_sparse_tensors"] = BlockSparseTensorsTorch(
                    full_block_cnt=cnt, full_block_idx=idx, mask_block_cnt=torch.empty_like(cnt),
                    mask_block_idx=torch.empty_like(idx), block_size=(sparse_q, 128))
            if vc_vbs128:
                kw["aux_tensors"] = [torch.empty(n // 128, device=dev, dtype=torch.int32)]
            if vc_vscale:
                kw["vc_vscale"] = torch.empty(b, h, head_dim, device=dev, dtype=torch.float32)
            if vc_mean:
                kw["vc_mean"] = torch.empty(b, h, n // 128, head_dim, device=dev, dtype=torch.bfloat16)
            interface._flash_attn_fwd(q, k, v, _arch=arch, tile_mn=tile_mn, vc_expcast=vc_expcast,
                                      vc_vbs128=vc_vbs128, alias_guard=alias_guard, **kw)
    except Exception as e:  # invalid combos must fail the same way before and after
        return ("err", type(e).__name__, (str(e).splitlines() or [""])[0])
    finally:
        interface.cute.compile, interface._flash_attn_fwd.compile_cache = old
    assert len(recs) == 1 and len(recs[0]) == 3, recs
    return ("rec",) + recs[0]


def strip_alias(rec):
    """The record with the alias element (key position 0 and the alias_guard_hint attribute) removed."""
    if rec[0] != "rec":
        return rec
    key, attrs, types = rec[1:]
    return (key[:ALIAS_POS] + key[ALIAS_POS + 1:], tuple(kv for kv in attrs if kv[0] != "alias_guard_hint"), types)


CASES = [
    dict(arch=arch, dtype=dtype, sparse_q=sq, vc_expcast=vc, head_dim=hd)
    for arch, dtype, sq, vc, hd in itertools.product(
        (100, 103), (torch.bfloat16, torch.float8_e4m3fn), (128, 256, None), (False, True), (64, 128))
    if not (vc and dtype != torch.float8_e4m3fn)  # vc_expcast is FP8 e4m3 only
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(str(v).replace("torch.", "") for v in c.values()))
def test_alias_guard_compile_key(case):
    none, false, true = (capture(**case, alias_guard=a) for a in (None, False, True))
    assert none[0] == "rec", none
    assert none == false  # one key for None and False
    if case["sparse_q"] is None:
        assert true == none  # the guard cannot engage on a dense launch
    else:
        assert true != none and strip_alias(true) == strip_alias(none)
        assert true[1][ALIAS_POS] is True and dict(true[2])["alias_guard_hint"] is True
        assert none[1][ALIAS_POS] is False and dict(none[2])["alias_guard_hint"] is False
