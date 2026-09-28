"""KV ping-pong: Q128 block-sparse BF16 SM100 forward with KV tiles alternating between two slots.

Checked against an FP32 reference (O, LSE, fully masked rows). A Q tile with one KV block must not
read slot-1 residue: bit-exact vs the same tiles in a run without the residue. CUDA Graph replay
must equal eager. The one-stream kernel is still traced for configs outside kv_pingpong (dense Q128).
"""

import math
import random

import pytest
import torch

from flash_attn.cute import flash_fwd_sm100, interface
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from mask_mod_definitions import cute_ima_mask

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("SM100 block-sparse forward", allow_module_level=True)

TILE, D = 128, 128
SCALE = 1.0 / math.sqrt(D)
# A mask-listed KV block b keeps its first SIZES[b] keys (block 3 is fully masked).
SIZES = [17, 67, 127, 0, 128, 45, 90, 1]
NB = len(SIZES)

# Per Q tile: (mask-list blocks, full-list blocks). Tiles run mask list then full list, each
# reversed (last entry first), and tile j goes to slot j % 2.
CASES = {
    "count1_full": [([], [2]), ([], [5])],
    "count1_masked": [([1], []), ([6], [])],
    "count1_all_masked": [([3], []), ([2], [5])],
    "count2": [([], [0, 4]), ([2], [7])],
    "odd_3_5": [([0, 2], [4]), ([1, 5, 6], [3, 7]), ([], [1, 2, 3])],
    "first_all_masked": [([2, 3], [4]), ([0, 1, 3], [])],
    "slot1_all_masked": [([3, 0], []), ([3, 5], [6])],
    "empty_tiles": [([], []), ([1], [4, 6]), ([], []), ([7], [])],
}


class _OneQStage(int):
    """max_seqlen_q that selects the one-stage (128-row Q) path for any real length."""

    def __mul__(self, other):
        return type(self)(int(self) * int(other))

    __rmul__ = __mul__

    def __gt__(self, other):
        return False if int(other) == TILE else int(self) > int(other)


def _random_spec(seed, tiles=8):
    rng = random.Random(seed)
    spec = []
    for _ in range(tiles):
        blocks = rng.sample(range(NB), rng.randint(0, 7))
        cut = rng.randint(0, len(blocks))
        spec.append((blocks[:cut], blocks[cut:]))
    return spec


def _keep(device):
    n = torch.arange(NB * TILE, device=device)
    return (n % TILE) < torch.tensor(SIZES, device=device)[n // TILE]


def _aux(device):
    n = torch.arange(NB * TILE, device=device, dtype=torch.int32)
    return [torch.where(_keep(device), n, n + 1).to(torch.int32)]  # keep n iff n >= bias[n]


def _lists(spec, heads, device="cuda"):
    """Return (mask_cnt, mask_idx, full_cnt, full_idx) for all heads."""
    out = []
    for which in range(2):
        cnt = torch.zeros(1, heads, len(spec), dtype=torch.int32)
        idx = torch.zeros(1, heads, len(spec), NB, dtype=torch.int32)
        for q, lists in enumerate(spec):
            cnt[..., q] = len(lists[which])
            for pos, block in enumerate(lists[which]):
                idx[..., q, pos] = block
        out += [cnt.to(device), idx.to(device)]
    return out


def _inputs(tiles, heads, seed=0):
    torch.manual_seed(seed)
    q, k, v = [
        torch.randn(1, n * TILE, heads, D, device="cuda", dtype=torch.bfloat16)
        for n in (tiles, NB, NB)
    ]
    return q, k, v


def _fwd(q, k, v, lists, masked, aux=None):
    mask_cnt, mask_idx, full_cnt, full_idx = lists
    sparse = BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=(TILE, TILE),
    )
    out, lse = interface._flash_attn_fwd(
        q,
        k,
        v,
        max_seqlen_q=_OneQStage(q.shape[1]),
        softmax_scale=SCALE,
        tile_mn=(TILE, TILE),
        pack_gqa=False,
        mask_mod=cute_ima_mask if masked else None,
        aux_tensors=(aux or _aux(q.device)) if masked else None,
        block_sparse_tensors=sparse,
        return_lse=True,
    )[:2]
    return out, lse


def _reference(q, k, v, spec, masked):
    keep = _keep(q.device)
    allowed = torch.zeros(q.shape[1], k.shape[1], dtype=torch.bool, device=q.device)
    for m, (mask_blocks, full_blocks) in enumerate(spec):
        rows = slice(m * TILE, (m + 1) * TILE)
        for block in mask_blocks + full_blocks:
            cols = slice(block * TILE, (block + 1) * TILE)
            use_mask = masked and block in mask_blocks
            allowed[rows, cols] = keep[cols] if use_mask else True
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * SCALE
    s = s.masked_fill(~allowed, -math.inf)
    lse = torch.logsumexp(s, dim=-1)
    p = torch.softmax(s, dim=-1).nan_to_num(0.0)  # fully masked rows -> zero output
    return torch.einsum("bhqk,bkhd->bqhd", p, v.float()), lse


@pytest.mark.parametrize("masked", [True, False])
@pytest.mark.parametrize("case", [*CASES, "random0", "random1", "random2"])
def test_kv_pingpong_matches_reference(case, masked):
    spec = CASES[case] if case in CASES else _random_spec(int(case[-1]))
    heads = 2
    q, k, v = _inputs(len(spec), heads)
    out, lse = _fwd(q, k, v, _lists(spec, heads), masked)
    out_ref, lse_ref = _reference(q, k, v, spec, masked)
    assert torch.isfinite(out).all() and not torch.isnan(lse).any()
    assert (out.float() - out_ref).abs().max() < 2e-2
    assert torch.equal(torch.isinf(lse), torch.isinf(lse_ref))
    finite = torch.isfinite(lse_ref)
    assert (lse[finite] - lse_ref[finite]).abs().max() < 1e-3


def test_unused_slot_ignores_nonfinite_residue():
    """A persistent CTA first overflows O1 (slot-1 V block at bf16 max), then runs a 1-block
    Q tile: that tile must not read O1, so it is bit-exact vs a run with no overflowing tiles."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    tiles = 4 * sms
    rng = random.Random(7)
    # ([], [0, 1]): block 1 -> slot 0, block 0 -> slot 1 (overflows); ([], [2]): slot 0 only.
    kinds = [rng.random() < 0.5 for _ in range(tiles)]
    spec = [([], [0, 1]) if overflow else ([], [2]) for overflow in kinds]
    q, k, v = _inputs(tiles, 1, seed=1)
    v[:, :TILE] = torch.finfo(torch.bfloat16).max
    out, lse = _fwd(q, k, v, _lists(spec, 1), False)
    clean = [([], [2])] * tiles
    out_clean, lse_clean = _fwd(q, k, v, _lists(clean, 1), False)
    out_ref = _reference(q, k, v, clean, False)[0]
    for m, overflow in enumerate(kinds):
        if overflow:
            continue
        rows = slice(m * TILE, (m + 1) * TILE)
        assert torch.isfinite(out[:, rows]).all() and torch.isfinite(lse[..., rows]).all()
        assert torch.equal(out[:, rows], out_clean[:, rows]), f"Q tile {m}"
        assert torch.equal(lse[..., rows], lse_clean[..., rows]), f"Q tile {m}"
        assert (out[:, rows].float() - out_ref[:, rows]).abs().max() < 2e-2


def test_graph_replay_matches_eager():
    specs = [CASES["odd_3_5"], [([1], [0]), ([], [5]), ([3, 2, 1], [6, 7])], [([], []), ([4], []), ([0], [1, 2, 3, 5])]]
    heads = 2
    q, k, v = _inputs(3, heads)
    lists = _lists(specs[0], heads)
    aux = _aux(q.device)  # built outside capture (host->device copy)
    run = lambda: _fwd(q, k, v, lists, True, aux)  # noqa: E731
    run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_g, lse_g = run()
    for spec in specs[1:] + specs[:1]:
        for dst, src in zip(lists, _lists(spec, heads)):
            dst.copy_(src)
        graph.replay()
        out_e, lse_e = run()
        torch.cuda.synchronize()
        assert torch.equal(out_g, out_e) and torch.equal(lse_g, lse_e)


def test_kernel_selects_kv_pingpong(monkeypatch):
    """Sparse Q128 BF16 traces the ping-pong kernel; dense Q128 traces the one-stream kernel."""
    seen = []
    original_init = flash_fwd_sm100.FlashAttentionForwardSm100.__init__

    def spy(self, *args, **kwargs):
        seen.append(self)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", {})
    q, k, v = _inputs(2, 2)
    lists = _lists(CASES["count2"], 2)
    monkeypatch.setattr(flash_fwd_sm100.FlashAttentionForwardSm100, "__init__", spy)
    _fwd(q, k, v, lists, False)
    interface._flash_attn_fwd(q[:, :TILE], k, v, softmax_scale=SCALE, return_lse=True)
    assert [kernel.kv_pingpong for kernel in seen] == [True, False]
    assert [kernel.q_stage for kernel in seen] == [1, 1]
