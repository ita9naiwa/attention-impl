"""Fused inference-only VSA gather/pooling and unsmoothed VC preparation.

Each call covers one document; source tensors can contain the entire packed
batch. Output pools retain H3's original FP32 reduction contract.
"""

import ctypes
import functools
from pathlib import Path

import torch

from flash_attn.cute import vc_preprocess as native


def _ok(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA compilation failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


@functools.cache
def _library(device):
    from cuda.bindings import nvrtc

    source = Path(__file__).with_suffix(".cu").read_bytes()
    header = Path(native.__file__).with_suffix(".cu").read_bytes()
    program = _ok(
        nvrtc.nvrtcCreateProgram(
            source, b"vc_vsa_preprocess.cu", 1, [header], [b"vc_preprocess.cu"]
        )
    )
    architecture = "".join(map(str, torch.cuda.get_device_capability(device)))
    options = [b"--std=c++17", f"--gpu-architecture=compute_{architecture}".encode()]
    try:
        _ok(nvrtc.nvrtcCompileProgram(program, len(options), options))
        ptx = bytearray(_ok(nvrtc.nvrtcGetPTXSize(program)))
        _ok(nvrtc.nvrtcGetPTX(program, ptx))
    except RuntimeError as exc:
        log = bytearray(_ok(nvrtc.nvrtcGetProgramLogSize(program)))
        _ok(nvrtc.nvrtcGetProgramLog(program, log))
        raise RuntimeError(log.decode(errors="replace")) from exc
    finally:
        _ok(nvrtc.nvrtcDestroyProgram(program))
    return native._load_module(
        ctypes.create_string_buffer(bytes(ptx)),
        (
            "vsa_stats",
            "vsa_reduce",
            "vsa_quantize",
            "vsa_routes",
            "vsa_routes_sorted",
            "vsa_routes_warp",
        ),
    )


@functools.cache
def _sm_count(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _launch(name, grid, threads, pointers, integers, stream):
    """Kernel arguments are ordered pointers, integers."""
    driver, _, functions = _library(torch.cuda.current_device())
    values = [ctypes.c_void_p(x.data_ptr()) for x in pointers] + [ctypes.c_int(x) for x in integers]
    args = (ctypes.c_void_p * len(values))(
        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in values]
    )
    native._check(
        driver.cuLaunchKernel(functions[name], *grid, threads, 1, 1, 0, stream, args, None), name
    )


def prepare_vsa(
    q,
    k,
    v,
    padded_to_original,
    variable_block_sizes,
    block_size,
    *,
    padded_to_query,
    query_tokens,
    query_offset=0,
):
    """Return (FP8 prepared dict, FP32 pooled QKV) for one document.

    Inputs are contiguous BSHD. Both maps cover this document's padded KV slots;
    original indices address the full source tensors. Query map values use the
    caller's compact coordinate system and are rebased by query_offset. Mapping
    values/valid sizes are owned by H3 metadata; no host synchronization is added.
    """
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise ValueError("Fused VSA VC preparation is inference-only")
    if q.ndim != 4 or min(q.shape) <= 0 or q.shape[-1] not in (64, 128):
        raise ValueError("expected nonempty BSHD input with D=64/128")
    if not q.is_cuda or q.dtype != torch.bfloat16:
        raise ValueError("expected CUDA BF16 input")
    if any(
        x.shape != q.shape or x.dtype != q.dtype or x.device != q.device or not x.is_contiguous()
        for x in (q, k, v)
    ):
        raise ValueError("q/k/v must share shape/dtype/device and be contiguous")
    if any(
        not isinstance(x, int) or isinstance(x, bool)
        for x in (block_size, query_tokens, query_offset)
    ):
        raise ValueError("block_size/query_tokens/query_offset must be integer scalars")
    if (
        block_size not in (128, 256)
        or not 0 <= query_tokens <= 2147483647
        or not 0 <= query_offset <= 2147483647
    ):
        raise ValueError("block_size must be 128/256 and query sizes nonnegative")
    if any(
        x.ndim != 1
        or x.device != q.device
        or x.dtype not in (torch.int32, torch.int64)
        or not x.is_contiguous()
        for x in (padded_to_original, padded_to_query, variable_block_sizes)
    ):
        raise ValueError("maps and block sizes must be contiguous CUDA int32/int64 vectors")
    blocks = variable_block_sizes.numel()
    padded_n = blocks * block_size
    if (
        not blocks
        or padded_to_original.numel() != padded_n
        or padded_to_query.numel() != padded_n
        or query_tokens > padded_n
    ):
        raise ValueError("maps must match padded KV size and query length must not exceed it")
    b, source_n, h, d = q.shape
    if b > 65535 or h > 65535:
        raise ValueError("fused VSA preparation supports batch/head counts up to 65535")
    with torch.cuda.device(q.device), torch.no_grad():
        factory = {"device": q.device, "dtype": torch.float32}
        pools = tuple(torch.empty((b, h, blocks, d), **factory) for _ in range(3))
        stats = torch.empty((b, h, blocks, 6, d), **factory)
        kmean = torch.empty((b, h, d), **factory)
        qs, ks = (torch.empty((b, h), **factory) for _ in range(2))
        vs = torch.empty((b, h, d), **factory)
        oq = torch.empty((b, query_tokens, h, d), device=q.device, dtype=torch.float8_e4m3fn)
        ok, ov = (
            torch.empty((b, padded_n, h, d), device=q.device, dtype=torch.float8_e4m3fn)
            for _ in range(2)
        )
        stream = torch.cuda.current_stream(q.device).cuda_stream
        metadata_mask = sum(
            (x.dtype == torch.int64) << i
            for i, x in enumerate((padded_to_original, padded_to_query, variable_block_sizes))
        )
        _launch(
            "vsa_stats",
            (blocks, h, b),
            256,
            [q, k, v, padded_to_original, padded_to_query, variable_block_sizes, stats, *pools],
            [source_n, blocks, h, d, block_size, query_tokens, query_offset, metadata_mask],
            stream,
        )
        _launch(
            "vsa_reduce",
            (b * h, 1, 1),
            1024,
            [stats, kmean, qs, ks, vs],
            [padded_n, d, blocks],
            stream,
        )
        _launch(
            "vsa_quantize",
            (min((padded_n + 3) // 4, 4 * _sm_count(q.device)), h, b),
            128,
            [q, k, v, padded_to_original, padded_to_query, kmean, qs, ks, vs, oq, ok, ov],
            [source_n, padded_n, query_tokens, query_offset, h, d, metadata_mask],
            stream,
        )
    return {"q": oq, "k": ok, "v": ov, "qs": qs, "ks": ks, "vs": vs}, pools


def prepare_vsa_routes(selected, sizes, block_size, prefix, document_start=0):
    """Return full indices/counts and partial indices/counts from top-k IDs.

    ``selected`` is int64 [B,H,Q,K], containing unique global video-parent IDs
    in [document_start + prefix, document_start + sizes.numel()). Prefix parents
    are implicit. ``sizes`` is int32 with values in [0, block_size]. These content
    invariants belong to the caller; validating them here would synchronize or
    add sorting. Invalid IDs are skipped safely, but malformed/duplicate inputs
    have no defined attention semantics. Output lists are ascending int32, with
    capacity (prefix+K)*(block_size//128). Partial256 parents retain both children.
    """
    if (
        selected.ndim != 4
        or selected.dtype != torch.int64
        or not selected.is_cuda
        or sizes.ndim != 1
        or sizes.dtype != torch.int32
        or sizes.device != selected.device
    ):
        raise ValueError(
            "selected must be CUDA int64 B/H/Q/K; sizes must be same-device int32 vector"
        )
    if any(type(value) is not int for value in (block_size, prefix, document_start)):
        raise ValueError("block_size, prefix and document_start must be integers")
    if block_size not in (128, 256):
        raise ValueError("block_size must be 128 or 256")
    b, h, q, topk = selected.shape
    parents = sizes.numel()
    rows = b * h * q
    capacity = (prefix + topk) * (block_size // 128)
    if (
        min(b, h, q, parents) < 1
        or not 0 <= prefix <= parents
        or topk > parents - prefix
        or document_start < 0
        or max(rows, capacity, parents * (block_size // 128), document_start + parents) > 2**31 - 1
        or capacity < 1
    ):
        raise ValueError("invalid or empty route dimensions, prefix, or document range")
    selected = selected.contiguous()
    sizes = sizes.contiguous()
    full_idx = torch.empty((b, h, q, capacity), dtype=torch.int32, device=selected.device)
    mask_idx = torch.empty_like(full_idx)
    full_cnt = torch.empty((b, h, q), dtype=torch.int32, device=selected.device)
    mask_cnt = torch.empty_like(full_cnt)
    short_route = prefix + topk <= 32 and parents <= 1073741823
    with torch.cuda.device(selected.device):
        _launch(
            "vsa_routes_warp" if short_route else ("vsa_routes_sorted" if prefix + topk <= 1024 and parents <= 1073741823 else "vsa_routes"),
            ((rows + 3) // 4 if short_route else rows, 1, 1),
            128,
            (selected, sizes, full_idx, full_cnt, mask_idx, mask_cnt),
            (rows, topk, prefix, document_start, block_size, capacity, parents),
            torch.cuda.current_stream().cuda_stream,
        )
    return full_idx, full_cnt, mask_idx, mask_cnt
