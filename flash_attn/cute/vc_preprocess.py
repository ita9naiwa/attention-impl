"""Native CUDA gather, centering, V block smoothing, and E4M3 quantization.

Input BHND (or BSHD with bshd=True), output BNHD. Q/K use per-head scales; V uses per-channel scales.
Means default to BF16 in quantized-V units (mean / vs), following Figure 2
and Appendix B; mean_dtype=torch.float32 retains higher-precision metadata.
Q/K orthonormal Hadamard rotation is enabled by default; RoPE fusion is omitted.
"""

import ctypes
import functools
import sys
from pathlib import Path

import torch


def _check(code, operation):
    if code:
        raise RuntimeError(f"{operation} failed: CUDA error {code}")


@functools.cache
def _library(device_index):
    """Compile native CUDA with the existing runtime compiler; no toolkit install."""
    candidates = [Path(root) / "nvidia/cu13/lib/libnvrtc.so.13" for root in sys.path]
    path = next((p for p in candidates if p.exists()), None)
    nvrtc = ctypes.CDLL(str(path) if path else "libnvrtc.so")
    nvrtc.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    nvrtc.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    program = ctypes.c_void_p()
    source = Path(__file__).with_suffix(".cu").read_bytes()
    _check(
        nvrtc.nvrtcCreateProgram(ctypes.byref(program), source, b"vc_preprocess.cu", 0, None, None),
        "nvrtcCreateProgram",
    )
    architecture = "".join(map(str, torch.cuda.get_device_capability(device_index)))
    options = (ctypes.c_char_p * 2)(
        b"--std=c++17", ("--gpu-architecture=compute_" + architecture).encode()
    )
    result = nvrtc.nvrtcCompileProgram(program, 2, options)
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    if result:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
        raise RuntimeError("CUDA preprocessing compilation failed: " + log.value.decode())
    _check(nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)), "nvrtcGetPTXSize")
    ptx = ctypes.create_string_buffer(size.value)
    _check(nvrtc.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
    nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    driver, module, functions = _load_module(
        ptx,
        (
            "rotate_qk",
            "block_stats",
            "reduce_stats",
            "quantize",
            "fused_stats",
            "fused_quantize",
            "fused_quantize_stats",
            "delayed_check",
            "fallback_quantize",
        ),
    )
    driver.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    for name in ("fused_stats", "fused_quantize_stats"):
        _check(driver.cuFuncSetAttribute(functions[name], 8, 36864), "cuFuncSetAttribute")
    return driver, module, functions


def _load_module(ptx, names):
    """Load PTX and retain its driver/module alongside the named entrypoints."""
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    driver.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    driver.cuLaunchKernel.argtypes = (
        [ctypes.c_void_p]
        + [ctypes.c_uint] * 7
        + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    )
    module = ctypes.c_void_p()
    _check(driver.cuModuleLoadData(ctypes.byref(module), ptx), "cuModuleLoadData")
    functions = {}
    for name in names:
        function = ctypes.c_void_p()
        _check(
            driver.cuModuleGetFunction(ctypes.byref(function), module, name.encode()),
            "cuModuleGetFunction",
        )
        functions[name] = function
    return driver, module, functions


def _launch(name, grid, threads, pointers, integers, stream, wide_last=False):
    driver, _, functions = _library(torch.cuda.current_device())
    values = [ctypes.c_void_p(x.data_ptr() if x is not None else 0) for x in pointers]
    values += [ctypes.c_int(x) for x in (integers[:-1] if wide_last else integers)]
    if wide_last:
        values.append(ctypes.c_longlong(integers[-1]))
    args = (ctypes.c_void_p * len(values))(
        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in values]
    )
    _check(
        driver.cuLaunchKernel(
            functions[name],
            *(grid if isinstance(grid, tuple) else (grid, 1, 1)),
            threads,
            1,
            1,
            ((32 if integers[1] == 128 else 128) + 16) * integers[1] * 4
            if name == "fused_stats"
            else ((32 if integers[1] == 128 else 128) + 16) * integers[1] * 4
            if name == "fused_quantize_stats"
            else 0,
            stream,
            args,
            None,
        ),
        name,
    )


@torch.no_grad()
def _rotate_qk(q, k, bshd=False):
    """Orthogonal D=64/128 transform; always return contiguous BHND tensors."""
    oq, ok = [
        torch.empty_like(
            q.transpose(1, 2) if bshd else q,
            dtype=torch.float32,
            memory_format=torch.contiguous_format,
        )
        for _ in range(2)
    ]
    dtype = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[q.dtype]
    tokens = q.numel() // q.shape[-1]
    with torch.cuda.device(q.device):
        _launch(
            "rotate_qk",
            (tokens + 3) // 4,
            128,
            [q, k, oq, ok],
            [
                q.shape[-1],
                dtype,
                q.shape[1] if bshd else q.shape[2],
                q.shape[2] if bshd else q.shape[1],
                int(bshd),
                tokens,
            ],
            torch.cuda.current_stream(q.device).cuda_stream,
            wide_last=True,
        )
    return oq, ok


# prepare() compiles its margins into vc_preprocess.cu (QK_MARGIN / V_MARGIN): powers of two, so warm codes stay on
# the cold E4M3 grid. prepare_vsa() passes the state's margins to its kernels at run time (default 1.5 / 2.0).
PREPARE_MARGINS = (2.0, 2.0)
PREPARE_VSA_MARGINS = (1.5, 2.0)


class VCScaleState:
    """Caller-owned delayed-scaling state for prepare() and prepare_vsa(): one per attention layer instance, entry
    point and stream (head/CFG).

    Opt-in: used only when passed as scale_state. Holds the previous call's {kmean, qs, ks, vs} (scales and
    statistics only, never payload) in device buffers updated in place, plus int32 device counters. With a matching
    state, the entry point quantizes in one pass with the previous scales times margins (qk_margin for Q/K,
    v_margin for V) while recomputing this call's statistics, then re-quantizes on the device, with fresh scales,
    every (b,h) whose fresh range the used scales do not cover or that shrank by more than 2x (no host
    synchronization). An empty or mismatched state (the signature includes the entry point's shapes) takes the
    unchanged cold path and (re)initializes. Returned descales are the ones actually used. Sharing one state across
    differently distributed inputs only causes fallbacks.

    Margins default (None) to the entry point's own: PREPARE_MARGINS for prepare(), which accepts only those
    (compiled), and PREPARE_VSA_MARGINS for prepare_vsa(), which accepts any.
    Unset margins stay None on the object (qk_margin/v_margin); margins(default) returns the resolved pair.
    """

    def __init__(self, qk_margin=None, v_margin=None):
        self.signature = None
        self.qk_margin = None if qk_margin is None else float(qk_margin)
        self.v_margin = None if v_margin is None else float(v_margin)
        self.fallbacks = None  # int32 device counter: (b,h) re-quantized with fresh scales
        # int32 device counter: (block, channel, tensor) slabs over E4M3 range in attempted warm-path stores
        # (prepare_vsa: including padded K slots and heads later overwritten by a fallback).
        self.saturations = None

    def margins(self, default):
        """(qk_margin, v_margin), each unset one taken from the entry point's default pair."""
        return tuple(
            d if m is None else m for m, d in zip((self.qk_margin, self.v_margin), default)
        )

    def _init(self, signature, kmean, qs, ks, vs):
        fresh = (kmean, qs, ks, vs)
        old = (self.kmean, self.qs, self.ks, self.vs) if self.signature is not None else ()
        # Reuse in place (graph-safe pointers) only if every buffer matches: kmean/vs depend on D, not only on
        # (B, H), and a dtype change must not be silently cast by copy_.
        if old and all(
            (dst.shape, dst.device, dst.dtype) == (src.shape, src.device, src.dtype)
            for dst, src in zip(old, fresh)
        ):
            for dst, src in zip(old, fresh):
                dst.copy_(src)
        else:
            self.kmean, self.qs, self.ks, self.vs = (
                kmean.clone(),
                qs.clone(),
                ks.clone(),
                vs.clone(),
            )
        self.signature = signature
        if self.fallbacks is None or self.fallbacks.device != qs.device:
            self.fallbacks = torch.zeros((), device=qs.device, dtype=torch.int32)
            self.saturations = torch.zeros((), device=qs.device, dtype=torch.int32)


@torch.no_grad()
def prepare(
    q,
    k,
    v,
    permutation=None,
    smooth=True,
    hadamard=True,
    mean_dtype=torch.bfloat16,
    bshd=False,
    scale_state=None,
):
    """Prepare finite contiguous CUDA BHND or BSHD inputs; permutation must be a bijection.

    The caller owns validation of permutation values and finite input values;
    shape/dtype/device validation is synchronization-free and always performed.
    """
    if q.ndim != 4 or min(q.shape) < 1 or q.shape[-1] not in (64, 128):
        raise ValueError("expected nonempty 4D input with D=64/128")
    if not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("expected CUDA BF16, FP16, or FP32 inputs")
    if any(
        x.shape != q.shape or x.dtype != q.dtype or x.device != q.device or not x.is_contiguous()
        for x in (q, k, v)
    ):
        raise ValueError("q, k, v must share shape/dtype/device and be contiguous")
    if mean_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("mean_dtype must be torch.bfloat16 or torch.float32")
    b, h, n, d = q.transpose(1, 2).shape if bshd else q.shape
    if permutation is not None and (
        permutation.shape != (b, h, n)
        or permutation.device != q.device
        or permutation.dtype != torch.int64
        or not permutation.is_contiguous()
    ):
        raise ValueError("permutation must be contiguous int64 BHN on the input device")
    fused = hadamard and not smooth and permutation is None and b <= 65535 and h <= 65535
    nb = (n + 127) // 128
    factory = {"device": q.device, "dtype": torch.float32}
    qs, ks = [torch.empty((b, h), **factory) for _ in range(2)]
    vs = torch.empty((b, h, d), **factory)
    means = torch.empty((b, h, nb, d), device=q.device, dtype=mean_dtype)
    stats = torch.empty((b, h, nb, 6, d), **factory)
    kmean = torch.empty((b, h, d), **factory)
    outputs = [
        torch.empty((b, n, h, d), device=q.device, dtype=torch.float8_e4m3fn) for _ in range(3)
    ]
    delayed = scale_state is not None and fused
    if delayed and scale_state.margins(PREPARE_MARGINS) != PREPARE_MARGINS:
        raise ValueError(f"prepare() delayed scaling uses the compiled margins {PREPARE_MARGINS}")
    signature = (q.device, q.dtype, b, h, n, d, bshd)
    if delayed and scale_state.signature == signature:
        return _prepare_delayed(
            q,
            k,
            v,
            scale_state,
            stats,
            means,
            kmean,
            qs,
            ks,
            vs,
            outputs,
            b,
            h,
            n,
            d,
            nb,
            bshd,
            mean_dtype,
        )
    with torch.cuda.device(q.device):
        stream = torch.cuda.current_stream(q.device).cuda_stream
        dtype = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[q.dtype]
        qinput, kinput = _rotate_qk(q, k, bshd=bshd) if hadamard and not fused else (q, k)
        qkdtype = 2 if hadamard and not fused else dtype
        _launch(
            "fused_stats" if fused else "block_stats",
            b * h * nb,
            256 if fused else 128,
            [qinput, kinput, v, permutation, stats],
            [
                n,
                d,
                nb,
                h,
                qkdtype,
                dtype,
                int(smooth),
                int(bshd and (not hadamard or fused)),
                int(bshd),
            ],
            stream,
        )
        _launch(
            "reduce_stats",
            b * h,
            128 if nb == 1 else 1024,
            [stats, means, kmean, qs, ks, vs],
            [n, d, nb, int(mean_dtype == torch.float32)],
            stream,
        )
        quantize_blocks = (n + 7) // 8 if fused else (n * (d // 4) + 255) // 256
        quantize_grid = (quantize_blocks, h, b)
        if not fused and (b > 65535 or h > 65535):
            quantize_grid = (quantize_blocks * b * h, 1, 1)
        _launch(
            "fused_quantize" if fused else "quantize",
            quantize_grid,
            256,
            [qinput, kinput, v, permutation, kmean, qs, ks, vs, stats, *outputs],
            [
                n,
                h,
                d,
                nb,
                b,
                qkdtype,
                dtype,
                int(bshd and (not hadamard or fused)),
                int(bshd),
                b * h * n * d,
            ],
            stream,
            wide_last=True,
        )
    if delayed:
        scale_state._init(signature, kmean, qs, ks, vs)
    return {
        "q": outputs[0],
        "k": outputs[1],
        "v": outputs[2],
        "qs": qs,
        "ks": ks,
        "vs": vs,
        "means": means,
    }


def _prepare_delayed(
    q, k, v, state, stats, means, kmean, qs, ks, vs, outputs, b, h, n, d, nb, bshd, mean_dtype
):
    """Steady state: fused_quantize_stats -> reduce_stats (fresh) -> delayed_check -> fallback_quantize (flagged heads)."""
    factory = {"device": q.device, "dtype": torch.float32}
    used_qs, used_ks = torch.empty((b, h), **factory), torch.empty((b, h), **factory)
    used_vs = torch.empty((b, h, d), **factory)
    flag = torch.empty((b, h), device=q.device, dtype=torch.int32)
    with torch.cuda.device(q.device):
        stream = torch.cuda.current_stream(q.device).cuda_stream
        dtype = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[q.dtype]
        _launch(
            "fused_quantize_stats",
            b * h * nb,
            256,
            [
                q,
                k,
                v,
                stats,
                state.kmean,
                state.qs,
                state.ks,
                state.vs,
                *outputs,
                state.saturations,
            ],
            [n, d, nb, h, dtype, int(bshd)],
            stream,
        )
        _launch(
            "reduce_stats",
            b * h,
            128 if nb == 1 else 1024,
            [stats, means, kmean, qs, ks, vs],
            [n, d, nb, int(mean_dtype == torch.float32)],
            stream,
        )
        _launch(
            "delayed_check",
            b * h,
            d,
            [
                state.kmean,
                state.qs,
                state.ks,
                state.vs,
                kmean,
                qs,
                ks,
                vs,
                flag,
                used_qs,
                used_ks,
                used_vs,
                state.fallbacks,
            ],
            [d],
            stream,
        )
        _launch(
            "fallback_quantize",
            b * h * nb,
            256,
            [q, k, v, flag, kmean, qs, ks, vs, *outputs],
            [n, d, nb, h, dtype, int(bshd)],
            stream,
        )
    return {
        "q": outputs[0],
        "k": outputs[1],
        "v": outputs[2],
        "qs": used_qs,
        "ks": used_ks,
        "vs": used_vs,
        "means": means,
    }


@torch.no_grad()
def grouping(v, clusters=16, iterations=3, centroids=None):
    """CUDA tensor Lloyd grouping; deterministic initialization, optional warm start.

    This eager grouping is outside kernel timing. No Triton kernels are used.
    """
    if v.ndim != 4 or min(v.shape) < 1 or not v.is_cuda or not v.is_floating_point():
        raise ValueError("expected nonempty floating CUDA BHND values")
    if clusters < 1 or iterations < 1:
        raise ValueError("clusters and iterations must be positive")
    b, h, n, d = v.shape
    clusters = min(clusters, n)
    x = v.float().reshape(b * h, n, d)
    if centroids is None:
        centers = x[:, torch.linspace(0, n - 1, clusters, device=v.device).long()].clone()
    else:
        if centroids.shape != (b, h, clusters, d) or centroids.device != v.device:
            raise ValueError("centroids must have shape BHKD on the input device")
        centers = centroids.float().reshape(b * h, clusters, d).clone()
    labels = torch.empty((b * h, n), device=v.device, dtype=torch.int64)
    # ponytail: eager chunked Lloyd; fuse assignment/reduction if grouping dominates.
    for _ in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros((b * h, clusters), device=v.device)
        for start in range(0, n, 4096):
            chunk = x[:, start : start + 4096]
            distances = (
                chunk.square().sum(-1, keepdim=True)
                + centers.square().sum(-1).unsqueeze(1)
                - 2 * chunk @ centers.transpose(-1, -2)
            )
            z = distances.argmin(-1)
            labels[:, start : start + chunk.shape[1]] = z
            sums.scatter_add_(1, z[..., None].expand_as(chunk), chunk)
            counts.scatter_add_(1, z, torch.ones_like(z, dtype=torch.float32))
        centers = torch.where(counts[..., None] > 0, sums / counts.clamp_min(1)[..., None], centers)
    return labels.argsort(dim=-1, stable=True).reshape(b, h, n), centers.reshape(b, h, clusters, d)
