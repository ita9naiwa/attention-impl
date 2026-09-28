"""Native CUDA centering, Q/K Hadamard rotation and E4M3 quantization for VC attention.

Input BHND (or BSHD with bshd=True), output BNHD. Q/K use per-head scales (K is centered); V uses per-channel scales.
Q/K orthonormal Hadamard rotation is always applied; RoPE fusion is omitted.
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
        ("reduce_stats", "fused_stats", "fused_quantize"),
    )
    driver.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _check(driver.cuFuncSetAttribute(functions["fused_stats"], 8, 36864), "cuFuncSetAttribute")
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
            else 0,
            stream,
            args,
            None,
        ),
        name,
    )


@torch.no_grad()
def prepare(q, k, v, permutation=None, smooth=True, hadamard=True, bshd=False):
    """Prepare finite contiguous CUDA BHND or BSHD inputs; returns FP8 q/k/v and qs/ks/vs.

    Only the fused path is implemented: smooth=False, no permutation, Hadamard on. V-Smooth
    (smooth=True), token permutation and hadamard=False were removed; they raise NotImplementedError.
    The caller owns validation of finite input values; shape/dtype/device validation is
    synchronization-free and always performed.
    """
    if smooth or permutation is not None or not hadamard:
        raise NotImplementedError("prepare() supports only smooth=False, permutation=None, hadamard=True "
                                  "(V-Smooth, permutation and the unrotated path were removed)")
    if q.ndim != 4 or min(q.shape) < 1 or q.shape[-1] not in (64, 128):
        raise ValueError("expected nonempty 4D input with D=64/128")
    if not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("expected CUDA BF16, FP16, or FP32 inputs")
    if any(
        x.shape != q.shape or x.dtype != q.dtype or x.device != q.device or not x.is_contiguous()
        for x in (q, k, v)
    ):
        raise ValueError("q, k, v must share shape/dtype/device and be contiguous")
    b, h, n, d = q.transpose(1, 2).shape if bshd else q.shape
    if b > 65535 or h > 65535:
        raise ValueError("batch and heads must be <= 65535 (quantize grid y/z limits)")
    nb = (n + 127) // 128
    factory = {"device": q.device, "dtype": torch.float32}
    qs, ks = [torch.empty((b, h), **factory) for _ in range(2)]
    vs = torch.empty((b, h, d), **factory)
    stats = torch.empty((b, h, nb, 6, d), **factory)
    kmean = torch.empty((b, h, d), **factory)
    outputs = [
        torch.empty((b, n, h, d), device=q.device, dtype=torch.float8_e4m3fn) for _ in range(3)
    ]
    with torch.cuda.device(q.device):
        stream = torch.cuda.current_stream(q.device).cuda_stream
        dtype = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[q.dtype]
        _launch(
            "fused_stats",
            b * h * nb,
            256,
            [q, k, v, None, stats],
            [n, d, nb, h, dtype, dtype, 0, int(bshd), int(bshd)],
            stream,
        )
        _launch("reduce_stats", b * h, 128 if nb == 1 else 1024, [stats, kmean, qs, ks, vs], [n, d, nb], stream)
        _launch(
            "fused_quantize",
            ((n + 7) // 8, h, b),
            256,
            [q, k, v, None, kmean, qs, ks, vs, stats, *outputs],
            [n, h, d, nb, b, dtype, dtype, int(bshd), int(bshd), b * h * n * d],
            stream,
            wide_last=True,
        )
    return {"q": outputs[0], "k": outputs[1], "v": outputs[2], "qs": qs, "ks": ks, "vs": vs}
