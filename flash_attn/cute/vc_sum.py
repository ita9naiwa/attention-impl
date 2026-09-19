# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# FP8 decoded-P denominator via a one- or two-CTA N16 MMA, using the existing FA4
# descriptor pattern and CUTLASS f614dc40 (BSD-3-Clause) TMEM conventions.
# Caller owns publication, completion and consumed-ack synchronization.
# P occupies stage+64..95; the sum scratch occupies stage+32..47.
import cutlass
from cutlass import Float32, Int32, const_expr, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from flash_attn.cute import mma_sm100_desc as desc


@cute.jit
def sum_mma(sb, tbase, idesc: cutlass.Constexpr, cta_group: cutlass.Constexpr = 2):
    base = const_expr(desc.smem_desc_base_from_tensor(sb, desc.Major.K))
    lo = Int32(base & 0xFFFFFFFF) | desc.make_smem_desc_start_addr(sb.iterator)
    for k in cutlass.range_constexpr(4):
        llvm.inline_asm(
            None,
            [lo.ir_value(), Int32(tbase + 64 + k * 8).ir_value(), Int32(tbase + 32).ir_value()],
            "{ .reg .b64 b; .reg .b32 id; .reg .pred leader; "
            f"mov.b64 b, {{$0,{hex(base >> 32)}}}; mov.b32 id,{hex(idesc)}; "
            "elect.sync _|leader,-1; "
            f"@leader tcgen05.mma.cta_group::{cta_group}.kind::f8f6f4 [$2], [$1], b, id, {0 if k == 0 else 1}; }}",
            "r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def load_sum(tbase, tid):
    addr = Int32(tbase + 32 + (tid // 32) * 32 * 65536)
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [addr.ir_value()],
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$1]; tcgen05.wait::ld.sync.aligned;",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def encode4(a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None):
    value = llvm.inline_asm(
        T.i32(),
        [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
        "{ .reg .b32 c0,c1,c2,c3,hi; "
        "cvt.rni.u32.f32 c0,$1; cvt.rni.u32.f32 c1,$2; "
        "cvt.rni.u32.f32 c2,$3; cvt.rni.u32.f32 c3,$4; "
        "cvt.pack.sat.u8.s32.b32 hi,c3,c2,0; "
        "cvt.pack.sat.u8.s32.b32 $0,c1,c0,hi; }",
        "=r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Uint32(value)


@cute.jit
def encode_only(scores: cute.Tensor, probabilities: cute.Tensor):
    packed = cute.make_tensor(
        cute.recast_ptr(probabilities.iterator, dtype=cutlass.Uint32),
        cute.make_layout(cute.size(scores) // 4),
    )
    for i in cutlass.range_constexpr(0, cute.size(scores), 4):
        packed[i // 4] = encode4(scores[i], scores[i + 1], scores[i + 2], scores[i + 3])
