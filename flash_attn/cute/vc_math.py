"""Packed native PTX ExpCast for pre-encoded score fragments.

The input is c = 8 * (score - row_max) * log2(e) + 119.65, not u+8.
With the paper's exact running maximum, c <= 119.65; negative infinity is
valid and converts to zero. No rescale-threshold approximation is permitted.
"""

import cutlass
from cutlass import Float32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@cute.jit
def scale_expcast_scores(scores: cute.Tensor, row_max: Float32, scale_log2: Float32):
    """Fuse score scaling, max subtraction and Eq.7's affine code map."""
    scale = scale_log2 * Float32(8.0)
    bias = Float32(119.65) - row_max * scale
    for i in cutlass.range_constexpr(0, cute.size(scores), 2):
        scores[i], scores[i + 1] = cute.arch.fma_packed_f32x2(
            (scores[i], scores[i + 1]), (scale, scale), (bias, bias)
        )


@dsl_user_op
def _expcast_accumulate4(
    c0: Float32,
    c1: Float32,
    c2: Float32,
    c3: Float32,
    a0: Float32,
    a1: Float32,
    a2: Float32,
    a3: Float32,
    *,
    loc=None,
    ip=None,
):
    """Encode four RNE bytes and add their exact decoded values to FP32 sums.

    Unsigned conversion clamps negatives, including masked negative infinity.
    The CVT/pack sequence fuses to native F2IP on Blackwell; inserting clamps
    or byte permutations prevents that fusion. Mixed add lowers to FHADD,
    avoiding separate half-to-FP32 conversions and addition instructions.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32(), T.f32(), T.f32(), T.f32()]),
        [Float32(x).ir_value(loc=loc, ip=ip) for x in (c0, c1, c2, c3, a0, a1, a2, a3)],
        "{ .reg .b32 c0,c1,c2,c3,hi,h01,h23; .reg .b16 b01,b23,h0,h1,h2,h3; "
        "cvt.rni.u32.f32 c0,$5; cvt.rni.u32.f32 c1,$6; "
        "cvt.rni.u32.f32 c2,$7; cvt.rni.u32.f32 c3,$8; "
        "cvt.pack.sat.u8.s32.b32 hi,c3,c2,0; "
        "cvt.pack.sat.u8.s32.b32 $0,c1,c0,hi; "
        "mov.b32 {b01,b23},$0; "
        "cvt.rn.f16x2.e4m3x2 h01,b01; cvt.rn.f16x2.e4m3x2 h23,b23; "
        "mov.b32 {h0,h1},h01; mov.b32 {h2,h3},h23; "
        "add.rn.f32.f16 $1,h0,$9; add.rn.f32.f16 $2,h1,$10; "
        "add.rn.f32.f16 $3,h2,$11; add.rn.f32.f16 $4,h3,$12; }",
        "=r,=f,=f,=f,=f,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        cutlass.Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        *(Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip)) for i in range(1, 5)),
    )


@cute.jit
def apply_expcast_sum(scores: cute.Tensor, probabilities: cute.Tensor):
    """Store E4M3 probabilities and return their exact FP32 block sum.

    Eight independent chains hide mixed-add latency. Every decoded value is
    a multiple of 1/512, and at most 128 values are summed, so integer units
    never exceed 2**24 and the FP32 sum is exact regardless of addition order.
    """
    assert cute.size(scores) % 8 == 0
    assert cute.size(scores) <= 128
    packed = cute.make_tensor(
        cute.recast_ptr(probabilities.iterator, dtype=cutlass.Uint32),
        cute.make_layout(cute.size(scores) // 4),
    )
    a0, a1, a2, a3 = Float32(0), Float32(0), Float32(0), Float32(0)
    a4, a5, a6, a7 = Float32(0), Float32(0), Float32(0), Float32(0)
    for i in cutlass.range_constexpr(0, cute.size(scores), 8):
        c0, a0, a1, a2, a3 = _expcast_accumulate4(
            scores[i], scores[i + 1], scores[i + 2], scores[i + 3], a0, a1, a2, a3
        )
        c1, a4, a5, a6, a7 = _expcast_accumulate4(
            scores[i + 4], scores[i + 5], scores[i + 6], scores[i + 7], a4, a5, a6, a7
        )
        packed[i // 4], packed[i // 4 + 1] = c0, c1
    s0 = cute.arch.add_packed_f32x2((a0, a1), (a2, a3))
    s1 = cute.arch.add_packed_f32x2((a4, a5), (a6, a7))
    s0 = cute.arch.add_packed_f32x2(s0, s1)
    return s0[0] + s0[1]
