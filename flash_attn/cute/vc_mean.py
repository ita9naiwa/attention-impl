"""Fused mean restoration for VC-Attention's FP8 FA4 path.

BF16 normalized means use one K16 BF16 MMA per four KV blocks. Each pending
row sum remains FP32 while online-softmax rescaling is applied, then is split
into three BF16 components for the MMA. FP32 metadata uses nine BF16 products
per block. Both paths accumulate into the existing FP32 PV accumulator.

P's publication barrier protects the TMEM row-sum operand. The next QK cannot
reuse its TMEM columns until the preceding PV and mean MMAs have completed.
"""

import cutlass
from cutlass import BFloat16, Float32, Int32, const_expr, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils import blackwell_helpers as basic

from flash_attn.cute import mma_sm100_desc as desc


@cute.jit
def store_mean_fp32(self, row_sum, means, sMean, stage, tidx, cta_idx):
    r0 = BFloat16(row_sum)
    r1 = BFloat16(row_sum - Float32(r0))
    r2 = BFloat16((row_sum - Float32(r0)) - Float32(r1))
    a = cute.make_rmem_tensor(16, BFloat16)
    for k in cutlass.range_constexpr(16):
        if const_expr(k < 3):
            a[k] = r0
        elif const_expr(k < 6):
            a[k] = r1
        elif const_expr(k < 9):
            a[k] = r2
        else:
            a[k] = BFloat16(0)
    packed = cute.make_tensor(cute.recast_ptr(a.iterator, dtype=Int32), cute.make_layout(8))
    addr = Int32(96 + stage * 128 + ((tidx // 32) * 32) * 65536)
    llvm.inline_asm(
        None,
        [addr.ir_value()] + [packed[k].ir_value() for k in range(8)],
        "tcgen05.st.sync.aligned.32x32b.x8.b32 [$0], {$1,$2,$3,$4,$5,$6,$7,$8};",
        "r,r,r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    n = self.head_dim_v_padded // self.cta_group_size
    mat = cute.make_tensor(
        sMean.iterator, cute.make_layout((n, 16, self.q_stage), stride=(16, 1, n * 16))
    )
    if tidx < n:
        v = Float32(means[cta_idx * n + tidx])
        v0 = BFloat16(v)
        v1 = BFloat16(v - Float32(v0))
        v2 = BFloat16((v - Float32(v0)) - Float32(v1))
        vals = cute.make_rmem_tensor(16, BFloat16)
        for k in cutlass.range_constexpr(16):
            if const_expr(k >= 9):
                vals[k] = BFloat16(0)
            elif const_expr(k % 3 == 0):
                vals[k] = v0
            elif const_expr(k % 3 == 1):
                vals[k] = v1
            else:
                vals[k] = v2
        cute.autovec_copy(vals, mat[tidx, None, stage])
    cute.arch.fence_proxy("async.shared", space="cta")


@cute.jit
def mean_mma(self, sMean, stage):
    op = basic.make_trivial_tiled_mma(
        BFloat16,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        Float32,
        tcgen05.CtaGroup.TWO if self.cta_group_size == 2 else tcgen05.CtaGroup.ONE,
        self.mma_tiler_pv[:2],
        tcgen05.OperandSource.TMEM,
    ).op
    idesc = const_expr(desc.mma_op_to_idesc(op))
    sb = sMean[None, None, None, stage]
    base = const_expr(desc.smem_desc_base_from_tensor(sb, desc.Major.K))
    lo = Int32(base & 0xFFFFFFFF) | desc.make_smem_desc_start_addr(sb.iterator)
    llvm.inline_asm(
        None,
        [
            lo.ir_value(),
            Int32(96 + 128 * stage).ir_value(),
            Int32(self.tmem_o_offset[0] + self.head_dim_v_padded * stage).ir_value(),
        ],
        "{ .reg .b64 b; .reg .b32 id; .reg .pred leader; "
        f"mov.b64 b, {{$0,{hex(base >> 32)}}}; mov.b32 id,{hex(idesc)}; "
        "elect.sync _|leader,-1; "
        f"@leader tcgen05.mma.cta_group::{self.cta_group_size}.kind::f16 [$2], [$1], b, id, 1; }}",
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def store_mean_operands(
    self, row_sum, means, sMean, stage, tidx, cta_idx, pending_sum, acc_scale, n_block, is_first
):
    if const_expr(means.element_type is not BFloat16):
        store_mean_fp32(self, row_sum, means, sMean, stage, tidx, cta_idx)
    else:
        n = self.head_dim_v_padded // self.cta_group_size
        mat = cute.make_tensor(
            sMean.iterator, cute.make_layout((n, 16, self.q_stage), stride=(16, 1, n * 16))
        )
        slot = 3 - n_block % 4
        if tidx < n:
            v = means[cta_idx * n + tidx]
            if const_expr(is_first):
                zeros = cute.make_rmem_tensor(16, BFloat16)
                zeros.fill(BFloat16(0))
                cute.autovec_copy(zeros, mat[tidx, None, stage])
            values = cute.make_rmem_tensor(4, BFloat16)
            values.fill(v)
            dest = cute.domain_offset((slot * 4,), mat[tidx, None, stage])
            dest = cute.composition(dest, cute.make_layout(4))
            cute.autovec_copy(values, dest)
        for j in cutlass.range_constexpr(4):
            old = Float32(0) if const_expr(is_first) else pending_sum[j] * acc_scale
            pending_sum[j] = row_sum if slot == j else old
        if n_block % 4 == 0:
            a = cute.make_rmem_tensor(16, BFloat16)
            for part in cutlass.range_constexpr(4):
                r = pending_sum[part]
                r0 = BFloat16(r)
                r1 = BFloat16(r - Float32(r0))
                r2 = BFloat16((r - Float32(r0)) - Float32(r1))
                a[part * 4] = r0
                a[part * 4 + 1] = r1
                a[part * 4 + 2] = r2
                a[part * 4 + 3] = BFloat16(0)
            packed = cute.make_tensor(cute.recast_ptr(a.iterator, dtype=Int32), cute.make_layout(8))
            addr = Int32(96 + stage * 128 + ((tidx // 32) * 32) * 65536)
            llvm.inline_asm(
                None,
                [addr.ir_value()] + [packed[k].ir_value() for k in range(8)],
                "tcgen05.st.sync.aligned.32x32b.x8.b32 [$0], {$1,$2,$3,$4,$5,$6,$7,$8};",
                "r,r,r,r,r,r,r,r,r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        cute.arch.fence_proxy("async.shared", space="cta")
