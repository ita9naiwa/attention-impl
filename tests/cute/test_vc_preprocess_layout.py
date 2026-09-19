"""BHND and BNHD inputs must describe identical attention values."""

import torch
from flash_attn.cute.vc_preprocess import _rotate_qk, prepare


@torch.no_grad()
def test_fused_stats_preserve_row_order():
    torch.manual_seed(91)
    for d in (64, 128):
        for n in (31, 32, 33, 127, 129, 257):
            q, k, v = [
                torch.randn((1, 2, n, d), device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            if d == 128 and n in (33, 129, 257):
                # Contiguous views may still be misaligned for quartet loads.
                views = []
                for offset, value in enumerate((q, k, v), start=1):
                    storage = torch.empty(
                        value.numel() + offset, device=value.device, dtype=value.dtype
                    )
                    view = storage[offset:].view_as(value)
                    view.copy_(value)
                    views.append(view)
                q, k, v = views
            rq, rk = _rotate_qk(q, k)
            # The unfused path retains the original serial row sum independently.
            expected = prepare(rq, rk, v.float(), smooth=False, hadamard=False)
            actual = prepare(q, k, v, smooth=False)
            for key in expected:
                assert torch.equal(
                    actual[key].view(torch.uint8), expected[key].view(torch.uint8)
                ), (d, n, key)


@torch.no_grad()
def test_preprocess_layout():
    torch.manual_seed(42)
    for d in (64, 128):
        inputs = [
            torch.randn((2, 129, 3, d), device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        for smooth in (False, True):
            expected = prepare(
                *(x.transpose(1, 2).contiguous() for x in inputs), smooth=smooth
            )
            actual = prepare(*inputs, smooth=smooth, bshd=True)
            for key in expected:
                assert torch.equal(
                    actual[key].view(torch.uint8), expected[key].view(torch.uint8)
                ), (d, smooth, key)
    # CUDA grid y/z cap: retain support for large batch/head counts at N=1.
    for shape in ((1, 1, 65536, 64), (65536, 1, 1, 64)):
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        p = prepare(x, x, x, smooth=False, hadamard=False, bshd=True)
        decoded_q = p["q"].float() * p["qs"][:, None, :, None]
        assert (decoded_q - x.float()).norm() / x.float().norm() < 0.04
        assert torch.count_nonzero(p["k"].float()) == 0
        decoded_v = p["v"].float() * p["vs"][:, None, :, :]
        torch.testing.assert_close(decoded_v, x.float(), rtol=2e-6, atol=1e-7)
    print(
        "PASS native preprocessing: layouts, scales, means, large batch/head grid boundaries"
    )


@torch.no_grad()
def test_fused_dynamic_graph():
    torch.manual_seed(151)
    for d in (64, 128):
        for bshd in (False, True):
            shape = (2, 257, 3, d) if bshd else (2, 3, 257, d)
            values = [
                torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            options = {"smooth": False, "bshd": bshd}
            for _ in range(3):
                prepare(*values, **options)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = prepare(*values, **options)
            for _ in range(3):
                for value in values:
                    value.add_(0.02)
                for value in captured.values():
                    value.fill_(float("nan"))
                graph.replay()
                expected = prepare(*values, **options)
                for key in expected:
                    assert torch.equal(
                        captured[key].view(torch.uint8), expected[key].view(torch.uint8)
                    ), key
    print("PASS fused dynamic CUDA Graph: D64/128, BHND/BSHD, poisoned outputs")


if __name__ == "__main__":
    test_fused_stats_preserve_row_order()
    test_preprocess_layout()
    test_fused_dynamic_graph()
