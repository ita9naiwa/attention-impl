"""Integer length subclasses must cross the CuTe scalar ABI as builtin ints."""

import pytest
import torch

from flash_attn.cute import interface


class SequenceLength(int):
    pass


@pytest.mark.parametrize("seqlen", [128, 384])
def test_max_seqlen_integer_subclass(seqlen, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("SM100/SM110 runtime max sequence length argument")
    # Compile the subclass call first; a cached builtin-int specialization could
    # otherwise hide an unsupported argument during compilation.
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", {})
    torch.manual_seed(53)
    q, k, v = [
        torch.randn(1, seqlen, 2, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    actual = interface._flash_attn_fwd(
        q, k, v, max_seqlen_q=SequenceLength(seqlen), return_lse=True
    )[:2]
    expected = interface._flash_attn_fwd(
        q, k, v, max_seqlen_q=seqlen, return_lse=True
    )[:2]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)
