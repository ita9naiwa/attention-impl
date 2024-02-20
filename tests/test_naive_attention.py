import pytest

import torch
import numpy as np
import time

from attention import naive_attention_forward

from test_util import get_qkv

std = 0.1

def reference_MHA(Q, K, V, mask=None, num_heads=1):
    batch_size, context_size, dim = Q.shape
    _Q = Q.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)
    _K = K.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)
    scale = (dim / num_heads) ** -0.5
    S = torch.bmm(_Q, _K.permute(0, 2, 1)).reshape(batch_size, num_heads, context_size, context_size)
    S *= scale
    if mask is not None:
        S += (mask.unsqueeze(1) - 1) * 100000.0
    P = S.softmax(dim=-1)
    O = torch.matmul(P, V.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3))
    O = O.permute(0, 2, 1, 3).reshape(batch_size, context_size, dim)
    return S, P, O

@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("context_size", [256, 512])
@pytest.mark.parametrize("dim", [32, 64])
@pytest.mark.parametrize("num_heads", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mha(batch_size, context_size, dim, num_heads, dtype):
    Q, K, V, mask = get_qkv(batch_size=batch_size, context_size=context_size, dim=dim, std=std, dtype=dtype, to_cuda=True)
    S1, P1, O1 = reference_MHA(Q, K, V, mask=mask, num_heads=num_heads)
    S2, P2, O2 = naive_attention_forward(Q, K, V, mask, num_heads)
    assert torch.allclose(O1, O2, atol=1e-2)


if __name__ == "__main__":
    test_mha(1, 256, 64, 1)