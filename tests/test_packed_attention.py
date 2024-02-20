import torch
import numpy as np
import pytest

from lm_ops import packed_attention

from test_util import get_packed_qkv

def reference_MHA(Q, K, V, mask=None, num_heads=1):
    batch_size, context_size, dim = Q.shape
    _Q = Q.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)
    _K = K.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)
    scale = (dim / num_heads) ** -0.5
    S = torch.bmm(_Q, _K.permute(0, 2, 1)).reshape(batch_size, num_heads, context_size, context_size)
    S *= scale
    print(scale)
    if mask is not None:
        S += (mask.unsqueeze(1) - 1) * 10000.0
    P = S.softmax(dim=-1)
    O = torch.matmul(P, V.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3))
    O = O.permute(0, 2, 1, 3).reshape(batch_size, context_size, dim)
    return S, P, O

std = 0.1

@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("context_size", [32, 64])
@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_packed_embedding_with_same_seq_len(batch_size, context_size, dim, num_heads, dtype):
    offsets = torch.IntTensor([(1 + i) * context_size for i in range(batch_size)]).cuda()
    Q, K, V = get_packed_qkv(batch_size * context_size, dim, torch.float16, True, std)
    S2, P2, O2 = packed_attention(Q, K, V, offsets, num_heads)

    O2 = O2.reshape(batch_size, context_size, -1)
    Q = Q.reshape(batch_size, context_size, dim)
    K = K.reshape(batch_size, context_size, dim)
    V = V.reshape(batch_size, context_size, dim)
    mask = torch.from_numpy(np.tril(np.ones(context_size).astype(np.float32))).reshape(1, context_size, context_size).repeat(batch_size, 1, 1).cuda()

    S1, P1, O1 = reference_MHA(Q, K, V, mask=mask, num_heads=num_heads)
    assert torch.allclose(O1, O2, atol=2 * 1e-2)


def test_packed_embedding_with_different_seq_len(batch_size, context_size, dim, num_heads, dtype):
    offsets = torch.IntTensor([(1 + i) * context_size for i in range(batch_size)]).cuda()
    raise NotImplementedError
