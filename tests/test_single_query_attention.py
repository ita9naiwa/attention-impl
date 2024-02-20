import torch
import numpy as np
import time
from lm_ops import single_query_attention

from test_util import get_packed_qkv, get_kv_cache
import pytest

def reference_kv_MHA(Q, K, V, K_cache, V_cache, num_heads=1):
    batch_size, dim = Q.shape
    batch_size, context_size, dim = K_cache.shape
    scale = (dim / num_heads) ** -0.5
    Q = Q.reshape(batch_size, num_heads, dim // num_heads)
    K = K.reshape(batch_size, num_heads, dim // num_heads)
    V = V.reshape(batch_size, num_heads, dim // num_heads)

    K_cache = K_cache.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3)
    V_cache = V_cache.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3)
    new_K = torch.concat([K_cache, K.unsqueeze(2)], dim=2)
    new_V = torch.concat([V_cache, V.unsqueeze(2)], dim=2)
    S = torch.matmul(
        new_K, # [batch_size, num_head, context_size + 1, dim // num_heads]
        Q.unsqueeze(-1), # [batch_size, num_heads, dim // num_heads, 1]
    ).squeeze(-1) # [batch_size, num_head, context_size + 1]
    P = S.softmax(dim=-1)

    O = torch.matmul(
        P.unsqueeze(2), # [batch_size, num_head, contetx_size + 1]
        new_V, # [batch_size, num_head, context_size + 1, dim // num_heads]
    ).squeeze(2) # [batch_size, num_head, dim]
    O = O.reshape(batch_size, dim)
    return S, P, O


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("context_size", [256, 512])
@pytest.mark.parametrize("dim", [32, 64, 256])
@pytest.mark.parametrize("num_heads", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_kv_attention(batch_size, context_size, dim, num_heads, dtype):
    Q, K, V = get_packed_qkv(batch_size=batch_size, dim=dim, dtype=dtype, to_cuda=True)
    K_cache, V_cache = get_kv_cache(batch_size=batch_size,
                                    context_size=context_size,
                                    dim=dim, dtype=dtype, to_cuda=True)
    S1, P1, O1 = reference_kv_MHA(Q, K, V, K_cache, V_cache)
    S2, P2, O2 = single_query_attention(Q, K, V, K_cache, V_cache, num_heads)
    torch.allclose(O1, O2, atol=1e-2)


if __name__ == "__main__":
    test_kv_attention(32, 1024, 64, 8, torch.float16)