import torch
import numpy as np
import time
from lm_ops import kv_multi_query_attention

import pytest

std = 0.1
batch_size = 2
num_queries = 3
dim = 4
num_heads = 2
cache_size = 1024
Q = torch.normal(mean=0, std=std, size=(batch_size, num_queries, dim)).cuda().to(torch.float32)
K = torch.normal(mean=0, std=std, size=(batch_size, num_queries, dim)).cuda().to(torch.float32)
V = torch.normal(mean=0, std=std, size=(batch_size, num_queries, dim)).cuda().to(torch.float32)
K_cache = torch.normal(mean=0, std=std, size=(cache_size, dim)).cuda().to(torch.float32)
V_cache = torch.normal(mean=0, std=std, size=(cache_size, dim)).cuda().to(torch.float32)
cache_indices = torch.LongTensor([16, 24, 1, 2]).to(torch.int32).cuda()
offsets = torch.LongTensor([2, 4]).to(torch.int32).cuda()

def reference_paged_kv_multi_query_attention(Q, K, V, K_cache, V_cache, cache_indices, num_heads):
    scale = (dim / num_heads) ** -0.5
    Q = Q.reshape(batch_size, num_queries, num_heads, dim // num_heads).permute(0, 2, 1, 3)
    K = K.reshape(batch_size, num_queries, num_heads, dim // num_heads).permute(0, 2, 1, 3)
    V = V.reshape(batch_size, num_queries, num_heads, dim // num_heads).permute(0, 2, 1, 3)
    K_cache = K_cache[cache_indices, :]
    V_cache = V_cache[cache_indices, :]
    K_cache = K_cache.reshape(batch_size, 2, num_heads, dim // num_heads).permute(0, 2, 1, 3)
    V_cache = V_cache.reshape(batch_size, 2, num_heads, dim // num_heads).permute(0, 2, 1, 3)
    new_K = torch.concat([K_cache, K], dim=2) # [batch_size, num_heads, context_size, dim // num_heads]
    new_V = torch.concat([V_cache, V], dim=2) # [batch_size, num_heads, context_size, dim // num_heads]
    S = torch.matmul(
        Q, # [batch_size, num_heads, num_queries, dim // num_heads]
        new_K.permute(0, 1, 3, 2), # [ba    tch_size, num_head, context_size + num_queries, dim // num_heads]
    ) * scale
    mask = torch.cat([torch.ones(num_queries, 2), torch.tril(torch.ones(num_queries, num_queries))], dim=-1).reshape(1, 1, num_queries, num_queries + 2)
    S = S - (1.0 - mask.to(S.device)) * 1e8

    P = S.softmax(dim=-1) # [batch_size, num_head, num_queries, context_size + num_queries]

    O = torch.matmul(
        P,
        new_V, # [batch_size, num_head, context_size + num_queries, dim // num_heads]
    ) # [batch_size, num_head, num_queries, dim // num_heads]
    O = O.permute(0, 2, 1, 3)
    O = O.reshape(batch_size, num_queries, dim)
    return S, P, O

S1, P1, O1 = reference_paged_kv_multi_query_attention(Q, K, V, K_cache, V_cache, cache_indices, num_heads)
S2, P2, O2 = kv_multi_query_attention(Q, K, V, K_cache, V_cache, cache_indices, offsets, num_heads)

print(O1.to(torch.float16))
print(O2)