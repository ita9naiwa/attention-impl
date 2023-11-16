import torch
import numpy as np
import time
from paged_attention import paged_kv_attention_forward

std = 0.1
batch_size = 1
dim = 16
num_heads = 4
cache_size = 1024
Q = torch.normal(mean=0, std=std, size=(1, dim)).cuda().to(torch.float32)
K = torch.normal(mean=0, std=std, size=(1, dim)).cuda().to(torch.float32)
V = torch.normal(mean=0, std=std, size=(1, dim)).cuda().to(torch.float32)
K_cache = torch.normal(mean=0, std=std, size=(cache_size, dim)).cuda().to(torch.float32)
V_cache = torch.normal(mean=0, std=std, size=(cache_size, dim)).cuda().to(torch.float32)
cache_indices = torch.LongTensor([16, 24]).to(torch.int32).cuda()
def reference_kv_MHA(Q, K, V, K_cache, V_cache, cache_indices):
    scale = (dim / num_heads) ** -0.5
    Q = Q.reshape(batch_size, num_heads, dim // num_heads)
    K = K.reshape(batch_size, num_heads, dim // num_heads)
    V = V.reshape(batch_size, num_heads, dim // num_heads)
    K_cache = K_cache[cache_indices, :]
    V_cache = V_cache[cache_indices, :]
    K_cache = K_cache.reshape(batch_size, len(cache_indices), num_heads, dim // num_heads).permute(0, 2, 1, 3)
    V_cache = V_cache.reshape(batch_size, len(cache_indices), num_heads, dim // num_heads).permute(0, 2, 1, 3)
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


beg = time.perf_counter()
S1, P1, O1 = reference_kv_MHA(Q, K, V, K_cache, V_cache, cache_indices)
end = time.perf_counter()
print(f"reference MHA implementation: {end - beg:0.4f} secs")
beg = time.perf_counter()
S2, P2, O2 = paged_kv_attention_forward(Q, K, V, K_cache, V_cache, cache_indices, num_heads)
end = time.perf_counter()
print(f"CUDA MHA implementation: {end - beg:0.4f} secs")

print("======(Max diff) Accuracy compared to ref implementation======")
print(torch.abs(S1 - S2).max())
print(torch.abs(P1 - P2).max())
print(torch.abs(O1 - O2).max())
