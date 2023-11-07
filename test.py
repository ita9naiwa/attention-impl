import torch
import numpy as np
import time
from attention import naive_attention_forward


std = 0.1

batch_size = 8
context_size = 2048
dim = 32
num_heads = 4

Q = torch.normal(mean=0, std=std,size=(batch_size, context_size, dim)).cuda()
K = torch.normal(mean=0, std=std,size=(batch_size, context_size, dim)).cuda()
V = torch.normal(mean=0, std=std,size=(batch_size, context_size, dim)).cuda()
mask = torch.from_numpy(np.tril(np.ones(context_size).astype(np.float32))).reshape(1, context_size, context_size).repeat(batch_size, 1, 1).cuda()

def reference_MHA(Q, K, V, mask=None):
    _Q = Q.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)
    _K = K.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)

    S = torch.bmm(_Q, _K.permute(0, 2, 1)).reshape(batch_size, num_heads, context_size, context_size)
    if mask is not None:
        S += (mask.unsqueeze(1) - 1) * 100000.0
    P = S.softmax(dim=-1)
    O = torch.matmul(P, V.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3))
    O = O.permute(0, 2, 1, 3).reshape(batch_size, context_size, dim)
    return S, P, O

beg = time.perf_counter()
S1, P1, O1 = reference_MHA(Q, K, V, mask=mask)
end = time.perf_counter()
print(f"reference MHA implementation: {end - beg:0.4f} secs")
beg = time.perf_counter()
S2, P2, O2 = naive_attention_forward(Q, K, V, mask, num_heads)
end = time.perf_counter()
print(f"CUDA MHA implementation: {end - beg:0.4f} secs")

print("======(Max diff) Accuracy compared to ref implementation======")
print(torch.abs(S1 - S2).max())
print(torch.abs(P1 - P2).max())
print(torch.abs(O1 - O2).max())
