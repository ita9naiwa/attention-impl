import torch
import numpy as np
import time
from attention import naive_attention_forward
from paged_attention import paged_attention_forward

std = 0.1

dim = 4
num_heads = 2
offsets = torch.Tensor([3, 6, 9]).to(torch.int32).cuda()
Q = torch.normal(mean=0, std=std,size=(9, dim)).cuda().to(torch.float16)
K = torch.normal(mean=0, std=std,size=(9, dim)).cuda().to(torch.float16)
V = torch.normal(mean=0, std=std,size=(9, dim)).cuda().to(torch.float16)

S2, P2, O2 = paged_attention_forward(Q, K, V, offsets, num_heads)
O2 = O2.reshape(3, 3, -1)

Q = Q.reshape(3, 3, dim)
K = K.reshape(3, 3, dim)
V = V.reshape(3, 3, dim)
mask = torch.from_numpy(np.tril(np.ones(3).astype(np.float32))).reshape(1, 3, 3).repeat(3, 1, 1).cuda()
def reference_MHA(Q, K, V, mask=None):
    batch_size = Q.shape[0]
    context_size = Q.shape[1]
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

S1, P1, O1 = reference_MHA(Q, K, V, mask=mask)

print(O1, O2)
print(torch.abs(S1 - S2).max())
print(torch.abs(P1 - P2).max())
print(torch.abs(O1 - O2).max())
