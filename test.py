import torch
import time
from attention import naive_attention_forward


std = 0.1

batch_size = 4
context_size = 2048
dim = 32
num_heads = 4
Q = torch.normal(mean=0, std=std,size=(batch_size, context_size, dim)).cuda()
K = torch.normal(mean=0, std=std,size=(batch_size, context_size, dim)).cuda()
V = torch.normal(mean=0, std=std,size=(batch_size, context_size, dim)).cuda()

def reference_MHA(Q, K, V):
    _Q = Q.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)
    _K = K.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)

    S = torch.bmm(_Q, _K.permute(0, 2, 1)).reshape(batch_size, num_heads, context_size, context_size)
    P = S.softmax(dim=-1)

    O = torch.matmul(P, V.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3))
    O = O.permute(0, 2, 1, 3).reshape(batch_size, context_size, dim)
    return S, P, O


beg = time.time()
S1, P1, O1 = reference_MHA(Q, K, V)
end = time.time()
print(end - beg)

beg = time.time()
S2, P2, O2 = naive_attention_forward(Q, K, V, num_heads)
end = time.time()
print(end - beg)
print('======')
print(torch.abs(S1 - S2).max())
print(torch.abs(P1 - P2).max())
print(torch.abs(O1 - O2).max())
