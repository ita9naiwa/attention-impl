import torch
from attention import naive_attention_forward



batch_size = 4
context_size = 64
dim = 16
num_heads = 4
Q = torch.normal(mean=0, std=0.1,size=(batch_size, context_size, dim)).cuda()
K = torch.normal(mean=0, std=0.1,size=(batch_size, context_size, dim)).cuda()
V = torch.normal(mean=0, std=0.1,size=(batch_size, context_size, dim)).cuda()

def reference_MHA(Q, K, V):
    _Q = Q.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)
    _K = K.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)

    S = torch.bmm(_Q, _K.permute(0, 2, 1)).reshape(batch_size, num_heads, context_size, context_size)
    P = S.softmax(dim=-1)
    O = torch.matmul(P, V.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3))
    O = O.permute(0, 2, 1, 3).reshape(batch_size, context_size, dim)
    return S, P, O

S2, P2, O2 = naive_attention_forward(Q, K, V, num_heads)