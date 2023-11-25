## CUDA multi head attention implementation

For study purpose

implemented attentions
- Naive Attention
- Attention with KV
- Attention with non-contagious memory
- Attention with non-contagious KV cache (PagedAttention with block size 1)

### comparison with MHA implementation

```python
def reference_MHA(Q, K, V):
    _Q = Q.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)
    _K = K.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3).reshape(batch_size * num_heads, context_size, dim // num_heads)

    S = torch.bmm(_Q, _K.permute(0, 2, 1)).reshape(batch_size, num_heads, context_size, context_size)
    P = S.softmax(dim=-1)

    O = torch.matmul(P, V.reshape(batch_size, context_size, num_heads, dim // num_heads).permute(0, 2, 1, 3))
    O = O.permute(0, 2, 1, 3).reshape(batch_size, context_size, dim)
    return S, P, O
...

beg = time.perf_counter()
S1, P1, O1 = reference_MHA(Q, K, V)
end = time.perf_counter()
print(f"reference MHA implementation: {end - beg:0.2f} secs")

beg = time.perf_counter()
S2, P2, O2 = naive_attention_forward(Q, K, V, num_heads)
end = time.perf_counter()
print(f"CUDA MHA implementation: {end - beg:0.2f} secs")
```

```bash
reference MHA implementation: 0.03 secs
CUDA MHA implementation: 0.01 secs
```

about 3 times faster!
