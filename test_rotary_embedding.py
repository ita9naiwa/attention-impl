import torch
from torch import nn
import numpy as np
import time
from rotary_embedding import rotary_embedding_inplace




class RefRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=self.inv_freq.device,
                                dtype=torch.float16)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, q, k, positions, return_cos_sin=False):
        cos = self.cos_cached[positions].unsqueeze(1)
        sin = self.sin_cached[positions].unsqueeze(1)

        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        if return_cos_sin:
            return cos, sin, q, k
        else:
            return q, k

    @staticmethod
    def _rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class MyRotaryEmbedding(nn.Module):
    def __init__(self, head_dim, rot_dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.head_dim = head_dim
        self.dim = rot_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=self.inv_freq.device,
                                dtype=torch.float16)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(dtype), persistent=False)

    def forward(self, q, k, positions, return_cos_sin=False):
        rotary_embedding_inplace(q, k, positions, self.cos_cached, self.sin_cached,
                                 self.head_dim)
        return q, k

if __name__ == "__main__":
    print("when rot_dim is equal to head_dim")
    num_heads = 1
    head_dim = 16
    rot_dim = 16
    max_position_embeddings = 32
    input_size = 4
    q1 = torch.normal(mean=0, std=0.1, size=(input_size, num_heads, head_dim)).cuda().to(torch.float16)
    k1 = torch.normal(mean=0, std=0.1, size=(input_size, num_heads, head_dim)).cuda().to(torch.float16)
    q2 = q1.clone().reshape(input_size, -1)
    k2 = k1.clone().reshape(input_size, -1)

    positions = torch.arange(input_size).cuda().int()
    ref_rotary_emb = RefRotaryEmbedding(dim=head_dim,
                                        max_position_embeddings=max_position_embeddings,
                                        base=10000,
                                        device="cuda")
    my_rotary_emb = MyRotaryEmbedding(head_dim=head_dim,
                                      rot_dim=rot_dim,
                                      max_position_embeddings=max_position_embeddings,
                                      base=10000,
                                      device="cuda")
    q1, k1 = ref_rotary_emb.forward(q1, k1, positions)
    q2, k2 = my_rotary_emb.forward(q2, k2, positions)
    q1 = q1.reshape(input_size, -1)
    k1 = k1.reshape(input_size, -1)
    print("======(Max diff) Accuracy compared to ref implementation======")
    print((q1 - q2).abs().sum())


    print("when rot_dim != head_dim")
    num_heads = 1
    head_dim = 32
    rot_dim = 16
    max_position_embeddings = 32
    input_size = 4
    q1 = torch.normal(mean=0, std=0.1, size=(input_size, num_heads, head_dim)).cuda().to(torch.float16)
    k1 = torch.normal(mean=0, std=0.1, size=(input_size, num_heads, head_dim)).cuda().to(torch.float16)
    q2 = q1.clone().reshape(input_size, -1)
    k2 = k1.clone().reshape(input_size, -1)

    positions = torch.arange(input_size).cuda().int()

    ref_rotary_emb = RefRotaryEmbedding(dim=rot_dim,
                                        max_position_embeddings=max_position_embeddings,
                                        base=10000,
                                        device="cuda")

    my_rotary_emb = MyRotaryEmbedding(head_dim=head_dim,
                                      rot_dim=rot_dim,
                                      max_position_embeddings=max_position_embeddings,
                                      base=10000,
                                      device="cuda")

    q_rot, q_pass = (q1[..., : rot_dim], q1[..., rot_dim :])
    k_rot, k_pass = (k1[..., : rot_dim], k1[..., rot_dim :])
    q_rot, k_rot = ref_rotary_emb.forward(q_rot, k_rot, positions)
    q1 = torch.cat((q_rot, q_pass), dim=-1)
    k1 = torch.cat((k_rot, k_pass), dim=-1)

    q2, k2 = my_rotary_emb.forward(q2, k2, positions)
    q1 = q1.reshape(input_size, -1)
    k1 = k1.reshape(input_size, -1)
    print("======(Max diff) Accuracy compared to ref implementation======")
    print((q1 - q2).abs().sum())