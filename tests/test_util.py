import torch
import numpy as np

def get_qkv(batch_size=4, context_size=2048, dim=64, std=1, dtype=torch.float16, to_cuda=False):
    Q = torch.normal(mean=0, std=std, size=(batch_size, context_size, dim), dtype=dtype)
    K = torch.normal(mean=0, std=std, size=(batch_size, context_size, dim), dtype=dtype)
    V = torch.normal(mean=0, std=std, size=(batch_size, context_size, dim), dtype=dtype)
    mask = torch.from_numpy(np.tril(np.ones(context_size))).reshape(1, context_size, context_size).repeat(batch_size, 1, 1).to(dtype).cuda()
    if to_cuda:
        Q = Q.cuda()
        K = K.cuda()
        V = V.cuda()
        mask = mask.cuda()
    return Q, K, V, mask

def get_kv_cache(batch_size=512, context_size=32, dim=64, dtype=torch.float16, to_cuda=False):
    K_cache = torch.zeros(size=(batch_size, context_size, dim), dtype=dtype)
    V_cache = torch.zeros(size=(batch_size, context_size, dim), dtype=dtype)
    if to_cuda:
        K_cache = K_cache.cuda()
        V_cache = V_cache.cuda()
    return K_cache, V_cache
