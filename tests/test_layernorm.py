import torch
from torch import nn

from lm_ops import layernorm_inplace

batch_size = 1
hidden_size = 8

x = torch.normal(mean=0, std=1.0, size=(batch_size, hidden_size))
ref_layernorm = nn.LayerNorm(hidden_size)

weight = ref_layernorm.weight.data
bias = ref_layernorm.bias.data
eps = ref_layernorm.eps
print(weight.shape, bias.shape)

y1 = ref_layernorm(x)
y2 = x.clone()
layernorm_inplace(y2, weight, bias, eps)
print(x)
print(y1)
print(y2)