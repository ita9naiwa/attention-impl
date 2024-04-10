#pragma once

#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> naive_attention(
    torch::Tensor &Q,       // [batch_size, context_len, dim]
    torch::Tensor &K,       // [batch_size, context_len, dim]
    torch::Tensor &V,       // [batch_size, context_len, dim]
    torch::Tensor &mask,    // [batch_size, context_len, context_len]
    int num_heads
);

std::vector<torch::Tensor> single_query_attention(
    torch::Tensor &Q,       // [batch_size, dim]
    torch::Tensor &K,       // [batch_size, dim]
    torch::Tensor &V,       // [batch_size, dim]
    torch::Tensor &K_cache, // [batch_size, context_len, dim]
    torch::Tensor &V_cache, // [batch_size, context_len, dim]
    int num_heads
);

std::vector<torch::Tensor> packed_attention(
    torch::Tensor &Q,       // [length, dim]
    torch::Tensor &K,       // [length, dim]
    torch::Tensor &V,       // [length, dim]
    torch::Tensor &offsets,   // [length]
    int num_heads
);

std::vector<torch::Tensor> kv_single_query_attention(
    torch::Tensor &Q,                   // [batch_size, dim]
    torch::Tensor &K,                   // [batch_size, dim]
    torch::Tensor &V,                   // [batch_size, dim]
    torch::Tensor &K_cache,             // [num tokens, num_heads, dim]
    torch::Tensor &V_cache,             // [num tokens, num_heads, dim]
    torch::Tensor &cache_indices,       // [num total working indices]
    torch::Tensor &offsets,             // [batch_size]
    int num_heads
);

std::vector<torch::Tensor> kv_multi_query_attention(
    torch::Tensor &Q,                   // [batch_size, num_queries, dim]
    torch::Tensor &K,                   // [batch_size, num_queries, dim]
    torch::Tensor &V,                   // [batch_size, num_queries, dim]
    torch::Tensor &K_cache,             // [num tokens, num_heads, dim]
    torch::Tensor &V_cache,             // [num tokens, num_heads, dim]
    torch::Tensor &cache_indices,       // [num total working indices]
    torch::Tensor &offsets,             // [batch_size]
    int num_heads
);

void rotary_embedding_inplace(
    torch::Tensor& Q,       // [input_size, num_heads * head_dim]
    torch::Tensor& K,         // [input_size, num_heads * head_dim]
    torch::Tensor& positions,   // [input_size]
    torch::Tensor& cos,         // [max_position, rot_half]
    torch::Tensor& sin,         // [max_position, rot_half]
    const int head_dim
);

void layernorm_inplace(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& bias,
    float eps);