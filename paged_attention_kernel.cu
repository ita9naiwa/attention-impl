#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

#include "util.cuh"

template <typename scalar_t>
__global__ void paged_attention_forward_kernel(
    const int max_context_len,
    const int dim,
    const float scale,
    scalar_t* __restrict__ Q,       // [length, num_heads, dim]
    scalar_t* __restrict__ K,       // [length, num_heads, dim]
    scalar_t* __restrict__ V,       // [length, num_heads, dim]
    int* __restrict__ offsets, // [length]
    scalar_t* __restrict__ S,
    scalar_t* __restrict__ P,
    scalar_t* __restrict__ O
) {
    const int thread_id     = threadIdx.x;
    const int block_dim     = blockDim.x;
    const int block_id      = blockIdx.x;
    const int head_id       = blockIdx.y;
    const int batch_size    = gridDim.x;
    const int num_heads     = gridDim.y;

    const int beg_idx = (block_id == 0)? 0 : offsets[block_id - 1];
    const int end_idx = offsets[block_id];
    const int size = end_idx - beg_idx;
    // S, P have shape [batch_size, num_heads, max_context_len, max_context_len]
    // Q, K have shape [length, num_heads, dim]
    for(int i = thread_id; i < size; i += block_dim) {
        const int shifted_i = i + beg_idx;
        for(int j = 0; j < size; ++j) {
            const int shifted_j = j + beg_idx;
            int S_idx = (num_heads * max_context_len * max_context_len) * block_id + \
                        (max_context_len * max_context_len) * head_id + \
                        (max_context_len) * i + j;
            if (i >= j)
            {
                for(int k = 0; k < dim; ++k) {
                    int Q_idx = (dim * num_heads) * shifted_i + dim * head_id + k;
                    int K_idx = (dim * num_heads) * shifted_j + dim * head_id + k;
                    S[S_idx] += Q[Q_idx] * K[K_idx] * scale;
                }
            } else {
                S[S_idx] = -10000.0;
            }
        }
    }
    float val_sum;
    float val_max;
    const int idx_beg = (num_heads * max_context_len * max_context_len) * block_id + \
                        (max_context_len * max_context_len) * head_id;
    // batch_size, num_heads, max_context_len, max_context_len
    for (int i = 0; i < size; ++i) {
        val_max = 0;
        val_sum = 1e-9;
        for(int j = thread_id; j < size; j += block_dim) {
            float val = S[idx_beg + max_context_len * i + j];
            val_max = max(val, val_max);
        }
        val_max = blockReduceMax<float>(val_max);

        for(int j = thread_id; j < size; j += block_dim) {
            float exp_val = exp(S[idx_beg + max_context_len * i + j] - val_max);
            val_sum += exp_val;
        }
        __syncthreads();
        val_sum = blockReduceSum<float>(val_sum);
        for(int j = thread_id; j < size; j += block_dim) {
            float exp_val = exp(S[idx_beg + max_context_len * i + j] - val_max);
            P[idx_beg + max_context_len * i + j] = (exp_val / val_sum);
        }
    }
    for(int i = thread_id; i < size; i += block_dim) {
        const int shifted_i = beg_idx + i;
        for(int j = 0; j < size; ++j) {
            const int shifted_j = beg_idx + j;
            for(int k = 0; k < dim; ++k) {
                int P_idx = (num_heads * max_context_len * max_context_len) * block_id + \
                            (max_context_len * max_context_len) * head_id + \
                            (max_context_len) * i + j;
                int V_idx = (num_heads * dim) * shifted_j + dim * head_id + k;
                int O_idx = (num_heads * dim) * shifted_i + dim * head_id + k;
                O[O_idx] += P[P_idx] * V[V_idx];
            }
        }
    }
}

template <typename scalar_t>
__global__ void paged_kv_attention_forward_kernel(
    const int max_context_len,
    const int dim,
    const float scale,
    scalar_t* __restrict__ Q,               // [length, num_heads, dim]
    scalar_t* __restrict__ K,               // [length, num_heads, dim]
    scalar_t* __restrict__ V,               // [length, num_heads, dim]
    scalar_t* __restrict__ K_cache,         // [cache_size, num_heads, dim]
    scalar_t* __restrict__ V_cache,         // [cache_size, num_heads, dim]
    int* __restrict__ cache_indices,        // [length]
    int* __restrict__ offsets,              // [batch_size]
    double* __restrict__ S,               // [batch_size, num_heads, max_context_len + 1]
    double* __restrict__ P,               // [batch_size, num_heads, max_context_len + 1]
    scalar_t* __restrict__ O                // [length, num_heads, dim]
) {
    const int thread_id = threadIdx.x;
    const int block_dim = blockDim.x;
    const int block_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int batch_size = gridDim.x;
    const int num_heads = gridDim.y;

    const int beg_idx = (block_id == 0) ? 0 : offsets[block_id - 1];
    const int end_idx = offsets[block_id];
    const int size = end_idx - beg_idx;
    // S[i] = K_cache[i][j] * Q[j];
    // S has shape [batch_size, num_heads, max_context_len + 1]
    for(int i = thread_id; i < size; i += block_dim) {
        int S_idx = ((1 + max_context_len) * num_heads) * block_id + \
                    (1 + max_context_len) * head_id + i;
        for(int j = 0;j < dim; ++j) {
            int K_cache_idx = (dim) * head_id + (num_heads * dim) * cache_indices[beg_idx + i] + j;
            int Q_idx = (num_heads * dim) * block_id + dim * head_id + j;
            S[S_idx] += K_cache[K_cache_idx] * Q[Q_idx] * scale;
        }
    }
    double tmp = 0.0;
    for(int i = thread_id; i < dim; i += block_dim) {
        int Q_idx = (num_heads * dim) * block_id + dim * head_id + i;
        int K_idx = Q_idx;
        tmp += (double)Q[Q_idx] * (double)K[K_idx] * scale;
    }

    // S shape [batch_size, num_heads, max_context_len + 1]
    __syncthreads();
    S[(num_heads * (1 + max_context_len)) * block_id + (max_context_len + 1) * head_id + size] = blockReduceSum<double>(tmp);

    double val_max = 0;
    for(int i = thread_id; i < size + 1; i += block_dim) {
        int idx = ((1 + max_context_len) * num_heads) * block_id + (1 + max_context_len) * head_id + i;
        val_max = max(val_max, S[idx]);
    }
    __syncthreads();
    val_max = blockReduceMax<double>(val_max);
    double exp_sum = 1e-6;
    for(int i = thread_id; i < size + 1; i += block_dim) {
        int idx = ((1 + max_context_len) * num_heads) * block_id + (1 + max_context_len) * head_id + i;
        double exp_val = exp(S[idx] - val_max);
        exp_sum += exp_val;
    }
    __syncthreads();
    exp_sum = blockReduceSum<double>(exp_sum);
    for(int i = thread_id; i < size + 1; i += block_dim) {
        int idx = ((1 + max_context_len) * num_heads) * block_id + (1 + max_context_len) * head_id + i;
        double exp_val = exp(S[idx] - val_max);
        P[idx] = exp_val / (1e-6 + exp_sum);
    }
    __syncthreads();
    for (int j = thread_id; j < dim; j += block_dim) {
        for (int i = 0; i < size; ++i) {
            int O_idx = (num_heads * dim) * block_id + (dim) * head_id + j;
            int P_idx = (num_heads * (1 + max_context_len)) * block_id + \
                        ((1 + max_context_len)) * head_id + i;
            int V_idx = dim * head_id + (num_heads * dim) * cache_indices[beg_idx + i] + j;
            O[O_idx] += P[P_idx] * V_cache[V_idx];
        }
    }

    for (int j = thread_id; j < dim; j += block_dim) {
        int O_idx = (num_heads * dim) * block_id + (dim) * head_id + j;
        int P_idx = (num_heads * (1 + max_context_len)) * block_id + \
                    ((1 + max_context_len)) * head_id + size;
        int V_idx = (num_heads * dim) * block_id + (dim) * head_id + j;
        O[O_idx] += P[P_idx] * V[V_idx];
    }

}

std::vector<torch::Tensor> paged_attention_forward(
    torch::Tensor &Q,       // [length, dim]
    torch::Tensor &K,       // [length, dim]
    torch::Tensor &V,       // [length, dim]
    torch::Tensor &offsets,   // [length]
    int num_heads
) {
    // always perform diagonal masking
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    auto batch_size = offsets.size(0);
    auto dim = Q.size(1);
    assert(dim % num_heads == 0);
    auto options = torch::TensorOptions().dtype(Q.scalar_type()).device(torch::kCUDA);
    int max_context_len = offsets[0].item<int>();
    for (int i = 1; i < batch_size; ++i) {
        max_context_len = max(max_context_len, (offsets[i] - offsets[i - 1]).item<int>());
    }

    auto S = torch::zeros({batch_size, num_heads, max_context_len, max_context_len}, options);
    auto P = torch::zeros({batch_size, num_heads, max_context_len, max_context_len}, options);
    auto O = torch::zeros_like(V);
    const int threads = std::min(max_context_len, 1024);
    const dim3 blocks(batch_size, num_heads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    float scale = 1.0 / std::sqrt(float(dim) / num_heads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        Q.scalar_type(),
        "paged_attention_forward_kernel",
        ([&] {
            paged_attention_forward_kernel<<<blocks, threads, 0, stream>>>(
                max_context_len,
                dim / num_heads,
                scale,
                Q.data_ptr<scalar_t>(),
                K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                offsets.data_ptr<int>(),
                S.data_ptr<scalar_t>(),
                P.data_ptr<scalar_t>(),
                O.data_ptr<scalar_t>()
            );
        })
    );
    return {S, P, O};
}

std::vector<torch::Tensor> paged_kv_attention_forward(
    torch::Tensor &Q,                   // [batch_size, dim]
    torch::Tensor &K,                   // [batch_size, dim]
    torch::Tensor &V,                   // [batch_size, dim]
    torch::Tensor &K_cache,             // [num tokens, num_heads, dim]
    torch::Tensor &V_cache,             // [num tokens, num_heads, dim]
    torch::Tensor &cache_indices,       // [num total working indices]
    torch::Tensor &offsets,             // [batch_size]
    int num_heads
) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    CHECK_INPUT(K_cache); CHECK_INPUT(V_cache);

    auto batch_size = Q.size(0);
    auto dim = Q.size(1);
    assert(dim % num_heads == 0);

    auto options_for_S = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    auto options = torch::TensorOptions().dtype(Q.scalar_type()).device(torch::kCUDA);
    auto max_context_len = offsets[0].item<int>();
    for(int i = 1; i < batch_size; ++i) {
        max_context_len = max(max_context_len, (offsets[i] - offsets[i - 1]).item<int>());
    }

    auto S = torch::zeros({batch_size, num_heads, max_context_len + 1}, options_for_S);
    auto P = torch::zeros({batch_size, num_heads, max_context_len + 1}, options_for_S);
    auto O = torch::zeros_like(Q);

    const int threads = std::min((int) (max_context_len + 1), 1024);
    const dim3 blocks(batch_size, num_heads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    float scale = 1.0 / std::sqrt(float(dim) / num_heads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        Q.scalar_type(),
        "paged_kv_attention_forward_kernel",
        ([&] {
            paged_kv_attention_forward_kernel<<<blocks, threads, max_context_len, stream>>>(
                max_context_len,
                dim / num_heads,
                scale,
                Q.data_ptr<scalar_t>(),
                K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(),
                cache_indices.data_ptr<int>(),
                offsets.data_ptr<int>(),
                S.data_ptr<double>(),
                P.data_ptr<double>(),
                O.data_ptr<scalar_t>()
            );
        })
    );

    return {S, P, O};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_forward", &paged_attention_forward, "naive attention forward");
    m.def("paged_kv_attention_forward", &paged_kv_attention_forward, "kv forward");
}
