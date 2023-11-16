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
                    int Q_idx = (dim * num_heads) * shifted_i + k;
                    int K_idx = (dim * num_heads) * shifted_j + k;
                    S[S_idx] += Q[Q_idx] * K[K_idx] * scale;
                }
            } else {
                S[S_idx] = -10000.0;
            }
        }
    }
    float val_sum;
    const int idx_beg = (num_heads * max_context_len * max_context_len) * block_id + \
                        (max_context_len * max_context_len) * head_id;
    for (int i = 0; i < max_context_len; ++i) {
        val_sum = 1e-9;
        for(int j = thread_id; j < size; j += block_dim) {
            float exp_val = exp(S[idx_beg + max_context_len * i + j]);
            val_sum += exp_val;
        }
        __syncthreads();
        val_sum = blockReduceSum<float>(val_sum);
        for(int j = thread_id; j < size; j += block_dim) {
            float exp_val = exp(S[idx_beg + max_context_len * i + j]);
            P[idx_beg + max_context_len * i + j] = (scalar_t)(exp_val / val_sum);
        }
    }
    for(int i = thread_id; i < max_context_len; i += block_dim) {
        const int shifted_i = beg_idx + i;
        for(int j = 0; j < max_context_len; ++j) {
            const int shifted_j = beg_idx + j;
            for(int k = 0; k < dim; ++k) {
                int P_idx = (num_heads * max_context_len * max_context_len) * block_id + \
                            (max_context_len * max_context_len) * head_id + \
                            (max_context_len) * i + j;
                int V_idx = (num_heads * dim) * shifted_j + k;
                int O_idx = (num_heads * dim) * shifted_i + k;
                printf("%0.4f %0.4f\n", P[P_idx], V[V_idx]);
                O[O_idx] += P[P_idx] * V[V_idx];
            }
        }
    }
}

template <typename scalar_t>
__global__ void paged_kv_attention_forward_kernel(
    const int context_len,
    const int dim,
    const float scale,
    scalar_t* __restrict__ Q,           // [batch_size, num_heads, dim]
    scalar_t* __restrict__ K,           // [batch_size, num_heads, dim]
    scalar_t* __restrict__ V,           // [batch_size, num_heads, dim]
    scalar_t* __restrict__ K_cache,     // [batch_size, context_len, num_heads, dim]
    scalar_t* __restrict__ V_cache,     // [batch_size, context_len, num_heads, dim]
    scalar_t* __restrict__ S,           // [batch_size, num_heads, context_len + 1]
    scalar_t* __restrict__ P,           // [batch_size, num_heads, context_len + 1]
    scalar_t* __restrict__ O            // [batch_size, num_heads, dim]
) {
    const int thread_id = threadIdx.x;
    const int block_dim = blockDim.x;
    const int batch_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int batch_size = gridDim.x;
    const int num_heads = gridDim.y;
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
    int max_context_len = 0;
    for (int i = 1; i < batch_size; ++i)
        max_context_len = max(max_context_len, (offsets[i] - offsets[i - 1]).item<int>());

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
    torch::Tensor &Q,       // [batch_size, dim]
    torch::Tensor &K,       // [batch_size, dim]
    torch::Tensor &V,       // [batch_size, dim]
    torch::Tensor &K_cache, // [batch_size, context_len, dim]
    torch::Tensor &V_cache, // [batch_size, context_len, dim]
    int num_heads
) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    CHECK_INPUT(K_cache); CHECK_INPUT(V_cache);

    auto batch_size = Q.size(0);
    auto dim = Q.size(1);
    auto context_len = K_cache.size(1);
    assert(dim % num_heads == 0);

    auto options = torch::TensorOptions().dtype(Q.scalar_type()).device(torch::kCUDA);

    auto S = torch::zeros({batch_size, num_heads, context_len + 1}, options);
    auto P = torch::zeros({batch_size, num_heads, context_len + 1}, options);
    auto O = torch::zeros({batch_size, dim}, options);

    const int threads = std::min((int) (context_len + 1), 1024);
    const dim3 blocks(batch_size, num_heads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    float scale = 1.0 / std::sqrt(float(dim) / num_heads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        Q.scalar_type(),
        "paged_kv_attention_forward_kernel",
        ([&] {
            paged_kv_attention_forward_kernel<<<blocks, threads, context_len, stream>>>(
                context_len,
                dim / num_heads,
                scale,
                Q.data_ptr<scalar_t>(),
                K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(),
                S.data_ptr<scalar_t>(),
                P.data_ptr<scalar_t>(),
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
