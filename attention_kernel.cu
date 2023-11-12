#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// brought from vLLM code https://github.com/vllm-project/vllm/blob/main/csrc/reduction_utils.cuh
template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  return val;
}

template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    if(threadIdx.x == 0) {
        for(int i = 0; i < 32;++i){
            shared[i] = 0.0;
        }
    }
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;
    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();
    T ret = 0;
    for(int j = 0; j < 32;++j){
        ret += shared[j];
    }
  return ret;
}

template <typename scalar_t>
__global__ void naive_attention_forward_kernel(
    const int context_len,
    const int dim,
    const float scale,
    scalar_t* __restrict__ Q,
    scalar_t* __restrict__ K,
    scalar_t* __restrict__ V,
    scalar_t* __restrict__ mask,
    scalar_t* __restrict__ S,
    scalar_t* __restrict__ P,
    scalar_t* __restrict__ O
) {
    extern __shared__ float S_softmax[];
    const int thread_id     = threadIdx.x;
    const int block_dim     = blockDim.x;
    const int batch_id      = blockIdx.x;
    const int head_id       = blockIdx.y;
    const int batch_size    = gridDim.x;
    const int num_heads     = gridDim.y;

    // Q has shape      [batch_size, context_len, num_heads, dim]
    // K has shape      [batch_size, context_len, num_heads, dim]
    // mask has shape   [batch_size, context_len, context_len]
    for(int i = thread_id; i < context_len; i += block_dim) {
        for(int j = 0; j < context_len; ++j) {
            int S_idx = (num_heads * context_len * context_len) * batch_id + \
                        (context_len * context_len) * head_id + \
                        (context_len) * i + j;
            if (mask[(context_len * context_len) * batch_id + context_len * i + j] > 0){
                for(int k = 0; k < dim; ++k) {
                    int Q_idx = (context_len * num_heads * dim) * batch_id + \
                                (dim) * head_id + \
                                (num_heads * dim) * i + k;
                    int K_idx = (context_len * num_heads * dim) * batch_id + \
                                (dim) * head_id + \
                                (num_heads * dim) * j + k;
                    S[S_idx] += (Q[Q_idx] * K[K_idx]) * scale;
                }
            } else {
                S[S_idx] = -100000.0;
            }
        }
    }

    float val_sum;

    int idx_beg = (num_heads * context_len * context_len) * batch_id + (context_len * context_len) * head_id;
    for(int i = 0; i < context_len; ++i){
        val_sum = 1e-9;
        for(int j = thread_id; j < context_len; j += block_dim) {
            float exp_val = exp(S[idx_beg + context_len * i + j]);
            val_sum += exp_val;
        }
        __syncthreads();
        val_sum = blockReduceSum<float>(val_sum);
        for(int j = thread_id; j < context_len; j += block_dim) {
            float exp_val = exp(S[idx_beg + context_len * i + j]);
            P[idx_beg + context_len * i + j] = exp_val / val_sum;
        }
    }
    // O has shape [batch_size, context_len, num_heads, dim]
    // V has shape [batch_size, context_len, num_heads, dim]
    // P has shape [batch_size, num_heads, context_len, context_len]
    for(int i = thread_id; i < context_len; i += block_dim) {
        for(int j = 0; j < context_len; ++j) {
            for(int k = 0; k < dim; ++k) {
                int P_idx = (num_heads * context_len * context_len) * batch_id + \
                            (context_len * context_len) * head_id + \
                            (context_len) * i + j;
                int V_idx = (context_len * num_heads * dim) * batch_id + \
                            (dim) * head_id + \
                            (num_heads * dim) * j + k;
                int O_idx = (context_len * num_heads * dim) * batch_id + \
                            (dim) * head_id + \
                            (num_heads * dim) * i + k;
                O[O_idx] += P[P_idx] * V[V_idx];
            }
        }
    }
}

template <typename scalar_t>
__global__ void kv_attention_forward_kernel(
    const int context_len,
    const int dim,
    const float scale,
    scalar_t* __restrict__ Q,           // [batch_size, num_heads, dim]
    scalar_t* __restrict__ K,           // [batch_size, num_heads, dim]
    scalar_t* __restrict__ V,           // [batch_size, num_heads, dim]
    scalar_t* __restrict__ K_cache,     // [batch_size, context_len, num_heads,, dim]
    scalar_t* __restrict__ V_cache,     // [batch_size, context_len, num_heads, dim]
    scalar_t* __restrict__ S,           // [batch_size, num_heads, context_len + 1]
    scalar_t* __restrict__ P,           // [batch_size, num_heads, context_len + 1]
    scalar_t* __restrict__ O            // [batch_size, num_heads, dim]
) {
    extern __shared__ float shared[];

    const int thread_id = threadIdx.x;
    const int block_dim = blockDim.x;
    const int batch_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int batch_size = gridDim.x;
    const int num_heads = gridDim.y;
    // S = K_cache dot dim
    // S[i] = K_cache[i][j] * Q[j]
    for(int i = thread_id; i < context_len; i += block_dim) {
        int S_idx = ((1 + context_len) * num_heads) * batch_id + \
                    ((1 + context_len)) * head_id + i;
        for(int j = 0; j < dim; ++j) {
            int K_cache_idx = (context_len * num_heads * dim) * batch_id + \
                              (dim) * head_id + \
                              (num_heads * dim) * i + j;
            int Q_idx = (num_heads * dim) * batch_id + \
                        (dim) * head_id + j;
            S[S_idx] += K_cache[K_cache_idx] * Q[Q_idx];
        }
    }

    __syncthreads();
    // S[context_len] = K dot Q
    // 이거 너무 느린데 개선 할 방법, implicit에서 dot_product하는 부분을 참고해 볼까,
    // vllm에서는 어떻게 처리할까?
    // Q shape [batch_size, num_heads, dim]
    // K shape [batch_size, num_heads, dim]
    float tmp = 0.0;
    for(int i = thread_id; i < dim; i += block_dim) {
        int Q_idx = (num_heads * dim) * batch_id + (dim) * head_id + i;
        int K_idx = Q_idx;
        tmp += Q[Q_idx] * K[K_idx];
    }
    // S shape [batch_size, num_heads, context_len + 1]
    S[(num_heads * (1 + context_len)) * batch_id + (context_len + 1) * head_id + context_len] = blockReduceSum<float>(tmp);

    float exp_sum = 0;
    for(int i = thread_id; i < context_len + 1; i += block_dim) {
        int idx = ((1 + context_len) * num_heads) * batch_id + (1 + context_len) * head_id + i;
        float exp_val = exp(S[idx]);
        exp_sum += exp_val;
    }

    exp_sum = blockReduceSum<float>(exp_sum);
    for(int i = thread_id; i < context_len + 1; i += block_dim) {
        int idx = ((1 + context_len) * num_heads) * batch_id + (1 + context_len) * head_id + i;
        float exp_val = exp(S[idx]);
        P[idx] = exp_val / exp_sum;
    }

    // P shape [batch_size, num_heads, context_len + 1]
    // O shape [batch_size, num_heads, dim]
    // O = P dot V
    // O[batch_size][num_heads][dim] += P[batch_size][num_heads][i] * V[batch_size][num_heads]
    // V_cache shape [batch_size, context_len, num_heads, dim]
    for (int j = thread_id; j < dim; j += block_dim) {
        for (int i = 0; i < context_len; ++i) {
            int O_idx = (num_heads * dim) * batch_id + (dim) * head_id + j;
            int P_idx = (num_heads * (1 + context_len)) * batch_id + \
                        ((1 + context_len)) * head_id + i;
            int V_idx = (context_len * num_heads * dim) * batch_id + \
                        dim * head_id + \
                        (num_heads * dim) * i + j;
            O[O_idx] += P[P_idx] * V_cache[V_idx];
        }
    }
    for (int j = thread_id; j < dim; j += block_dim) {
        int O_idx = (num_heads * dim) * batch_id + (dim) * head_id + j;
        int P_idx = (num_heads * (1 + context_len)) * batch_id + \
                    ((1 + context_len)) * head_id + context_len;
        int V_idx = (num_heads * dim) * batch_id + (dim) * head_id + j;
        O[O_idx] += P[P_idx] * V[V_idx];
    }
}

std::vector<torch::Tensor> naive_attention_forward(
    torch::Tensor &Q,       // [batch_size, context_len, dim]
    torch::Tensor &K,       // [batch_size, context_len, dim]
    torch::Tensor &V,       // [batch_size, context_len, dim]
    torch::Tensor &mask,    // [batch_size, context_len, context_len]
    int num_heads
) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V); CHECK_INPUT(mask);
    auto batch_size = Q.size(0);
    auto context_len = Q.size(1);
    auto dim = Q.size(2);
    assert(dim % num_heads == 0);
    torch::Device device(torch::kCUDA);

    auto S = torch::zeros({batch_size, num_heads, context_len, context_len}, device);
    auto P = torch::zeros({batch_size, num_heads, context_len, context_len}, device);
    auto O = torch::zeros_like(Q);
    const int threads = std::min((int)context_len, 1024);
    const dim3 blocks(batch_size, num_heads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    float scale = 1.0 / std::sqrt(float(dim) / num_heads);

    AT_DISPATCH_FLOATING_TYPES(
        Q.scalar_type(),
        "naive_attention_forward_kernel",
        ([&] {
            naive_attention_forward_kernel<<<blocks, threads, context_len, stream>>>(
                context_len,
                dim / num_heads,
                scale,
                Q.data_ptr<scalar_t>(),
                K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                mask.data_ptr<scalar_t>(),
                S.data_ptr<scalar_t>(),
                P.data_ptr<scalar_t>(),
                O.data_ptr<scalar_t>()
            );
        })
    );
    return {S, P, O};
}

std::vector<torch::Tensor> kv_attention_forward(
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

    torch::Device device(torch::kCUDA);

    auto S = torch::zeros({batch_size, num_heads, context_len + 1}, device);
    auto P = torch::zeros({batch_size, num_heads, context_len + 1}, device);
    auto O = torch::zeros({batch_size, dim}, device);

    const int threads = std::min((int) (context_len + 1), 1024);
    const dim3 blocks(batch_size, num_heads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    float scale = 1.0 / std::sqrt(float(dim) / num_heads);

    AT_DISPATCH_FLOATING_TYPES(
        Q.scalar_type(),
        "kv_attention_forward_kernel",
        ([&] {
            kv_attention_forward_kernel<<<blocks, threads, context_len, stream>>>(
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
    m.def("naive_attention_forward", &naive_attention_forward, "naive attention forward");
    m.def("kv_attention_forward", &kv_attention_forward, "kv forward");
}
