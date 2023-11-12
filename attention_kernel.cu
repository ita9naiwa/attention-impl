#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>
#include <vector>

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
                    S[S_idx] += (Q[Q_idx] * K[K_idx]) / scale;
                }
            } else {
                S[S_idx] = -100000.0;
            }
        }
    }

    float val_sum;

    int idx_beg = (num_heads * context_len * context_len) * batch_id + (context_len * context_len) * head_id;
    for(int i = 0; i < context_len; ++i){
        // val_max = -1e09;
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

std::vector<torch::Tensor> naive_attention_forward(
    torch::Tensor &Q,       // [batch_size, context_len, dim]
    torch::Tensor &K,       // [batch_size, context_len, dim]
    torch::Tensor &V,       // [batch_size, context_len, dim]
    torch::Tensor &mask,    // [batch_size, context_len, context_len]
    int num_heads,
    float scale
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_attention_forward", &naive_attention_forward, "forward");
}
