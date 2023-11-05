#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>
#include <vector>

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  return val;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
        return val;
}

template<typename T>
__device__ T blockReduceSum(T val){
    static __shared__ T shared[32];
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);
    if (lane_id == 0)
        shared[warp_id] = val;
    __syncthreads();

    val = shared[lane_id];
    __syncthreads();
    val = warpReduceSum(val);
    return val;
}

template<typename T>
__device__ T blockReduceMax(T val){
    static __shared__ T shared[32];
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warpReduceMax<T>(val);

    if (lane_id == 0)
        shared[warp_id] = val;
    __syncthreads();
    val = shared[lane_id];
    __syncthreads();
    val = warpReduceMax(val);
    return val;
}

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void naive_attention_forward_kernel(
    const int context_len,
    const int dim,
    scalar_t* __restrict__ Q,
    scalar_t* __restrict__ K,
    scalar_t* __restrict__ V,
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
    // Q has shape [batch_size, context_len, num_heads, dim]

    int batch_stride = context_len * num_heads * dim;
    int context_stride = num_heads * dim;
    int head_stride = dim;
    for(int i = thread_id; i < context_len; i += block_dim) {
        for(int j = 0; j < context_len; ++j) {
            for(int k = 0; k < dim; ++k) {
                int Q_idx = batch_id * batch_stride + head_id * head_stride + \
                            (num_heads * dim) * i + k;
                int K_idx = batch_id * batch_stride + head_id * head_stride + \
                            (num_heads * dim) * j + k;
                int S_idx = (num_heads * context_len * context_len) * batch_id + \
                            (context_len * context_len) * head_id + \
                            (context_len) * i + \
                            j;
                S[S_idx] += Q[Q_idx] * K[K_idx];
            }
        }
    }
    __syncthreads();
    float val_max;
    float val_sum;

    int idx_beg = (num_heads * context_len * context_len) * batch_id + (context_len * context_len) * head_id;
    for(int i = 0; i < context_len; ++i){
        // val_max = -1e09;
        val_sum = 1e-6;
        for(int j = thread_id; j < context_len; j += block_dim) {
            float exp_val = exp(S[idx_beg + context_len * i + j]);
            // val_max = max(val_max, exp_val);
            val_sum += exp_val;
        }
        __syncthreads();
        // why I can't reduce these two functions into single function and func pointer?
        // val_max = blockReduceMax<float>(val_max);
        val_sum = blockReduceSum<float>(val_sum);
        for(int j = thread_id; j < context_len; j += block_dim) {
            float exp_val = exp(S[idx_beg + context_len * i + j]);
            P[idx_beg + context_len * i + j] = exp_val / val_sum;
        }
        __syncthreads();
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
                int V_idx = batch_id * batch_stride + head_id * head_stride + \
                            (num_heads * dim) * j + k;
                int O_idx = batch_id * batch_stride + head_id * head_stride + \
                            (num_heads * dim) * i + k;
                O[O_idx] += P[P_idx] * V[V_idx];
            }
        }
    }
}

std::vector<torch::Tensor> naive_attention_forward(
    torch::Tensor &Q, // [batch_size, context_len, dim]
    torch::Tensor &K, // [batch_size, context_len, dim]
    torch::Tensor &V, // [batch_size, context_len, dim]
    int num_heads
) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    auto batch_size = Q.size(0);
    auto context_len = Q.size(1);
    auto dim = Q.size(2);
    assert(dim % num_heads == 0);

    auto _Q = Q.reshape({batch_size, context_len, num_heads, dim / num_heads});
    auto _K = K.reshape({batch_size, context_len, num_heads, dim / num_heads});
    auto _V = V.reshape({batch_size, context_len, num_heads, dim / num_heads});
    auto _S = torch::zeros({batch_size, num_heads, context_len, context_len}).cuda();
    auto _P = torch::zeros({batch_size, num_heads, context_len, context_len}).cuda();
    auto _O = torch::zeros_like(_Q);
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
                _Q.data_ptr<scalar_t>(),
                _K.data_ptr<scalar_t>(),
                _K.data_ptr<scalar_t>(),
                _S.data_ptr<scalar_t>(),
                _P.data_ptr<scalar_t>(),
                _O.data_ptr<scalar_t>()
            );
        })
    );

    auto O = _O.reshape({batch_size, context_len, dim});
    return {_S, _P, O};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_attention_forward", &naive_attention_forward, "forward");
}
