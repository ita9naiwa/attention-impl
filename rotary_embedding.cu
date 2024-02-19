#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "util.cuh"


template <typename scalar_t>
__global__ void rotary_embedding_inplace_kernel(
    scalar_t* Q,
    scalar_t* K,
    const int32_t* positions,
    scalar_t* __restrict__ cos,
    scalar_t* __restrict__ sin,
    const int input_size,
    const int num_heads,
    const int head_dim,
    const int rot_half
) {
    const int idx = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int rot_dim = rot_half * 2;
    const int pos = positions[idx];
    const scalar_t c = cos[pos * rot_half + (d % rot_half)];
    const scalar_t s = sin[pos * rot_half + (d % rot_half)];

    const int i = idx * num_heads * head_dim + h * head_dim;
    // I think this opeartion doesn't make branch divergence
    const int sign = d + rot_half >= rot_dim ? 1 : -1;
    Q[i + d] = Q[i + d] * c + sign * Q[i + (d + rot_half) % rot_dim] * s;
    K[i + d] = K[i + d] * c + sign * K[i + (d + rot_half) % rot_dim] * s;

}

// Neox(Llama) Style Rotary Embedding
// Q, K are modified. It's a inplace operation!
void rotary_embedding_inplace(
    torch::Tensor& Q,       // [input_size, num_heads * head_dim]
    torch::Tensor& K,         // [input_size, num_heads * head_dim]
    torch::Tensor& positions,   // [input_size]
    torch::Tensor& cos,         // [max_position, rot_half]
    torch::Tensor& sin,         // [max_position, rot_half]
    const int head_dim
) {
    CHECK_INPUT(Q); CHECK_INPUT(K);
    CHECK_INPUT(positions);
    CHECK_INPUT(cos); CHECK_INPUT(sin);

    auto input_size = Q.size(0);
    auto rot_half = cos.size(1);
    auto num_heads = Q.size(1) / head_dim;

    const dim3 blocks(input_size, num_heads);
    const dim3 threads(rot_half * 2);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        Q.scalar_type(),
        "rotary_embedding_inplace",
        ([&] {
            // Launch kernel
            rotary_embedding_inplace_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                Q.data_ptr<scalar_t>(),
                K.data_ptr<scalar_t>(),
                positions.data_ptr<int32_t>(),
                cos.data_ptr<scalar_t>(),
                sin.data_ptr<scalar_t>(),
                input_size,
                num_heads,
                head_dim,
                rot_half
            );
        })
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_embedding_inplace",
          &rotary_embedding_inplace,
          "rotary_embedding_inplace");
}
