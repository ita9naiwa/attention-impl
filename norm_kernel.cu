#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

#include "ops.h"
#include "util.cuh"

template <typename scalar_t>
__global__ void layernorm_inplace_kernel(
    scalar_t* input,
    scalar_t* weight,
    scalar_t* bias,
    float eps,
    int hidden_size
) {
    int thread_id = threadIdx.x;
    int block_id  = blockIdx.x;
    float mean_sum = 0.0f;
    float sq_sum = 0.0f;
    for(int i = thread_id; i < hidden_size; i += blockDim.x) {
        mean_sum += (float)(input[hidden_size * block_id + i]);
        sq_sum += (float)(input[hidden_size * block_id + i] * input[hidden_size * block_id + i]);
    }

    mean_sum = blockReduceSum(mean_sum) / hidden_size;
    sq_sum = blockReduceSum(sq_sum) / hidden_size;
    float s_var = rsqrtf(eps + sq_sum - mean_sum * mean_sum);
    // __syncthreads();
    for(int i = thread_id; i < hidden_size; i += blockDim.x) {
        // 여기 뭔 연산을 넣으면 커널이 뻗음... 왜지?
    }
    printf("done!\n");
}

void layernorm_inplace(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& bias,
    float eps) {

    int hidden_size = input.size(-1);
    int num_tokens = input.numel() / hidden_size;
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "layernorm_inplace_kernel",
        ([&] {
            layernorm_inplace_kernel<scalar_t><<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                eps,
                hidden_size);
        })
    );
}