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

// TODO: implement fused-matmul-activation with cutlass
//

template<typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L38
  const float f = (float) x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T) (f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template<typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float) (x * x * x);
  const T t = (T) tanhf((T) (0.79788456f * (float) (x + (T) (0.044715f * x3))));
  return ((T) 0.5) * x * (((T) 1.0) + t);
}

template<typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float) x;
  const T t = (T) tanhf(((T) (f * 0.79788456f)) * (((T) 1.0) + (T) (0.044715f * f) * x));
  return ((T) 0.5) * x * (((T) 1.0) + t);
}

template<typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T) (((float) x) / (1.0f + expf((float) -x)));
}
