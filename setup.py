from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lm_ops',
    ext_modules=[
        CUDAExtension('lm_ops', [
            'attention_kernel.cu',
            'packed_attention_kernel.cu',
            'rotary_embedding.cu',
            'norm_kernel.cu',
            'pybind.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
