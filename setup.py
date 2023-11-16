from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='attention',
    ext_modules=[
        CUDAExtension('attention', [
            'attention_kernel.cu',
        ]),
        CUDAExtension('paged_attention', [
            'paged_attention_kernel.cu',
        ]),

    ],
    cmdclass={
        'build_ext': BuildExtension
    })
