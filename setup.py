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
        CUDAExtension('rotary_embedding', [
            'rotary_embedding.cu',
        ]),

    ],
    cmdclass={
        'build_ext': BuildExtension
    })
