from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torch_ext',
    ext_modules=[
        CUDAExtension('torch_ext', [
            'torch_ext.cpp',
            'matrix_multiply.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })